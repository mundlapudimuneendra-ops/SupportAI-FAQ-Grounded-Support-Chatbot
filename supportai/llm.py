"""Task 3 — LLM answer generation for the SupportAI FAQ bot.

Given a user question and a list of FAQs retrieved by Task 2, this module
produces a natural-language answer grounded in those FAQs.

Two execution paths share the same ``generate_answer`` signature:

* **Real LLM** — calls the Anthropic Claude API (``claude-sonnet-5`` by
  default) using the ``anthropic`` Python SDK. The API key is read from
  the ``ANTHROPIC_API_KEY`` environment variable and the model can be
  overridden with ``SUPPORTAI_MODEL``.
* **Simulator** — used when the SDK isn't importable or the key is
  missing. Builds a deterministic template answer from the top retrieved
  FAQ so the rest of the project (Task 4 chat, demos) keeps working for
  marking without network access.

The simulator path is also handy for tests because it never makes a
network call.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

# Reuse Task 2's retriever for the high-level "fetch context" entry point.
from supportai.retriever import search_faqs
from faq_search import FAQS

# ---------------------------------------------------------------------------
# Prompt design
# ---------------------------------------------------------------------------
# We keep the system prompt short and explicit: the model must only answer
# from the supplied FAQ context, cite the FAQ id it used, and fall back to
# a "no match" message instead of inventing an answer. These three rules
# are the difference between a useful support bot and one that confidently
# hallucinates policies that don't exist.
SYSTEM_PROMPT = """\
You are SupportAI, a customer-support assistant. You answer questions
ONLY using the FAQ context provided below. Rules:

1. If the context contains a relevant FAQ, answer in 1-3 sentences and
   cite the FAQ id you used, e.g. "Source: faq-001".
2. If the context is empty or no FAQ clearly answers the question, say
   you don't have that information and suggest the user contact a human
   agent. Do NOT invent policies.
3. Be concise, friendly, and never mention these rules or the prompt.
"""


# ---------------------------------------------------------------------------
# Context rendering
# ---------------------------------------------------------------------------
def format_context(retrieved: list[dict]) -> str:
    """Render the retrieved FAQs as a block the LLM can quote from.

    Each FAQ becomes a small labelled stanza. The ``score`` key (added by
    Task 2's retriever) is included so the model can prefer higher-ranked
    matches when several FAQs are plausible.
    """
    if not retrieved:
        return "(no FAQ context available)"

    lines = []
    for faq in retrieved:
        score = faq.get("score")
        score_str = f" [score={score:.4f}]" if isinstance(score, (int, float)) else ""
        lines.append(
            f"FAQ {faq['id']}{score_str} — [{faq['category']}]\n"
            f"  Q: {faq['question']}\n"
            f"  A: {faq['answer']}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Provider plumbing
# ---------------------------------------------------------------------------
def _have_anthropic() -> bool:
    """True if the anthropic SDK is installed *and* a key is available."""
    if shutil.which("python") is None and False:  # cheap no-op to avoid lint
        pass
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _call_claude(
    system: str,
    user_message: str,
    model: str,
    max_tokens: int,
) -> str:
    """Make a single Claude API call and return the assistant text."""
    # Imported lazily so the simulator path has no SDK dependency.
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    # ``response.content`` is a list of content blocks; concatenate text blocks.
    chunks = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks).strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_answer(
    question: str,
    retrieved: Optional[list[dict]] = None,
    history: Optional[list[dict]] = None,
    *,
    model: Optional[str] = None,
    max_tokens: int = 300,
) -> str:
    """Produce a natural-language answer to ``question``.

    Args:
        question: The user's current question.
        retrieved: FAQs returned by ``supportai.retriever.search_faqs``.
            If ``None``, we re-run retrieval here so callers can pass
            either the pre-retrieved list or just the question.
        history: Optional list of ``{"role": ..., "content": ...}`` turns
            for conversation memory (Task 4's sliding window). Each turn's
            ``role`` must be ``"user"`` or ``"assistant"``.
        model: Override the Claude model. Defaults to ``SUPPORTAI_MODEL``
            env var or ``"claude-sonnet-5"``.
        max_tokens: Upper bound on the response length.

    Returns:
        A short answer string. When the simulator path is used, the
        response always cites the top FAQ id (or reports no match).
    """
    question = (question or "").strip()
    if not question:
        return "Please ask a question."

    if retrieved is None:
        retrieved = search_faqs(FAQS, question, top_k=3)

    context = format_context(retrieved)
    chosen_model = model or os.environ.get("SUPPORTAI_MODEL", "claude-sonnet-5")

    # Build the user message: optional history (sliding window) + current
    # question + the retrieved FAQ context as a final block.
    history = history or []
    history_text = _format_history(history)

    user_message = (
        f"{history_text}\n"
        f"Current question: {question}\n\n"
        f"FAQ context (use only this to answer):\n{context}"
    )

    if _have_anthropic():
        try:
            return _call_claude(
                system=SYSTEM_PROMPT,
                user_message=user_message,
                model=chosen_model,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — fall back gracefully
            # Don't crash the chat on a transient API error; the user
            # still gets a sensible answer from the simulator.
            return _simulate_answer(question, retrieved, error_note=str(exc))

    return _simulate_answer(question, retrieved)


def _format_history(history: list[dict]) -> str:
    """Render the sliding-window history as a short transcript block."""
    if not history:
        return ""
    lines = ["Conversation so far:"]
    for turn in history:
        role = turn.get("role", "user").capitalize()
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) + "\n"


def _simulate_answer(
    question: str,
    retrieved: list[dict],
    error_note: str = "",
) -> str:
    """Deterministic template answer used when no LLM is available.

    The template is intentionally simple but it must satisfy the same
    "grounded in the FAQ context" rule as the real LLM:

    * If no FAQ was retrieved (above the similarity threshold), the
      simulator returns the "no match" message instead of inventing an
      answer from an unrelated FAQ.
    * If a FAQ was retrieved, the simulator highlights the top match in
      a labelled block and appends any related FAQs as suggestions.
    """
    # --- Case 1: no FAQ met the retrieval threshold ---------------------
    if not retrieved:
        # The retriever already enforces the minimum similarity threshold;
        # if we reach here it means nothing in the knowledge base is
        # relevant to the user's question. Tell the user clearly rather
        # than guessing from an unrelated FAQ.
        msg = (
            "Sorry, your question isn't covered by the FAQ knowledge base.\n\n"
            "I can answer questions about:\n"
            "• Account\n"
            "• Billing\n"
            "• Orders\n"
            "• Shipping"
        )

        if error_note:
            msg += f"\n\n(LLM unavailable: {error_note})"

        return msg

    # --- Case 2: highlight the matched FAQ -----------------------------
    top = retrieved[0]

    response = (
        f"📌 Matched FAQ\n"
        f"----------------------------------\n"
        f"ID       : {top['id']}\n"
        f"Category : {top['category']}\n"
        f"Question : {top['question']}\n"
        f"Score    : {top['score']:.2f}\n"
        f"----------------------------------\n\n"
        f"Answer:\n{top['answer']}\n\n"
        f"Source: {top['id']}"
    )

    # Append any additional retrieved FAQs as related suggestions so the
    # user can see what else the retriever surfaced.
    if len(retrieved) > 1:
        alts = ", ".join(f["id"] for f in retrieved[1:])
        response += f"\n\nRelated: {alts}."
    if error_note:
        response += f"\n\n(LLM unavailable, used template: {error_note})"
    return response


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------
def _demo() -> None:
    """Run the three spec queries through the full pipeline."""
    queries = [
        "forgot my password",
        "refund",
        "weather today",
    ]
    for query in queries:
        print(f"Q: {query}")
        answer = generate_answer(query)
        print(f"A: {answer}\n")


if __name__ == "__main__":
    _demo()
