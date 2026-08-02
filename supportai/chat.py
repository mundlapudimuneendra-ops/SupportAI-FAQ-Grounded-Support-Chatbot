"""Task 4 — SupportAI CLI chat.

A small REPL that wires together Task 2 (retrieval) and Task 3
(answer generation). The chat keeps a sliding window of the last
``MAX_TURNS`` user/assistant turns so follow-up questions like
"and how long is the link valid?" can be answered in context.
"""

from __future__ import annotations

import sys
from typing import Optional

from faq_search import FAQS
from supportai.retriever import search_faqs
from supportai.llm import generate_answer, _have_anthropic

# Three user/assistant pairs = six total turns. Anything older is dropped
# so the prompt stays small and cheap.
MAX_TURNS = 6

BANNER = """\
SupportAI — type your question and press Enter.
Type 'quit' or 'exit' to leave, or press Ctrl-C / Ctrl-D.
"""


class ChatSession:
    """One conversation: retrieval + LLM + sliding-window memory."""

    def __init__(self, model: Optional[str] = None, max_turns: int = MAX_TURNS) -> None:
        self.history: list[dict] = []
        self.model = model
        self.max_turns = max_turns

    def handle(self, user_input: str) -> tuple[str, list[dict]]:
        """Process one user turn.

        Returns the assistant's answer plus the retrieved FAQ list (so the
        CLI can show *why* the bot answered what it did).
        """
        question = user_input.strip()
        if not question:
            return "Please ask a question.", []

        # 1) Retrieve context (Task 2).
        retrieved = search_faqs(FAQS, question, top_k=3)

        # 2) Generate the answer (Task 3), passing the sliding window.
        #    We pass a *copy* of the history so the model can't mutate it.
        history_snapshot = list(self.history)
        answer = generate_answer(
            question,
            retrieved=retrieved,
            history=history_snapshot,
            model=self.model,
        )

        # 3) Update and trim the history.
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > self.max_turns:
            # Drop the oldest turns; keep the most recent ``max_turns``.
            self.history = self.history[-self.max_turns:]

        return answer, retrieved


# ---------------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------------
def _print_turn(answer: str, retrieved: list[dict]) -> None:
    """Display one assistant turn to the user."""
    # Show the retrieved FAQ count first so the user can sanity-check
    # whether the answer is grounded in real matches. Use ``FAQ(s)`` so
    # the format is identical for both branches.
    count = len(retrieved)
    if retrieved:
        ids = ", ".join(f"{f['id']} ({f['category']})" for f in retrieved)
        print(f"  [retrieved] {count} FAQ(s): {ids}")
    else:
        print(f"  [retrieved] 0 FAQ(s)")
    print(f"SupportAI: {answer}")


def main() -> None:
    """Entry point: print banner, then loop reading input until EOF / exit."""
    print(BANNER)
    mode = "real Claude API" if _have_anthropic() else "simulator (no ANTHROPIC_API_KEY)"
    print(f"[mode: {mode}]\n")

    session = ChatSession()

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        cmd = user_input.strip().lower()
        if cmd in {"quit", "exit", ":q"}:
            print("Goodbye!")
            return
        if cmd == "history":
            for turn in session.history:
                print(f"  {turn['role']}: {turn['content']}")
            print()
            continue
        if cmd == "reset":
            session.history.clear()
            print("Conversation cleared.\n")
            continue

        answer, retrieved = session.handle(user_input)
        _print_turn(answer, retrieved)
        print()


if __name__ == "__main__":
    try:
        main()
    except EOFError:
        # Reading input already caught EOF above; this is just a safety net
        # for direct module invocation in some shells.
        sys.exit(0)
