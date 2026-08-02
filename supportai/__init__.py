"""SupportAI — a small FAQ-grounded support chatbot.

The package is a thin composition layer over:

* ``faq_search`` — Task 1: FAQ data + keyword search helpers.
* ``supportai.retriever`` — Task 2: TF-IDF + cosine similarity ranking.
* ``supportai.llm`` — Task 3: Claude API + simulator fallback.
* ``supportai.chat`` — Task 4: CLI REPL with sliding-window memory.

Nothing here rebuilds Task 1 — every later task imports the original
``faq_search`` module per the brief's "extend and reuse" rule.
"""

# Lazy attribute access so ``python -m supportai.<submodule>`` doesn't
# double-import its target module (which would emit a RuntimeWarning).
__all__ = [
    "FAQS",
    "search_by_keyword",
    "get_faq_by_id",
    "get_faqs_by_category",
    "search_faqs",
    "rank_faqs",
    "generate_answer",
    "format_context",
    "ChatSession",
    "main",
]


def __getattr__(name):
    """Resolve public names on first access to avoid eager submodule imports."""
    if name in {"FAQS", "search_by_keyword", "get_faq_by_id", "get_faqs_by_category"}:
        from faq_search import (
            FAQS,
            search_by_keyword,
            get_faq_by_id,
            get_faqs_by_category,
        )
        return {
            "FAQS": FAQS,
            "search_by_keyword": search_by_keyword,
            "get_faq_by_id": get_faq_by_id,
            "get_faqs_by_category": get_faqs_by_category,
        }[name]
    if name in {"search_faqs", "rank_faqs"}:
        from supportai.retriever import search_faqs, rank_faqs
        return {"search_faqs": search_faqs, "rank_faqs": rank_faqs}[name]
    if name in {"generate_answer", "format_context"}:
        from supportai.llm import generate_answer, format_context
        return {
            "generate_answer": generate_answer,
            "format_context": format_context,
        }[name]
    if name in {"ChatSession", "main"}:
        from supportai.chat import ChatSession, main
        return {"ChatSession": ChatSession, "main": main}[name]
    raise AttributeError(f"module 'supportai' has no attribute {name!r}")
