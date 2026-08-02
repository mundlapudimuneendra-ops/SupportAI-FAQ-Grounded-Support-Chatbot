"""
SupportAI - FAQ Knowledge Base and Keyword Search
==================================================

This module provides:
1. A small FAQ knowledge base (list of dicts).
2. Three search helpers used by later SupportAI tasks:
   - search_by_keyword(faqs, query)
   - get_faq_by_id(faqs, faq_id)
   - get_faqs_by_category(faqs, category)
3. A demonstration block that runs the queries required by the spec.
"""

import re

# ---------------------------------------------------------------------------
# 1. FAQ Knowledge Base
# ---------------------------------------------------------------------------
# Each FAQ is a plain dict with: id, category, question, answer, keywords.
# Keeping the data as a list of dicts (rather than e.g. a dataclass) makes the
# structure easy to serialise to JSON in later tasks.
FAQS = [
    {
        "id": "faq-001",
        "category": "Account",
        "question": "How do I reset my password?",
        "answer": (
            "Click 'Forgot Password' on the login page. Enter your registered "
            "email address and check your inbox for a reset link valid for 24 hours."
        ),
        "keywords": ["password", "reset", "forgot", "login", "account", "email"],
    },
    {
        "id": "faq-002",
        "category": "Billing",
        "question": "What is your refund policy?",
        "answer": (
            "We offer full refunds within 30 days of purchase for unused "
            "subscriptions. Partial refunds are available for annual plans "
            "cancelled after 30 days."
        ),
        "keywords": ["refund", "money", "cancel", "billing", "subscription"],
    },
    {
        "id": "faq-003",
        "category": "Billing",
        "question": "How can I update my payment method?",
        "answer": (
            "Go to Settings > Billing > Payment Method and add a new card or "
            "PayPal account. The new method is used from your next billing cycle."
        ),
        "keywords": ["payment", "card", "paypal", "billing", "update", "method"],
    },
    {
        "id": "faq-004",
        "category": "Account",
        "question": "How do I change my account email address?",
        "answer": (
            "Open Settings > Account > Email, enter the new address, and confirm "
            "it using the verification link sent to your new inbox."
        ),
        "keywords": ["email", "change", "account", "update", "profile"],
    },
    {
        "id": "faq-005",
        "category": "Technical",
        "question": "The app is not loading, what should I do?",
        "answer": (
            "First, check our status page for any ongoing outages. Then try "
            "clearing your browser cache, disabling extensions, and reloading. "
            "If the issue persists, contact support with a screenshot."
        ),
        "keywords": ["loading", "error", "broken", "issue", "outage", "support"],
    },
    {
        "id": "faq-006",
        "category": "Privacy",
        "question": "How is my personal data handled?",
        "answer": (
            "We store only the data needed to provide the service and never "
            "sell it to third parties. You can request export or deletion at "
            "any time from Settings > Privacy."
        ),
        "keywords": ["privacy", "data", "personal", "gdpr", "delete", "export"],
    },
    {
        "id": "faq-007",
        "category": "Technical",
        "question": "Which browsers are supported?",
        "answer": (
            "SupportAI works on the latest two stable versions of Chrome, "
            "Firefox, Edge, and Safari. Internet Explorer is not supported."
        ),
        "keywords": ["browser", "chrome", "firefox", "edge", "safari", "support"],
    },
]


# ---------------------------------------------------------------------------
# 2. Search Functions
# ---------------------------------------------------------------------------
def _normalise(text: str) -> str:
    """Return ``text`` lower-cased and stripped of surrounding whitespace.

    Centralising the normalisation keeps the matching rules consistent across
    every helper in this module.
    """
    return text.lower().strip()


def _count_keyword_hits(faq: dict, query_words: list[str]) -> int:
    """Count how many of ``query_words`` appear in the FAQ's searchable text.

    We search across ``keywords`` (explicit search terms), ``question``, and
    ``category`` so a user can find a FAQ either by typing a known term or by
    describing the topic.

    Matching is done with a whole-word regex (``\\bword\\b``) so that e.g.
    searching for "password" does not also match "payment method".
    """
    haystack_fields = [
        " ".join(faq.get("keywords", [])),
        faq.get("question", ""),
        faq.get("category", ""),
    ]
    haystack = _normalise(" ".join(haystack_fields))

    hits = 0
    for word in query_words:
        if not word:
            continue
        # ``\b`` ensures we only count full words, avoiding substring false
        # positives such as "password" matching inside "payment".
        if re.search(rf"\b{re.escape(word)}\b", haystack):
            hits += 1
    return hits


def search_by_keyword(faqs: list[dict], query: str) -> list[dict]:
    """Search FAQs by matching query words against keywords, question, and category.

    Args:
        faqs: List of FAQ dicts to search.
        query: Free-text search query from the user.

    Returns:
        Matching FAQs ordered by the number of keyword hits (most hits first).
        If two FAQs tie on hits, the original order is preserved (stable sort).
    """
    if not query or not query.strip():
        return []

    # Split the query into individual words; each word is one search term.
    query_words = _normalise(query).split()

    # Build (faq, hit_count) pairs and keep only those with at least one hit.
    scored = []
    for faq in faqs:
        hits = _count_keyword_hits(faq, query_words)
        if hits > 0:
            scored.append((hits, faq))

    # Sort by hit count descending. Python's sort is stable, so the original
    # order is the tiebreaker for FAQs with the same number of hits.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [faq for _, faq in scored]


def get_faq_by_id(faqs: list[dict], faq_id: str) -> dict | None:
    """Return the FAQ with the given ``faq_id``, or ``None`` if not found.

    Matching is case-insensitive so callers can pass either ``"faq-001"`` or
    ``"FAQ-001"`` and get the same result.
    """
    target = _normalise(faq_id)
    for faq in faqs:
        if _normalise(faq.get("id", "")) == target:
            return faq
    return None


def get_faqs_by_category(faqs: list[dict], category: str) -> list[dict]:
    """Return all FAQs in ``category`` (case-insensitive)."""
    target = _normalise(category)
    return [faq for faq in faqs if _normalise(faq.get("category", "")) == target]


# ---------------------------------------------------------------------------
# 3. Demonstration
# ---------------------------------------------------------------------------
def _print_faq(faq: dict) -> None:
    """Pretty-print one FAQ in the format expected by the spec."""
    print(f"  [{faq['category']}] {faq['question']}")
    # Indent the answer so the arrow stays visually attached to the text.
    answer_lines = faq["answer"].splitlines() or [faq["answer"]]
    for i, line in enumerate(answer_lines):
        prefix = "  → " if i == 0 else "    "
        print(f"{prefix}{line}")


def _print_no_match() -> None:
    """Print the 'no match' message expected by the spec."""
    print("  No matching FAQs found.")


def demonstrate() -> None:
    """Run the demonstration queries required by the assignment brief."""
    test_queries = [
        "forgot my password",
        "refund",
        "weather today",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        results = search_by_keyword(FAQS, query)
        if results:
            for faq in results:
                _print_faq(faq)
        else:
            _print_no_match()
        print()  # blank line between queries


if __name__ == "__main__":
    demonstrate()
