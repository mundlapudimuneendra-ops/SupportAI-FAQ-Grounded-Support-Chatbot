"""Task 2 — TF-IDF + cosine similarity retriever for the SupportAI FAQ base.

This module is a strict extension of Task 1: it imports the FAQ data and the
``search_by_keyword`` helper from ``faq_search.py`` and adds a real ranking
algorithm on top. We deliberately avoid any external NLP library — TF-IDF is
a few dozen lines of stdlib, and a heavy dependency (e.g. scikit-learn,
sentence-transformers) would be overkill for a 7-entry knowledge base.

Public surface
--------------
- ``search_faqs(faqs, query, top_k=3)`` — the function the rest of SupportAI
  uses to fetch context. Falls back to the Task 1 keyword search if the
  query produces no TF-IDF hits but does produce keyword hits, so behaviour
  is never worse than Task 1.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

# Reuse Task 1 — the brief explicitly says "extend and reuse what you
# create here", not "rebuild from scratch".
from faq_search import FAQS as _DEFAULT_FAQS, search_by_keyword

# A token is a run of letters/digits; everything else is a separator. We
# lowercase and drop tokens shorter than 2 chars because single letters
# carry almost no signal at this corpus size.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split ``text`` into lower-cased word tokens of length >= 2."""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2]


def _doc_text(faq: dict) -> str:
    """Build the searchable text for one FAQ.

    We concatenate ``keywords`` (with the question repeated twice so it
    counts for more than a bare keyword), the ``question`` itself, and the
    ``category``. The answer is intentionally **not** included because we
    want the ranker to match on what a user is likely searching *for*, not
    on the long body of the answer.
    """
    keywords_text = " ".join(faq.get("keywords", []))
    question = faq.get("question", "")
    category = faq.get("category", "")
    # Repeat the question once so a query that closely paraphrases the
    # question naturally scores higher than one that only hits keywords.
    return f"{keywords_text} {question} {question} {category}"


def build_tfidf(docs: list[str]) -> tuple[list[dict[str, float]], dict[str, float], list[float]]:
    """Build TF-IDF vectors and document norms for a list of documents.

    Returns
    -------
    vectors : list[dict[str, float]]
        One sparse vector per document, mapping term -> tf-idf weight.
    idf : dict[str, float]
        Inverse document frequency for every term in the vocabulary.
    norms : list[float]
        L2 norm of each document vector (precomputed for cosine similarity).
    """
    n = len(docs)
    if n == 0:
        return [], {}, []

    tokenized_docs = [tokenize(d) for d in docs]

    # Document frequency: how many docs contain each term at least once.
    df: dict[str, int] = {}
    for tokens in tokenized_docs:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    # Smoothed IDF: log((N+1) / (df+1)) + 1 keeps the weight >= 1 and
    # avoids log(0) for terms that only appear in the query.
    idf: dict[str, float] = {
        term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()
    }

    vectors: list[dict[str, float]] = []
    norms: list[float] = []
    for tokens in tokenized_docs:
        # Term frequency with sub-linear scaling: 1 + log(tf) — dampens the
        # effect of a term that appears many times in one document.
        tf: dict[str, int] = {}
        for term in tokens:
            tf[term] = tf.get(term, 0) + 1

        vec = {term: (1.0 + math.log(count)) * idf[term] for term, count in tf.items()}
        norm = math.sqrt(sum(w * w for w in vec.values()))
        vectors.append(vec)
        norms.append(norm)

    return vectors, idf, norms


def vectorize(query: str, idf: dict[str, float]) -> dict[str, float]:
    """Turn a query string into a TF-IDF vector using the corpus IDF table."""
    tokens = tokenize(query)
    if not tokens:
        return {}
    tf: dict[str, int] = {}
    for term in tokens:
        tf[term] = tf.get(term, 0) + 1
    # Unknown terms get idf == 0; this is intentional so a query word that
    # appears in no FAQ cannot push up a random document's score.
    return {term: (1.0 + math.log(count)) * idf.get(term, 0.0) for term, count in tf.items()}


def cosine(v1: dict[str, float], n1: float, v2: dict[str, float], n2: float) -> float:
    """Cosine similarity between two sparse TF-IDF vectors.

    Iterating over the smaller vector's keys is the standard trick that
    keeps this O(min(|v1|, |v2|)) instead of O(|v1| * |v2|).
    """
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    if len(v1) > len(v2):
        v1, v2 = v2, v1
        n1, n2 = n2, n1
    dot = 0.0
    for term, weight in v1.items():
        if term in v2:
            dot += weight * v2[term]
    return dot / (n1 * n2)


def rank_faqs(
    faqs: Iterable[dict],
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """Rank ``faqs`` for ``query`` by TF-IDF cosine similarity.

    Returns up to ``top_k`` FAQs with a positive score, ordered by score
    descending. Each returned dict is augmented with a ``"score"`` key so
    callers can show confidence / debug.
    """
    faq_list = list(faqs)
    if not faq_list or not query or not query.strip():
        return []

    docs = [_doc_text(f) for f in faq_list]
    vectors, idf, norms = build_tfidf(docs)
    q_vec = vectorize(query, idf)
    q_norm = math.sqrt(sum(w * w for w in q_vec.values()))

    scored: list[tuple[float, dict]] = []
    for faq, vec, norm in zip(faq_list, vectors, norms):
        score = cosine(q_vec, q_norm, vec, norm)
        if score > 0:
            # Copy so we don't mutate the caller's FAQ dicts.
            enriched = dict(faq)
            enriched["score"] = score
            scored.append((score, enriched))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [faq for _, faq in scored[:top_k]]


def search_faqs(faqs: list[dict], query: str, top_k: int = 3) -> list[dict]:
    """High-level helper used by Tasks 3 and 4.

    Returns the top-K FAQs for ``query``. If TF-IDF produces nothing (e.g.
    the query is one short word that doesn't appear in any document's
    vocabulary), we fall back to Task 1's keyword search so the user
    always gets *some* answer when one exists in the corpus.
    """
    if not query or not query.strip():
        return []

    ranked = rank_faqs(faqs, query, top_k=top_k)
    if ranked:
        return ranked

    # Fallback: keyword search from Task 1, but only count "score" for it
    # so downstream code can still inspect / display it.
    keyword_hits = search_by_keyword(faqs, query)[:top_k]
    for i, faq in enumerate(keyword_hits):
        # Decay score so the TF-IDF wins still rank above the fallback.
        enriched = dict(faq)
        enriched["score"] = 0.0 - (i + 1) * 1e-6
        keyword_hits[i] = enriched
    return keyword_hits


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------
def _demo() -> None:
    """Print the top-K retrieval for the spec queries."""
    queries = [
        "forgot my password",
        "refund",
        "weather today",
    ]
    for query in queries:
        print(f"Query: {query}")
        results = search_faqs(_DEFAULT_FAQS, query, top_k=3)
        if not results:
            print("  No matching FAQs found.")
        for faq in results:
            score = faq.get("score", 0.0)
            print(f"  [{faq['category']}] {faq['question']}  (score={score:.4f})")
        print()


if __name__ == "__main__":
    _demo()
