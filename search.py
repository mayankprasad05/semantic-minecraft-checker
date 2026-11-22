"""Retrieval: BM25, dense, and hybrid via Reciprocal Rank Fusion.

    python search.py "how do i make plants grow faster"
    python search.py "Netherite Ingot" --system bm25 --k 3

Models are imported lazily, so `--system bm25` works without torch installed.
"""

import argparse
import json
import pickle
import re

import config

_CHUNKS = None
_BM25 = None
_MATRIX = None
_MODEL = None
_META = None


def chunks():
    global _CHUNKS
    if _CHUNKS is None:
        _CHUNKS = index_chunks()
    return _CHUNKS


def index_chunks():
    from index import load_chunks

    return load_chunks()


def bm25_available():
    try:
        bm25_index()
        return True
    except Exception:
        return False


def bm25_index():
    global _BM25
    if _BM25 is None:
        with open(config.BM25_PATH, "rb") as handle:
            payload = pickle.load(handle)
        if payload["n_docs"] != len(chunks()):
            raise RuntimeError(
                "BM25 index is stale: rebuild with `python index.py`. "
                f"index has {payload['n_docs']} docs, chunks.jsonl has {len(chunks())}"
            )
        _BM25 = payload["index"]
    return _BM25


def bm25_search(query, k=10):
    return bm25_index().search(query, k)


def dense_available():
    return config.EMB_PATH.exists()


def dense_resources():
    """Return (normalized matrix, embedding model); import torch only here."""
    global _MATRIX, _MODEL, _META
    if _META is None:
        _META = {}
        if config.EMB_META_PATH.exists():
            with open(config.EMB_META_PATH, encoding="utf-8") as handle:
                _META = json.load(handle)
    if _MATRIX is None:
        import numpy as np

        _MATRIX = np.load(config.EMB_PATH)
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(config.EMBED_MODEL)
        _MODEL.max_seq_length = config.EMBED_MAX_SEQ_LEN
    return _MATRIX, _MODEL


def dense_search(query, k=10):
    import numpy as np

    matrix, model = dense_resources()
    if matrix.shape[0] != len(chunks()):
        raise RuntimeError(
            "Embedding matrix is stale: rerun `python index.py`. "
            f"matrix has {matrix.shape[0]} rows, chunks.jsonl has {len(chunks())}"
        )
    built_with = (_META or {}).get("model")
    if built_with and built_with != config.EMBED_MODEL:
        raise RuntimeError(
            f"Embedding matrix was built with {built_with!r} but config.EMBED_MODEL "
            f"is {config.EMBED_MODEL!r}: rerun `python index.py`."
        )
    # bge wants its instruction prefix on the query side only; documents were
    # encoded without it. Other models are encoded as-is.
    query_text = config.EMBED_QUERY_PREFIX + query if config.EMBED_QUERY_PREFIX else query
    query_vec = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)[0]
    sims = matrix @ query_vec.astype(np.float32)
    top = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in top]


VERSION_QUERY_RE = re.compile(
    "(?<![a-z])("
    "version|versions|update|updates|patch|patches|snapshot|snapshots|"
    "added|introduced|released|came out|when did|when was"
    ")(?![a-z])",
    re.I,
)


def is_version_query(query):
    """True when the user is asking when something was added/changed.

    These are the only queries where a chunk's `version` metadata is evidence:
    for "which version added netherite" the item page and its derivative pages
    (Netherite Upgrade, Netherite Shovel) are equally on-topic, but only the
    item page carries an annotation saying it arrived in 1.16.
    """
    return bool(VERSION_QUERY_RE.search(query))


def _apply_version_prior(query, items):
    """Boost versioned chunks (and title matches) for version-style queries.

    Bonuses are scaled by the score span of the list, so the same code works
    for BM25, cosine and RRF scores without per-system tuning. Queries that are
    not version questions are returned unchanged.
    """
    if not items or not is_version_query(query):
        return items

    scores = [score for _idx, score in items]
    span = max(scores) - min(scores)
    if span <= 0:
        span = abs(scores[0]) or 1.0

    query_words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 3}
    all_chunks = chunks()

    boosted = []
    for idx, score in items:
        chunk = all_chunks[idx]
        bonus = 0.0
        if chunk.get("version"):
            bonus += config.VERSION_QUERY_BOOST * span
        if query_words:
            title_words = set(re.findall(r"[a-z0-9]+", chunk["page"].lower()))
            overlap = len(query_words & title_words) / len(query_words)
            bonus += config.VERSION_TITLE_BOOST * span * overlap
        boosted.append((idx, score + bonus))
    boosted.sort(key=lambda kv: (-kv[1], kv[0]))
    return boosted


def bm25_signal(query, pool=100):
    """Return a rough confidence that BM25 has a real lexical match.

    Used to decide whether BM25 should still count in the fused score.
    When the user is describing an item instead of naming it, BM25's top
    hits are often just loud pages that share a common noun, and fusing
    them at parity buries the dense result.
    """
    if not bm25_available():
        return 0.0
    ranked = bm25_search(query, pool)
    if not ranked:
        return 0.0
    top_page = chunks()[ranked[0][0]]["page"].lower()
    query_low = query.lower()
    if top_page in query_low:
        return 1.0
    if any(word in top_page.split() for word in query_low.split() if len(word) > 3):
        return 0.5
    return 0.0


def rrf(ranked_lists, k=None, weights=None):
    """Weighted Reciprocal Rank Fusion.

    score(doc) = sum over lists of weight * 1 / (k + rank)

    Weighting matters because fusion trusts both retrievers equally by default.
    When BM25 has no lexical match for an intent-style query ("the drink that
    lets you see in the dark") its list is noise, and unweighted RRF still hands
    those documents a full share of the score -- enough to bury the correct
    result that dense retrieval had at rank 1.
    """
    k = config.RRF_K if k is None else k
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    fused = {}
    for weight, ranked in zip(weights, ranked_lists):
        if weight <= 0:
            continue
        for rank, (doc_idx, _score) in enumerate(ranked, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def ranked(query, system="hybrid", pool=100):
    """Return [(chunk_idx, score), ...] for one system."""
    if system == "bm25":
        base = bm25_search(query, pool)
    elif system == "dense":
        base = dense_search(query, pool)
    elif system == "hybrid":
        base = rrf(
            [bm25_search(query, pool), dense_search(query, pool)],
            weights=[config.RRF_BM25_WEIGHT * bm25_signal(query, pool), 1.0],
        )
    else:
        raise ValueError(f"unknown system: {system!r} (use bm25, dense or hybrid)")
    return _apply_version_prior(query, base)


def to_results(ranked_items, k=10, per_page=True):
    """Map (idx, score) to chunk dicts, keeping only the best chunk per page.

    Keeps the highest-ranked chunk seen for each page so the result order
    reflects the retriever's intended page order, not the chunk order Python
    happened to materialize first.
    """
    all_chunks = chunks()
    best = {}  # page or id -> (rank, chunk, score)
    for rank, (doc_idx, score) in enumerate(ranked_items, start=1):
        chunk = all_chunks[doc_idx]
        key = chunk["page"] if per_page else chunk["id"]
        if key not in best or rank < best[key][0]:
            best[key] = (rank, chunk, score)
    results = []
    for rank, chunk, score in sorted(best.values(), key=lambda item: item[0]):
        result = dict(chunk)
        result["score"] = score
        result["rank"] = len(results) + 1
        results.append(result)
        if len(results) >= k:
            break
    return results


def search(query, system="hybrid", k=10, pool=100):
    return to_results(ranked(query, system, pool), k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--system", default="hybrid", choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pool", type=int, default=100)
    args = parser.parse_args()

    for result in search(args.query, args.system, args.k, args.pool):
        print(f"{result['rank']:>2}. {result['score']:.4f}  {result['page']} > {result['section']}")
        snippet = " ".join(result["text"].split())[:160]
        print(f"     {snippet}...")


if __name__ == "__main__":
    main()
