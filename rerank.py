"""Stage 2 ranking: a cross-encoder reads (query, chunk) together.

The bi-encoder in search.py embeds query and document separately, which is why
it can precompute every document vector and still be fast. A cross-encoder sees
both texts in one forward pass, so it is much more accurate and much slower.
Hence the pattern: retrieve ~100 with hybrid search, then rerank those.

Blending, and why it exists
---------------------------
Reranking used to throw the retriever's ordering away and trust the
cross-encoder absolutely. On the materials slice that was measurably worse than
plain hybrid (MRR 0.783 vs 0.886) because the cross-encoder happily promotes a
related page (Netherite Sword, Netherite Upgrade) over the page the retriever
had already put first. So the final score is now a blend:

    blended = w * normalise(cross_encoder) + (1 - w) * normalise(hybrid)

with w = config.RERANK_BLEND_WEIGHT. The cross-encoder still supplies the
judgement, but when it is unsure the retriever's own order breaks the tie.

    python rerank.py "green thing that explodes near you"
"""

import argparse

import config
import search

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import CrossEncoder

        _MODEL = CrossEncoder(config.RERANK_MODEL)
    return _MODEL


def cross_encoder_scores(query, candidates):
    """Raw cross-encoder logits for (query, candidate) pairs."""
    if not candidates:
        return []
    return [float(s) for s in model().predict([(query, c["search_text"]) for c in candidates])]


def _normalise(values):
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo <= 0:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def blend_rankings(ce_scores, hybrid_scores, weight):
    """Combine normalised cross-encoder and hybrid scores."""
    ce_n = _normalise(ce_scores)
    hyb_n = _normalise(hybrid_scores)
    return [weight * a + (1.0 - weight) * b for a, b in zip(ce_n, hyb_n)]


def rerank_chunks(query, candidates, top_k=10, blend=None):
    if not candidates:
        return []
    weight = config.RERANK_BLEND_WEIGHT if blend is None else blend
    ce = cross_encoder_scores(query, candidates)
    hybrid = [float(c.get("score", 0.0)) for c in candidates]
    final = blend_rankings(ce, hybrid, weight)

    order = sorted(range(len(candidates)), key=lambda i: -final[i])
    results = []
    for rank, i in enumerate(order[:top_k], start=1):
        item = dict(candidates[i])
        item["score"] = float(final[i])
        item["rerank_score"] = float(ce[i])
        item["rank"] = rank
        results.append(item)
    return results


def candidates_for(query, retrieve_k=100, pool=200):
    """Hybrid candidates, one best chunk per page, with the version prior applied."""
    ranked = search.ranked(query, "hybrid", pool)
    return search.to_results(ranked, retrieve_k, per_page=True)


def search_with_rerank(query, k=10, retrieve_k=100, pool=200, blend=None):
    """Retrieve with hybrid, keep one best chunk per page, then blend-rerank."""
    return rerank_chunks(query, candidates_for(query, retrieve_k, pool), k, blend)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--retrieve-k", type=int, default=100)
    parser.add_argument("--blend", type=float, default=None)
    args = parser.parse_args()

    print("--- hybrid (pre-rerank) ---")
    for result in search.search(args.query, "hybrid", args.k):
        print(
            f"{result['rank']:>2}. {result['score']:.4f}  {result['page']} > {result['section']}"
        )
        snippet = " ".join(result["text"].split())[:160]
        print(f"     {snippet}...")

    print("--- after blended cross-encoder rerank ---")
    for result in search_with_rerank(args.query, args.k, args.retrieve_k, blend=args.blend):
        print(
            f"{result['rank']:>2}. {result['score']:+.4f} (ce {result['rerank_score']:+.4f})"
            f"  {result['page']} > {result['section']}"
        )
        snippet = " ".join(result["text"].split())[:160]
        print(f"     {snippet}...")


if __name__ == "__main__":
    main()
