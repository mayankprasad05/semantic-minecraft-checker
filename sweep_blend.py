"""Sweep the rerank blend weight without paying for the cross-encoder twice.

Cross-encoder scoring is the expensive part, so it is computed once per query
and every candidate weight is then evaluated from the cached scores.

    python sweep_blend.py
"""

import eval as E
import rerank

WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RETRIEVE_K = 100
POOL = 200


def page_order(cands, ce, weight, k=10):
    final = rerank.blend_rankings(ce, [float(c.get("score", 0.0)) for c in cands], weight)
    order = sorted(range(len(cands)), key=lambda i: -final[i])
    pages, seen = [], set()
    for i in order:
        p = cands[i]["page"]
        if p in seen:
            continue
        seen.add(p)
        pages.append(p)
        if len(pages) >= k:
            break
    return pages


def main():
    queries = E.load_queries()
    cached = []
    for n, q in enumerate(queries, 1):
        cands = rerank.candidates_for(q["query"], RETRIEVE_K, POOL)
        ce = rerank.cross_encoder_scores(q["query"], cands)
        cached.append((q, cands, ce))
        print(f"scored {n}/{len(queries)}: {q['query'][:50]}", flush=True)

    print("\nweight | overall            | potions            | materials          | mobs")
    print("       | R@10   MRR   nDCG  | R@10   MRR   nDCG  | R@10   MRR   nDCG  | R@10   MRR   nDCG")

    for w in WEIGHTS:
        rows = []
        for q, cands, ce in cached:
            pages = page_order(cands, ce, w)
            rows.append(
                {
                    "category": q.get("category"),
                    "bucket": q.get("bucket"),
                    "recall": E.recall_at_k(pages, q["relevant"], 10),
                    "mrr": E.reciprocal_rank(pages, q["relevant"]),
                    "ndcg": E.ndcg_at_k(pages, q["relevant"], 10),
                }
            )

        def agg(sub):
            if not sub:
                return (0.0, 0.0, 0.0)
            return (
                sum(r["recall"] for r in sub) / len(sub),
                sum(r["mrr"] for r in sub) / len(sub),
                sum(r["ndcg"] for r in sub) / len(sub),
            )

        o = agg(rows)
        p = agg([r for r in rows if r["category"] == "potions"])
        m = agg([r for r in rows if r["category"] == "materials"])
        mo = agg([r for r in rows if r["category"] == "mobs"])
        print(
            f"{w:>6.1f} | {o[0]:.3f} {o[1]:.3f} {o[2]:.3f}  | "
            f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}  | "
            f"{m[0]:.3f} {m[1]:.3f} {m[2]:.3f}  | "
            f"{mo[0]:.3f} {mo[1]:.3f} {mo[2]:.3f}",
            flush=True,
        )

    print("\nSWEEP DONE")


if __name__ == "__main__":
    main()
