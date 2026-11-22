"""The file that makes this a project instead of a tutorial: measure things.

    python eval.py                      # all four systems if their indexes exist
    python eval.py --systems bm25       # lexical only, no torch needed

Labels live in queries.jsonl, one object per line:

    {"query": "...", "bucket": "fuzzy-intent", "relevant": ["Creeper"],
     "seed": true, "category": "mobs"}

`relevant` is a list of PAGE titles (not chunk ids), so labels survive
re-chunking. `bucket` is the query style (exact-token / fuzzy-intent /
multi-hop / version-sensitive). `category` is the subject matter
(potions / materials / mobs / ...) and is reported separately, so a category
with only description-style queries is not hidden inside the fuzzy bucket.

Metrics are computed after collapsing results to one chunk per page, which is
also what the UI shows.
"""

import argparse
import datetime
import json
import math

import config

BUCKETS = ["exact-token", "fuzzy-intent", "multi-hop", "version-sensitive"]
ALL_SYSTEMS = ["bm25", "dense", "hybrid", "hybrid+rerank"]

CATEGORY_ORDER = [
    "potions",
    "materials",
    "mobs",
    "blocks",
    "tools",
    "redstone",
    "farming",
    "villagers",
    "version-sensitive",
]


def load_queries(path=None):
    path = path or (config.ROOT / "queries.jsonl")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_system(query, system, k):
    import search

    if system == "hybrid+rerank":
        import rerank

        return rerank.search_with_rerank(query, k=k)
    return search.search(query, system=system, k=k)


def returned_pages(results):
    """Ordered, de-duplicated page titles as the system ranked them."""
    pages, seen = [], set()
    for result in results:
        if result["page"] not in seen:
            seen.add(result["page"])
            pages.append(result["page"])
    return pages


def recall_at_k(pages, relevant, k):
    return 1.0 if set(pages[:k]) & set(relevant) else 0.0


def reciprocal_rank(pages, relevant):
    for i, page in enumerate(pages, start=1):
        if page in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(pages, relevant, k=10):
    gains = [1.0 if page in relevant else 0.0 for page in pages[:k]]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def group_metrics(per_query, key, order):
    """Mean metrics for each value of `key`, in `order` when known."""
    present = {row[key] for row in per_query if row.get(key)}
    keys = [k for k in order if k in present]
    keys += sorted(present - set(keys))

    def mean(rows, field):
        return sum(row[field] for row in rows) / len(rows) if rows else 0.0

    out = {}
    for name in keys:
        rows = [row for row in per_query if row.get(key) == name]
        if not rows:
            continue
        out[name] = {
            "n": len(rows),
            "recall": mean(rows, "recall"),
            "mrr": mean(rows, "mrr"),
            "ndcg": mean(rows, "ndcg"),
        }
    return out


def evaluate(queries, systems, k):
    import search as search_mod

    available_pages = {chunk["page"] for chunk in search_mod.chunks()}

    missing = {}
    for query in queries:
        for title in query["relevant"]:
            if title not in available_pages:
                missing.setdefault(title, []).append(query["query"])
    if missing:
        print("WARNING: labelled pages not found in the corpus:")
        for title in sorted(missing):
            print(f"  - {title!r}")
        print()

    summary = {}
    for system in systems:
        per_query = []
        for query in queries:
            try:
                results = run_system(query["query"], system, k)
            except (ImportError, RuntimeError, FileNotFoundError) as exc:
                print(f"{system}: unavailable ({exc})")
                per_query = None
                break
            pages = returned_pages(results)
            per_query.append(
                {
                    "bucket": query.get("bucket", "unknown"),
                    "category": query.get("category"),
                    "recall": recall_at_k(pages, query["relevant"], k),
                    "mrr": reciprocal_rank(pages, query["relevant"]),
                    "ndcg": ndcg_at_k(pages, query["relevant"], 10),
                }
            )
        if per_query is None:
            continue

        def mean(rows, field):
            return sum(row[field] for row in rows) / len(rows) if rows else 0.0

        summary[system] = {
            "n": len(per_query),
            "overall": {
                "recall": mean(per_query, "recall"),
                "mrr": mean(per_query, "mrr"),
                "ndcg": mean(per_query, "ndcg"),
            },
            "buckets": group_metrics(per_query, "bucket", BUCKETS),
            "categories": group_metrics(per_query, "category", CATEGORY_ORDER),
        }
    return summary


def print_table(summary, k):
    print(f"\nOverall over all labelled queries (Recall@{k})")
    print(f"{'system':<16}{'n':>4}{'Recall@'+str(k):>11}{'MRR':>8}{'nDCG@10':>9}")
    for system, data in summary.items():
        overall = data["overall"]
        print(
            f"{system:<16}{data['n']:>4}{overall['recall']:>11.3f}"
            f"{overall['mrr']:>8.3f}{overall['ndcg']:>9.3f}"
        )

    print("\nBy query bucket (query style)")
    for bucket in BUCKETS:
        rows = [(s, d["buckets"][bucket]) for s, d in summary.items() if bucket in d["buckets"]]
        if not rows:
            continue
        print(f"\n  {bucket}")
        for system, data in rows:
            print(
                f"    {system:<14} n={data['n']:<3} Recall@{k}={data['recall']:.3f}"
                f"  MRR={data['mrr']:.3f}  nDCG@10={data['ndcg']:.3f}"
            )

    print("\nBy category (subject matter)")
    for cat in CATEGORY_ORDER:
        rows = [(s, d["categories"][cat]) for s, d in summary.items() if cat in d["categories"]]
        if not rows:
            continue
        print(f"\n  {cat}")
        for system, data in rows:
            print(
                f"    {system:<14} n={data['n']:<3} Recall@{k}={data['recall']:.3f}"
                f"  MRR={data['mrr']:.3f}  nDCG@10={data['ndcg']:.3f}"
            )


def metric_table(group, k):
    lines = [f"| system | n | Recall@{k} | MRR | nDCG@10 |", "|---|---|---|---|---|"]
    for system, data in group.items():
        lines.append(
            f"| {system} | {data['n']} | {data['recall']:.3f} | "
            f"{data['mrr']:.3f} | {data['ndcg']:.3f} |"
        )
    return lines


def write_results_md(summary, k, n_queries, systems):
    lines = [
        "# Results",
        "",
        f"_Generated {datetime.date.today().isoformat()} by `eval.py`._",
        "",
        f"- Queries labelled: **{n_queries}**",
        f"- Systems: {', '.join(summary) or 'none'}",
        f"- Recall cut-off: @{k}",
        "",
        "## Overall",
        "",
        f"| system | n | Recall@{k} | MRR | nDCG@10 |",
        "|---|---|---|---|---|",
    ]
    for system, data in summary.items():
        overall = data["overall"]
        lines.append(
            f"| {system} | {data['n']} | {overall['recall']:.3f} | "
            f"{overall['mrr']:.3f} | {overall['ndcg']:.3f} |"
        )

    lines += ["", "## By query bucket", ""]
    for bucket in BUCKETS:
        rows = {s: d["buckets"][bucket] for s, d in summary.items() if bucket in d["buckets"]}
        if not rows:
            continue
        lines += [f"### {bucket}", ""] + metric_table(rows, k) + [""]

    lines += ["", "## By category", ""]
    for cat in CATEGORY_ORDER:
        rows = {s: d["categories"][cat] for s, d in summary.items() if cat in d["categories"]}
        if not rows:
            continue
        lines += [f"### {cat}", ""] + metric_table(rows, k) + [""]

    lines += [
        "",
        "## Tools category",
        "",
        "This section is written by hand, like the notes below; the tables above are generated.",
        "",
        "Tools is a subject-matter category in `queries.jsonl`, not a separate system or model. For each",
        "tool query the labelled set records the tool-related page titles that should appear, and `eval.py`",
        "reports Recall@K, MRR and nDCG@10 for tools the same way it does for potions, materials and mobs.",
        "",
        "What 'tool' means here: the bucket groups queries whose answer is a crafting/use tool or a tool",
        "recipe/upgrade path -- pickaxe, axe, shovel, hoe, sword, bow, fishing rod, shears, compass,",
        "clock, sickle-style harvesting, smithing-upgrade chains, and similar. It is separate from the",
        "materials bucket even when a tool query mentions materials, because the judged target is the tool",
        "page, not the raw material page.",
        "",
        "How to read a tools table: a high Recall@10 with a much lower nDCG@10 usually means the right",
        "tool appears somewhere in the top 10 but not near the top; a low Recall@10 means the tool is not",
        "being retrieved at all from this corpus. Because tools often share materials and recipes with other",
        "categories, weak tool results can come from either retrieval failure or from label/target ambiguity",
        "(for example a query that could be about a material or a tool).",
        "",
        "When a tool change is made, check three things in the tools section: (1) does Recall@10 move for",
        "the priority tool queries, (2) does MRR or nDCG@10 move, and (3) is any gain coming from one or two",
        "easy queries rather than the category as a whole. The category table is small, so a single query can",
        "swing the mean -- read it alongside the per-query failure list rather than as a precise measurement.",
        "",
        "## Notes",
        "",
        "Failure analysis is written by hand below; the tables above are generated.",
        "",
    ]
    config.RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {config.RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default=str(config.ROOT / "queries.jsonl"))
    parser.add_argument("--systems", nargs="*", default=ALL_SYSTEMS, choices=ALL_SYSTEMS)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    queries = load_queries(args.queries)
    if not queries:
        raise SystemExit("No labelled queries. Fill in queries.jsonl first.")

    seed = sum(1 for q in queries if q.get("seed"))
    if seed:
        print(f"NOTE: {seed}/{len(queries)} queries are marked seed -- verify the labels yourself.\n")

    summary = evaluate(queries, args.systems, args.k)
    if not summary:
        raise SystemExit("No system could be evaluated. Build indexes first (index.py).")

    print_table(summary, args.k)
    write_results_md(summary, args.k, len(queries), args.systems)


if __name__ == "__main__":
    main()
