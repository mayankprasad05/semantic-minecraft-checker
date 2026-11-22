# Results

_Generated 2026-11-22 by `eval.py`._

- Queries labelled: **57**
- Systems: bm25, dense, hybrid, hybrid+rerank
- Recall cut-off: @10

## Overall

| system | n | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| bm25 | 57 | 0.807 | 0.669 | 0.682 |
| dense | 57 | 0.895 | 0.825 | 0.817 |
| hybrid | 57 | 0.895 | 0.816 | 0.814 |
| hybrid+rerank | 57 | 0.930 | 0.812 | 0.822 |

## By query bucket

### exact-token

| system | n | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| bm25 | 29 | 1.000 | 0.930 | 0.947 |
| dense | 29 | 1.000 | 0.983 | 0.987 |
| hybrid | 29 | 1.000 | 0.983 | 0.987 |
| hybrid+rerank | 29 | 1.000 | 0.983 | 0.987 |

### fuzzy-intent

| system | n | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| bm25 | 20 | 0.450 | 0.191 | 0.251 |
| dense | 20 | 0.750 | 0.577 | 0.599 |
| hybrid | 20 | 0.750 | 0.577 | 0.599 |
| hybrid+rerank | 20 | 0.850 | 0.565 | 0.620 |

### multi-hop

| system | n | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| bm25 | 3 | 1.000 | 0.778 | 0.503 |
| dense | 3 | 0.667 | 0.667 | 0.365 |
| hybrid | 3 | 0.667 | 0.500 | 0.307 |
| hybrid+rerank | 3 | 0.667 | 0.500 | 0.333 |

### version-sensitive

| system | n | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| bm25 | 5 | 1.000 | 1.000 | 0.975 |
| dense | 5 | 1.000 | 1.000 | 0.975 |
| hybrid | 5 | 1.000 | 1.000 | 0.970 |
| hybrid+rerank | 5 | 1.000 | 1.000 | 0.970 |


## By category

### potions

| system | n | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| bm25 | 27 | 0.704 | 0.637 | 0.651 |
| dense | 27 | 0.852 | 0.798 | 0.811 |
| hybrid | 27 | 0.852 | 0.798 | 0.811 |
| hybrid+rerank | 27 | 0.889 | 0.776 | 0.802 |

### materials

| system | n | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| bm25 | 19 | 0.947 | 0.841 | 0.811 |
| dense | 19 | 1.000 | 0.974 | 0.913 |
| hybrid | 19 | 1.000 | 0.974 | 0.923 |
| hybrid+rerank | 19 | 1.000 | 0.947 | 0.915 |

### mobs

| system | n | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| bm25 | 11 | 0.818 | 0.450 | 0.535 |
| dense | 11 | 0.818 | 0.636 | 0.669 |
| hybrid | 11 | 0.818 | 0.591 | 0.633 |
| hybrid+rerank | 11 | 0.909 | 0.668 | 0.712 |


## Tools category

This section is written by hand, like the notes below; the tables above are generated.

Tools is a subject-matter category in `queries.jsonl`, not a separate system or model. For each
tool query the labelled set records the tool-related page titles that should appear, and `eval.py`
reports Recall@K, MRR and nDCG@10 for tools the same way it does for potions, materials and mobs.

What 'tool' means here: the bucket groups queries whose answer is a crafting/use tool or a tool
recipe/upgrade path -- pickaxe, axe, shovel, hoe, sword, bow, fishing rod, shears, compass,
clock, sickle-style harvesting, smithing-upgrade chains, and similar. It is separate from the
materials bucket even when a tool query mentions materials, because the judged target is the tool
page, not the raw material page.

How to read a tools table: a high Recall@10 with a much lower nDCG@10 usually means the right
tool appears somewhere in the top 10 but not near the top; a low Recall@10 means the tool is not
being retrieved at all from this corpus. Because tools often share materials and recipes with other
categories, weak tool results can come from either retrieval failure or from label/target ambiguity
(for example a query that could be about a material or a tool).

When a tool change is made, check three things in the tools section: (1) does Recall@10 move for
the priority tool queries, (2) does MRR or nDCG@10 move, and (3) is any gain coming from one or two
easy queries rather than the category as a whole. The category table is small, so a single query can
swing the mean -- read it alongside the per-query failure list rather than as a precise measurement.

## Notes

Failure analysis is written by hand below; the tables above are generated.
