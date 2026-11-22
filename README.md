# Minecraft semantic search and reranking

Four retrieval systems over a Minecraft wiki corpus, measured against each other
on labelled queries, a vague-description probe, and a deliberately ambiguous set.

| system | what it is |
|---|---|
| `bm25` | Okapi BM25 keyword scoring, implemented from scratch in `bm25.py` (no `rank_bm25`) |
| `dense` | bi-encoder embeddings (`BAAI/bge-base-en-v1.5`), cosine similarity |
| `hybrid` | BM25 + dense fused with weighted Reciprocal Rank Fusion |
| `hybrid+rerank` | hybrid candidates re-scored by a cross-encoder, blended with retrieval order |

## Corpus

Fetched from the live wiki through the MediaWiki API (`fetch_corpus.py`):
**2,418 pages → 13,820 chunks** across 9 categories, ~334 characters per chunk.
Wikitext is stripped of templates and tables and split by section, with a
`page > section` breadcrumb prefixed to every chunk so a fragment keeps its
context once embedded. 48 curated pages also carry a `version` field, which 369
chunks inherit.

## Headline results

57 labelled queries (`eval.py`, Recall@10 / MRR / nDCG@10):

| system | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| bm25 | 0.807 | 0.660 | 0.678 |
| dense | 0.895 | 0.816 | 0.820 |
| hybrid | 0.895 | 0.808 | 0.816 |
| **hybrid+rerank** | **0.930** | **0.804** | **0.822** |

The 30-query ambiguous set (`ambiguous_eval.py`, top-1 / top-3) tells a
different story: dense 0.30 / 0.60, hybrid+rerank 0.30 / 0.47, hybrid 0.20 / 0.50,
bm25 0.03 / 0.07. Both numbers are real; they measure different query
populations, and the second one is the honest one for a player describing what
they want instead of naming it.

Full tables, per-category breakdowns and per-query failure analysis are in
`results.md`.

## Findings

**Reranking must be blended, not trusted.** Sorting purely by cross-encoder
score discarded the retriever's ordering and made materials *worse* than plain
hybrid (MRR 0.783 vs 0.886). `rerank.blend_rankings()` normalises both score
lists and mixes them as `w·cross_encoder + (1−w)·hybrid`. `sweep_blend.py`
sweeps `w` from one cached scoring pass:

| w | overall MRR | nDCG | materials MRR |
|---|---|---|---|
| 0.4 | 0.804 | 0.822 | 0.947 |
| 0.6 | 0.785 | 0.809 | 0.882 |
| 1.0 (pure rerank) | 0.730 | 0.756 | 0.783 |

Pure reranking was the worst setting in the table. `w = 0.4` is the default.

**Storing version metadata is not enough.** Annotating pages changed nothing
until a query-time prior used it (`search.is_version_query` +
`_apply_version_prior`): for "when was X added" phrasing, chunks carrying a
version and matching the query title get a bonus scaled to the score span, so
the same code works for BM25, cosine and RRF. The version-sensitive bucket went
from 0.600–0.800 recall to 1.000 across all four systems.

**Dense wins on ambiguity; no single default wins both.** Named lookups favour
hybrid+rerank, vague descriptions favour dense on top-3. This would need
query-type routing, which is not implemented — the default stays on
`hybrid+rerank` because named queries are the common case.

**Chunking moves results more than model choice.** `MIN_CHUNK_CHARS`,
`MAX_CHUNK_CHARS` and section skipping change retrieval more than swapping
`EMBED_MODEL` does.

**Negative results worth keeping.** Sweeping `RRF_K` and `RRF_BM25_WEIGHT`
neither moved the vague-description probe nor beat the default on labelled MRR,
so fusion was not the bottleneck. Franchise-prefixed pages (`Story Mode:`,
`Dungeons:`) are 31% of titles but only 20–23% of retrieved top slots, i.e.
slightly *under*-retrieved — deprioritising them would not fix the ambiguous-set
failures.

## Setup

```bash
python -m pip install -r requirements.txt
python -m pip install sentence-transformers   # needed for dense + rerank
```

Run everything under the project venv (`.venv/Scripts/python.exe` on Windows).
`app.py` warns if it is launched with a different interpreter, because the dense
and rerank routes fail without `sentence-transformers`.

## Build

```bash
python fetch_corpus.py --limit 40     # smoke test
python fetch_corpus.py                # full CATEGORIES list from config.py
python chunk.py                       # -> data/chunks.jsonl
python index.py --skip-dense          # BM25 only
python index.py                       # + data/embeddings.npy (~30 min on CPU)
```

`data/pages.jsonl`, `data/embeddings.npy`, `data/embeddings_meta.json` and
`data/bm25.pkl` are gitignored build artifacts; `data/chunks.jsonl` is committed
so the corpus is inspectable without re-fetching.

## Use

```bash
python search.py "how do i make plants grow faster" --system bm25
python rerank.py "green thing that explodes near you"
python eval.py --systems bm25 dense hybrid hybrid+rerank --k 10   # -> results.md
python probe.py                                                    # vague descriptions
python ambiguous_eval.py                                           # top-1/top-3, mixed set
python ambiguous_eval.py --check-titles                             # validate labels
python sweep_blend.py                                               # rerank blend weight
python app.py                                                       # UI on :8000
```

Results in the UI link to the corresponding `minecraft.wiki` page.

## Layout

```
config.py           paths, categories, section filters, model names, tuning knobs
fetch_corpus.py     MediaWiki API -> data/pages.jsonl
populate_versions.py  hand-curated version map -> versioned pages
chunk.py            wikitext -> section chunks with breadcrumb prefix
bm25.py             dependency-free Okapi BM25
index.py            builds data/bm25.pkl and data/embeddings.npy
search.py           bm25 / dense / hybrid retrieval, version-aware prior
rerank.py           cross-encoder stage 2, blended with hybrid order
eval.py             Recall@k, MRR, nDCG@10 per system and per bucket -> results.md
probe.py            vague-description check: rank of the correct page per system
ambiguous_eval.py   strict top-1/top-3 accuracy on ambiguous mixed-category queries
sweep_blend.py      rerank blend-weight sweep from one cached scoring pass
app.py              stdlib UI with a system toggle and wiki links
queries.jsonl       57 labelled queries (all labels checked against the corpus)
results.md          generated tables plus the failure analysis
```

## Method notes and limitations

- **Label provenance.** All 57 labels were checked against the corpus
  (`eval.py` warns about any target page that does not exist). One label was
  corrected this way: "how do i stop my items from burning in lava" originally
  targeted `Lava`, but `Netherite Ingot > Overview` is the page that answers it
  ("immune to being destroyed by fire and lava"), so both are now accepted.
- **Label bias.** The query set, the labels, and the acceptable-answer lists in
  `ambiguous_eval.py` were all authored by the same person who built the
  systems. Where the ambiguous wording genuinely admits more than one answer,
  that choice is a judgement call and is visible in `ambiguous_eval.py`.
- **Version coverage is small.** The version map covers 48 of 2,418 pages, and
  the version-sensitive bucket has only 5 queries. The prior is gated narrowly
  on purpose so it cannot distort ordinary lookups, which also means it will not
  help "what did 1.20 change" style queries.
- **BM25 is hand-written and unvalidated against a reference.** It has not been
  compared against `rank_bm25`, so "is this BM25 correct" rests on the
  implementation rather than a test.
- **Known failures** are catalogued in `results.md`: Enderman for behavioural
  descriptions, `String` and `Diamond` when the question names the *use* rather
  than the item, and Potion of the Turtle Master for every system.
