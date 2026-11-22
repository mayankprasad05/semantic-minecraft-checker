# Minecraft Wiki Semantic Retrieval

**A comparative study of lexical, dense, hybrid and cross-encoder retrieval over a wiki corpus**

---

## 1. Problem statement

Searching a game wiki is a small but honest information-retrieval problem. Players often know *what a thing does* but not *what it is called*: "the drink that lets you see in the dark", "green thing that explodes near you", "how do i stop my items from burning in lava". Keyword search fails on that phrasing because none of the answer's words appear in the query. Semantic search handles it, but can be worse than exact matching when the player *does* name the item.

This project builds four retrieval systems over a corpus crawled from `minecraft.wiki` and measures them against each other, rather than assuming that "embeddings are better". The goal is not a product; it is a controlled answer to one question: **which retrieval strategy wins, on which kind of query, and by how much?**

---

## 2. Corpus construction

### 2.1 Fetching

Pages are pulled through the MediaWiki API rather than by scraping HTML (`fetch_corpus.py`). That choice matters for three reasons: the API returns clean wikitext instead of rendered chrome, one request can carry fifty page titles, and pagination through `categorymembers` gives a principled way to enumerate a topic instead of guessing page lists.

The crawler starts from nine top-level categories in `config.py` (Blocks, Items, Mobs, Biomes, Food, Villagers, Redstone, Tools, Potions), recurses one level into subcategories with a `seen_categories` guard so no category is crawled twice, deduplicates titles reached through several categories while recording every category that reached them, and then fetches the latest wikitext revision for each title in batches of 50.

### 2.2 Chunking

Chunks are **sections, not fixed token windows**. Wiki pages already carry meaningful boundaries (`== Obtaining ==`, `== Usage ==`), so splitting on headings preserves the unit a reader would actually cite.

Two preprocessing decisions did most of the work:

- **Noise sections are dropped** (`Trivia`, `Gallery`, `History`, `Sounds`, `Data values`, …). These are the classic source of "the search found a fun fact" failures.
- **Every chunk is prefixed with a breadcrumb**, `"<page> > <section>: <body>"`, before indexing and embedding. A section body often begins "It can be crafted from…" with no subject of its own; the breadcrumb restores the subject for both BM25 terms and the embedding.

Cleaning runs the wikitext regexes to a fixed point rather than once each, because markup nests — `[[File:x.png|thumb|A [[creeper]] in a field]]` only resolves after the inner link collapses. Gallery detection catches leftovers by counting image filenames and surviving pipes.

### 2.3 Corpus at a glance

| property | value |
|---|---|
| pages | 2,418 |
| chunks | 13,820 |
| mean chunk length | 334 characters |
| chunks carrying `version` | 369 (from 48 curated pages) |
| chunks by primary category | Items 5,140 · Blocks 4,940 · Mobs 2,275 · Biomes 813 · Villagers 321 · Redstone 294 · Food 37 |

The distribution is itself a finding: **`Tools` and `Potions` are page titles, not categories**, so they resolve to zero members and contribute no chunks. Potion, tool and material queries are therefore answered by `Items` and `Blocks` pages. This is exactly the kind of assumption that silently ruins a corpus, and the fetcher warns when a category returns nothing.

### 2.4 Version metadata

Forty-eight priority pages (potions, brewing materials, key materials, mobs) carry a manually curated `version` field — the earliest version in which the thing meaningfully existed. This is deliberately *not* a full version scrape; it exists to test whether metadata alone buys anything, which it does not (see §6.2).

---

## 3. The four systems

| system | what it is | cost |
|---|---|---|
| `bm25` | Okapi BM25 written from scratch in `bm25.py` (no `rank_bm25`), inverted index over `search_text` | milliseconds |
| `dense` | bi-encoder embeddings (`BAAI/bge-base-en-v1.5`, 768-dim, L2-normalised), cosine similarity | ~30 min to index, fast to query |
| `hybrid` | BM25 + dense fused with weighted Reciprocal Rank Fusion, plus a BM25-confidence gate and the version prior | slightly above dense |
| `hybrid+rerank` | 100 hybrid candidates re-scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`, **blended** with the hybrid order | ~1 s per query on CPU |

### 3.1 BM25 without a library

`bm25.py` implements the index directly: per-document term frequencies, document lengths, IDF over the inverted index, and the standard `k1`/`b` saturation and length normalisation. Writing it by hand rather than importing `rank_bm25` keeps the project dependency-free for the lexical path and makes the scoring inspectable, at the cost of having no reference implementation to validate against (see §7).

### 3.2 Dense retrieval

`bge-base-en-v1.5` is used with its instruction prefix on the **query side only** (`"Represent this sentence for searching relevant passages: "`). This is how the model was trained; applying the prefix to documents as well costs recall. Sequence length is capped at 256 tokens, which covers the vast majority of 334-character chunks.

Both indexes store a `n_docs`/`n` count and the embedding metadata stores the model name. At query time the retriever raises a loud error if the index row count or the model name no longer matches `chunks.jsonl` or `config.EMBED_MODEL`, instead of silently returning nonsense from a stale index.

### 3.3 Hybrid fusion with a confidence gate

Naive RRF treats both retrievers as equally trustworthy. For a descriptive query ("the drink that lets you see in the dark") BM25 returns loud pages that merely share a noun, and unweighted fusion hands those documents a full share of the score — enough to bury a correct dense hit at rank 1.

Two mechanisms address that:

- **Weighted RRF.** `score(doc) = Σ weight · 1/(k + rank)`, with `k = 60`.
- **A BM25 signal gate.** `bm25_signal()` returns 1.0 when the top BM25 page title appears in the query, 0.5 on a partial title-word overlap, and 0.0 otherwise; that value multiplies BM25's fusion weight. When the user is describing an item rather than naming it, BM25 effectively drops out of the fusion.

### 3.4 Version-aware ranking

Storing `version` on a page changes nothing by itself. The metadata is only consumed by a narrowly gated prior (`is_version_query` + `_apply_version_prior`): when the query contains version/question words (`version`, `update`, `patch`, `snapshot`, `added`, `introduced`, `released`, `when did`, `when was`), chunks that carry a `version` get a bonus, and so do chunks whose page title overlaps the query.

Crucially, both bonuses are **scaled by the score span of the candidate list**, so one implementation works for BM25 scores, cosine similarities and RRF scores without per-system tuning. Non-version queries pass through untouched.

### 3.5 Blended reranking

The cross-encoder sees query and chunk in one forward pass, so it is much more accurate than a bi-encoder — and much slower, which is why it only ever sees 100 pre-retrieved candidates.

The first implementation sorted purely by cross-encoder logit and was **worse than plain hybrid on materials** (MRR 0.783 vs 0.886). The cross-encoder happily promotes a related page (`Netherite Sword`, `Netherite Upgrade`) over the item page the retriever had already ranked first. The fix is to blend normalised scores:

```
blended = w · normalise(cross_encoder) + (1 − w) · normalise(hybrid)
```

with `w = 0.4`. The cross-encoder supplies the judgement; when it is unsure, the retriever's own ordering breaks the tie.

---

## 4. Evaluation methodology

### 4.1 Label set

Labels live in `queries.jsonl`, one JSON object per line, and reference **page titles, not chunk ids**, so they survive re-chunking.

| property | value |
|---|---|
| labelled queries | 57 |
| by style (`bucket`) | exact-token 29 · fuzzy-intent 20 · version-sensitive 5 · multi-hop 3 |
| by subject (`category`) | potions 27 · materials 19 · mobs 11 |
| mean relevant pages per query | 1.12 |

The two axes are deliberately orthogonal. `bucket` describes *how the query is phrased*; `category` describes *what it is about*. Reporting them separately stops a subject-matter category that happens to be phrased descriptively from disappearing inside the fuzzy bucket.

### 4.2 Metrics

All metrics are computed **after collapsing results to one chunk per page**, which is also what the UI shows, so the measurement matches the user experience.

- **Recall@10** — did any relevant page appear in the top 10.
- **MRR** — reciprocal rank of the first relevant page.
- **nDCG@10** — position-weighted, with binary gains and an ideal ranking sized to the number of relevant pages.

### 4.3 Three complementary evaluations

A single labelled set rewards named lookups and hides everything else, so three populations are measured:

1. **`eval.py`** — the 57 labelled queries, full breakdown by bucket and category, writes `results.md`.
2. **`probe.py`** — a small set of purely vague natural-language descriptions, reporting *the rank of the correct page* per system. This is the qualitative check that a metric can miss.
3. **`ambiguous_eval.py`** — 30 deliberately ambiguous mixed-category queries scored with strict top-1 / top-3 accuracy and an accepted-answer list per query.

### 4.4 Label hygiene

`eval.py` warns about any labelled page that does not exist in `data/chunks.jsonl`. That check already corrected one label: "how do i stop my items from burning in lava" originally targeted `Lava`, but the page that actually answers it is `Netherite Ingot` ("immune to being destroyed by fire and lava"), so both are now accepted.

One caveat worth stating plainly: **a query whose labelled pages are absent from the corpus is not skipped — it is scored zero and stays in every mean.** Missing labels therefore depress results silently apart from the warning, which is why label validation happens before any comparison is trusted.

---

## 5. Results

### 5.1 Overall (57 queries, Recall@10)

| system | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| bm25 | 0.807 | 0.669 | 0.682 |
| dense | 0.895 | **0.825** | 0.817 |
| hybrid | 0.895 | 0.816 | 0.814 |
| **hybrid+rerank** | **0.930** | 0.812 | **0.822** |

The headline is that **reranking buys recall, not rank quality**. It lifts Recall@10 from 0.895 to 0.930 — it finds the right page more often by pulling it in from the retrieve-100 pool — while its MRR is slightly *below* Dense's 0.825. That is the blend doing its job: it does not panic and reorder what was already correct.

BM25 is the clear floor on every metric. On a corpus of wiki prose that is expected, and it is the reference that makes the other numbers meaningful.

### 5.2 By query style

| bucket | n | system | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| exact-token | 29 | bm25 | 1.000 | 0.930 | 0.947 |
| | | dense | 1.000 | 0.983 | 0.987 |
| | | hybrid | 1.000 | 0.983 | 0.987 |
| | | rerank | 1.000 | 0.983 | 0.987 |
| fuzzy-intent | 20 | bm25 | 0.450 | 0.191 | 0.251 |
| | | dense | 0.750 | 0.577 | 0.599 |
| | | hybrid | 0.750 | 0.577 | 0.599 |
| | | rerank | **0.850** | 0.565 | **0.620** |
| multi-hop | 3 | bm25 | 1.000 | **0.778** | **0.503** |
| | | dense | 0.667 | 0.667 | 0.365 |
| | | hybrid | 0.667 | 0.500 | 0.307 |
| | | rerank | 0.667 | 0.500 | 0.333 |
| version-sensitive | 5 | bm25 | 1.000 | 1.000 | 0.975 |
| | | dense | 1.000 | 1.000 | 0.975 |
| | | hybrid | 1.000 | 1.000 | 0.970 |
| | | rerank | 1.000 | 1.000 | 0.970 |

Three things stand out.

**Fuzzy-intent is where the systems actually differ.** BM25 loses half its queries outright (recall 0.450, MRR 0.191): when the words are not on the page, keywords cannot help. Dense recovers to 0.750 and rerank to 0.850. This is the bucket the project exists to measure, and the gap is large and consistent.

**Exact-token saturates and then Dense wins it.** Every system gets recall 1.000, but BM25's MRR is 0.930 against 0.983 for the others. Even for named lookups, the embedding model ranks the right page higher than lexical matching does — because the breadcrumb-prefixed chunk and the surrounding prose give it more to match against than the title string alone.

**Multi-hop is the weak bucket for every system, and BM25 is the best of them.** With only three queries this is anecdote, not measurement, but the direction is intuitive: multi-hop questions need evidence from two pages, and both dense and cross-encoder scoring are built to pick *one* best passage. BM25's literal term matching accidentally behaves better when the query names both steps.

### 5.3 By subject category

| category | n | bm25 | dense | hybrid | rerank |
|---|---|---|---|---|---|
| potions | 27 | 0.704 / 0.637 / 0.651 | 0.852 / 0.798 / 0.811 | 0.852 / 0.798 / 0.811 | **0.889** / 0.776 / 0.802 |
| materials | 19 | 0.947 / 0.841 / 0.811 | 1.000 / 0.974 / 0.913 | 1.000 / 0.974 / **0.923** | 1.000 / 0.947 / 0.915 |
| mobs | 11 | 0.818 / 0.450 / 0.535 | 0.818 / 0.636 / 0.669 | 0.818 / 0.591 / 0.633 | **0.909** / **0.668** / **0.712** |

*(cells are Recall@10 / MRR / nDCG@10)*

Potions are the hardest category and the only one where reranking's recall gain survives without an MRR loss worth worrying about. Materials are nearly solved by dense retrieval — 1.000 recall, 0.923 nDCG for plain hybrid — which is precisely why the *pure* rerank regression (0.783 MRR) was so visible: it broke the one slice that was already working. Mobs are the category where rerank is clearly best on every metric, because mob queries are the most descriptive ("green thing that explodes near you").

---

## 6. Findings

### 6.1 Reranking must be blended, not trusted

The strongest result in the project is a negative one. Sorting purely by cross-encoder score was **worse than doing nothing**:

| blend weight | overall MRR | nDCG@10 | materials MRR |
|---|---|---|---|
| 0.4 (default) | **0.804** | **0.822** | **0.947** |
| 0.6 | 0.785 | 0.809 | 0.882 |
| 1.0 (pure rerank) | 0.730 | 0.756 | 0.783 |

The sweep is monotonic, which is the point: every step toward trusting the cross-encoder absolutely makes the system worse. `sweep_blend.py` computes all of these from one cached scoring pass, so the tuning cost is one encoding plus a table lookup.

A cross-encoder is trained on passage-level relevance; it has no view of the retriever's global ranking, and it will confidently promote a sibling page. Blending is not a hedge, it is a correction for a structural blind spot.

### 6.2 Storing version metadata is not enough

Annotating 48 pages changed nothing measurable until a query-time prior consumed it. With the prior, the version-sensitive bucket went from ~0.600–0.800 recall to **1.000 across all four systems**, and MRR to 1.000. The metadata was never the missing piece; the *routing* was — recognising that a query is asking *when*, and that in that case a chunk's `version` field is evidence rather than decoration.

Scaling bonuses by the score span is what makes this safe: the same prior works on BM25, cosine and RRF scores without any per-system constants, and because it is gated on version phrasing, ordinary lookups are untouched.

### 6.3 No single default wins both query populations

The 30-query ambiguous set (`ambiguous_eval.py`, strict top-1 / top-3):

| system | top-1 | top-3 |
|---|---|---|
| dense | **0.30** | **0.60** |
| hybrid+rerank | **0.30** | 0.47 |
| hybrid | 0.20 | 0.50 |
| bm25 | 0.03 | 0.07 |

Dense and rerank tie on top-1, but dense is clearly better at top-3. On the labelled set the ordering is reversed: rerank wins recall and nDCG, dense wins MRR. Both numbers are real — they describe different query populations — and the ambiguous one is the honest measure for a player describing what they want instead of naming it.

The correct fix would be **query-type routing** (detect a purely descriptive query and skip the lexical branch entirely). This is not implemented; the UI default stays on `hybrid+rerank` because named queries are the common case, and that limitation is stated rather than hidden.

### 6.4 Chunking moves results more than model choice

Changing `MIN_CHUNK_CHARS`, `MAX_CHUNK_CHARS` or the section skip list moves retrieval more than swapping `EMBED_MODEL` does. This is the least glamorous finding and probably the most useful: most of the quality lives in how the corpus is cut up, not in which checkpoint is loaded.

### 6.5 Negative results worth keeping

- **Fusion was not the bottleneck.** Sweeping `RRF_K` (10/30/60/100) and `RRF_BM25_WEIGHT` (0.25–1.0) neither improved the vague-description probe nor beat the default on labelled MRR. Deweighting BM25 to 0.25 cost 0.09 MRR. The knobs remain in `config.py` because the sweep is itself a result.
- **Franchise prefixes are not the problem.** Pages prefixed `Story Mode:` or `Dungeons:` are 31% of titles but only 20–23% of retrieved top slots — they are slightly *under*-retrieved, so deprioritising them would not fix the ambiguous-set failures.

---

## 7. Limitations

- **Label bias.** The queries, the labels and the accepted-answer lists were all authored by the same person who built the systems. Where ambiguous wording genuinely admits more than one answer, the choice is a judgement call and is visible in `ambiguous_eval.py`.
- **Small evaluation sets.** 57 labelled queries, 3 multi-hop queries, 5 version-sensitive queries, 30 ambiguous queries. A single query can swing a category mean; the category tables should be read alongside the per-query failure list, not as precise measurements.
- **Version coverage is narrow by design.** 48 of 2,418 pages and a prior gated on version phrasing. It will not help "what did 1.20 change" style questions.
- **BM25 is unvalidated against a reference implementation.** It has not been compared against `rank_bm25`, so correctness rests on the code rather than a test.
- **Embedding truncation.** At `EMBED_MAX_SEQ_LEN = 256`, 1,022 chunks (7.25%) are truncated, leaving 94,260 tokens never encoded. Raising the cap to 512 cuts that to 127 chunks (0.92%) at roughly double the encode time — not done here because the measured gain did not justify a 30-minute rebuild.
- **Known failures**, catalogued in `results.md`: `Enderman` for behavioural descriptions, `String` and `Diamond` when the question names the *use* rather than the item, and `Potion of the Turtle Master` for every system.
- **Reranking is CPU-bound.** ~1 second per query for 100 candidates, which is invisible in a demo and unacceptable at scale without a GPU or a distilled model.

---

## 8. Reproducing

```bash
python fetch_corpus.py --limit 40    # smoke test
python fetch_corpus.py               # full category list from config.py
python chunk.py                      # -> data/chunks.jsonl
python index.py --skip-dense         # BM25 only, no torch needed
python index.py                      # + embeddings (~30 min on CPU)

python eval.py --systems bm25 dense hybrid hybrid+rerank --k 10
python probe.py
python ambiguous_eval.py
python sweep_blend.py
python app.py                        # UI on http://127.0.0.1:8000
```

All commands run under the project venv (`.venv/Scripts/python.exe` on Windows); `app.py` warns if launched with a different interpreter, because the dense and rerank routes fail without `sentence-transformers`.

Build artifacts (`data/pages.jsonl`, `data/bm25.pkl`, `data/embeddings.npy`, `data/embeddings_meta.json`) are gitignored; `data/chunks.jsonl` is committed so the corpus is inspectable without re-fetching. Both indexes validate their row count against `chunks.jsonl` at load time and refuse to serve a stale index.

---

## 9. Conclusion

The honest summary is that **the pipeline matters more than the model**. Section-based chunking with breadcrumb prefixes, a BM25 gate on fusion, a span-scaled version prior and a blended cross-encoder each moved results more than the choice between four retrieval paradigms did.

The measured hierarchy is:

1. **Hybrid + blended rerank** is the best default for named and semi-named queries: highest recall, highest nDCG, and — because of the blend — no regression against the retriever that already had the answer at rank 1.
2. **Dense alone** is the best default for purely descriptive queries at top-3, and the least fragile system overall.
3. **BM25** is a necessary baseline and a surprisingly competitive one on multi-hop lookups, but it collapses on intent phrasing.
4. **Pure cross-encoder reranking** is worse than not reranking at all, and the blend-weight sweep is the evidence.

What would come next, in order of expected value: query-type routing between dense and rerank, a larger and independently labelled evaluation set (especially multi-hop), and validating the BM25 implementation against a reference.
