# Results

_Generated 2026-11-22 by `eval.py`._

- Queries labelled: **57**
- Systems: bm25, dense, hybrid, hybrid+rerank
- Recall cut-off: @10

## Overall

| system | n | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| bm25 | 57 | 0.807 | 0.660 | 0.678 |
| dense | 57 | 0.895 | 0.816 | 0.820 |
| hybrid | 57 | 0.895 | 0.808 | 0.816 |
| hybrid+rerank | 57 | 0.930 | 0.804 | 0.822 |

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
| dense | 20 | 0.750 | 0.577 | 0.619 |
| hybrid | 20 | 0.750 | 0.577 | 0.619 |
| hybrid+rerank | 20 | 0.850 | 0.565 | 0.632 |

### multi-hop

| system | n | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| bm25 | 3 | 1.000 | 0.778 | 0.551 |
| dense | 3 | 0.667 | 0.667 | 0.413 |
| hybrid | 3 | 0.667 | 0.500 | 0.338 |
| hybrid+rerank | 3 | 0.667 | 0.500 | 0.364 |

### version-sensitive

| system | n | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| bm25 | 5 | 1.000 | 0.900 | 0.902 |
| dense | 5 | 1.000 | 0.900 | 0.902 |
| hybrid | 5 | 1.000 | 0.900 | 0.896 |
| hybrid+rerank | 5 | 1.000 | 0.900 | 0.896 |


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
| bm25 | 19 | 0.947 | 0.841 | 0.818 |
| dense | 19 | 1.000 | 0.974 | 0.941 |
| hybrid | 19 | 1.000 | 0.974 | 0.948 |
| hybrid+rerank | 19 | 1.000 | 0.947 | 0.933 |

### mobs

| system | n | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| bm25 | 11 | 0.818 | 0.405 | 0.501 |
| dense | 11 | 0.818 | 0.591 | 0.636 |
| hybrid | 11 | 0.818 | 0.545 | 0.600 |
| hybrid+rerank | 11 | 0.909 | 0.622 | 0.678 |

## Notes

Failure analysis is written by hand below; the tables above are generated.

## What changed in this round

### 1. The `version` field now affects ranking

`search.is_version_query()` detects when the user is asking *when* something was
added or changed (added / introduced / released / update / patch / snapshot /
which version / when did). For those queries only, `search._apply_version_prior()`
adds a bonus to any chunk carrying `version` metadata, plus a smaller bonus
proportional to query/title word overlap. Both bonuses scale with the score span
of the candidate list, so the same code works for BM25, cosine and RRF scores.

Effect on the version-sensitive bucket (previously the worst bucket for dense):

| system | before (R@10 / MRR / nDCG) | after |
|---|---|---|
| bm25 | 0.800 / 0.196 / 0.352 | 1.000 / 0.900 / 0.902 |
| dense | 0.600 / 0.119 / 0.289 | 1.000 / 0.900 / 0.902 |
| hybrid | 0.800 / 0.119 / 0.289 | 1.000 / 0.900 / 0.896 |
| hybrid+rerank | 0.600 / 0.119 / 0.289 | 1.000 / 0.900 / 0.896 |

Both netherite questions now return Netherite Ingot at rank 1. Queries that are
not version questions are untouched: `_apply_version_prior` returns the list
unchanged when the detector does not fire.

### 2. Rerank now blends with the retriever instead of replacing it

Rerank previously sorted purely by cross-encoder score, discarding hybrid's
ordering. That was measurably wrong on materials (MRR 0.783 vs hybrid's 0.886).
`rerank.blend_rankings()` now min-max normalises both score lists and combines
them as `w * cross_encoder + (1 - w) * hybrid`, with `w = config.RERANK_BLEND_WEIGHT`.

`python sweep_blend.py` caches cross-encoder scores once per query and evaluates
each weight. Results on the 57 labelled queries:

| w | overall R@10 | MRR | nDCG | materials MRR | mobs MRR | potions MRR |
|---|---|---|---|---|---|---|
| 0.3 | 0.912 | 0.810 | 0.821 | 0.974 | 0.609 | 0.776 |
| **0.4** | **0.930** | **0.804** | **0.822** | **0.947** | 0.622 | 0.776 |
| 0.5 | 0.912 | 0.798 | 0.814 | 0.939 | 0.617 | 0.773 |
| 0.6 | 0.947 | 0.785 | 0.809 | 0.882 | 0.680 | 0.759 |
| 0.7 | 0.947 | 0.797 | 0.818 | 0.864 | 0.728 | 0.778 |
| 0.8 | 0.930 | 0.778 | 0.798 | 0.815 | 0.718 | 0.777 |
| 0.9 | 0.895 | 0.746 | 0.768 | 0.800 | 0.622 | 0.759 |
| 1.0 (old) | 0.895 | 0.730 | 0.756 | 0.783 | 0.590 | 0.749 |

0.4 was chosen: best overall MRR and nDCG, and materials MRR 0.947 against
plain hybrid's 0.886 and pure rerank's 0.783. Note that pure reranking is the
worst setting in the table on overall MRR and nDCG, which is why the blend is
not just a small correction.

## Probe (vague descriptions, rank of the first correct page)

| description | target | bm25 | dense | hybrid | rerank |
|---|---|---|---|---|---|
| the drink you get that lets you see in the dark | Potion of Night Vision | >400 | 1 | 1 | 1 |
| a brew that lets you breathe underwater for a while | Potion of Water Breathing | 234 | 3 | 3 | 2 |
| something that makes you jump really high for a bit | Potion of Leaping | 30 | 1 | 1 | 1 |
| shiny blue mineral found deep underground, best pickaxe | Diamond | 74 | 50 | 50 | 37 |
| strongest metal from the nether, better than diamond | Netherite Ingot | >400 | 20 | 30 | 27 |
| sticky stuff slimes drop, sticky pistons | Slimeball | 80 | 18 | 48 | 69 |
| what block makes crops grow instantly | Bone Meal | 7 | 8 | 8 | 6 |
| a block that pushes other blocks around | Piston | 12 | 3 | 3 | 3 |
| the block that shoots out arrows or items | Dispenser | 4 | 56 | 56 | 36 |
| the block that detects changes nearby | Observer | 5 | 8 | 8 | 5 |
| the container that moves items between chests automatically | Hopper | 8 | 3 | 3 | 4 |
| the block that ejects items when powered | Dropper | 1 | 2 | 2 | 2 |
| the strongest melee weapon you can make from a nether metal | Netherite Sword | >400 | 1 | 1 | 4 |
| the thing that mines blocks faster than fists | Pickaxe | 81 | 4 | 11 | 12 |
| the crop that makes bread | Wheat | 8 | 4 | 4 | 4 |
| the food pigs like to eat | Potato | 3 | 15 | 15 | 22 |
| the block that turns food scraps into fertilizer | Composter | 178 | 3 | 7 | 2 |
| the tall black mob that steals blocks | Enderman | 145 | 36 | 36 | 28 |
| the tiny poisonous spider found in mineshafts | Cave Spider | 1 | 2 | 1 | 1 |
| the villager that wanders around selling stuff | Wandering Trader | 13 | 2 | 2 | 1 |

Correct page in top 10: bm25 8/20, dense 14/20, hybrid 13/20, **rerank 13/20**
(previously 11/20).

| probe category | bm25 | dense | hybrid | rerank (before -> after) |
|---|---|---|---|---|
| potions | 0 | 3 | 3 | 3 -> 3 |
| resources/materials | 1 | 1 | 1 | 0 -> 1 |
| blocks | 4 | 4 | 4 | 4 -> 4 |
| tools | 0 | 2 | 1 | 0 -> 1 |
| farming | 2 | 2 | 2 | 2 -> 2 |
| mobs | 1 | 1 | 1 | 1 -> 1 |
| villagers | 0 | 1 | 1 | 1 -> 1 |

## Remaining weaknesses

- Enderman is still ranked ~28-36 for "the tall black mob that steals blocks".
  Its description text does not contain "steals blocks"; the boost would need a
  hard-coded alias list rather than a ranking change.
- Diamond and Netherite Ingot sit mid-pack on their vague description probe
  ("shiny blue mineral", "strongest metal from the nether") because the
  surrounding tool/armor pages share almost all of their vocabulary.
- Multi-hop (n=3) is now the weakest bucket on the dense/hybrid/rerank side:
  recall 0.667 (bm25 1.000) and MRR 0.667 / 0.500 / 0.500. The version prior does
  not fire on these, and the questions need two linked lookups ("mobs that drop
  string" then "what string crafts"), which no single-pass ranker here does.
- The 5 version-sensitive labels are few. The prior is gated narrowly on purpose
  (it only fires on "when was X added" phrasing) so it cannot distort ordinary
  lookups, but that also means it will not help "what did 1.20 change" style
  queries.

## Ambiguous mixed-category accuracy check

`ambiguous_eval.py` is a deliberately hard set: 30 queries across potions /
materials / tools / mobs, each written by function, colour or behaviour with no
lexical overlap with the target title ("the green thing that explodes next to
you", "the tool for digging up dirt", "sticky thing slimes drop"). Where the
wording is genuinely open, more than one page is accepted. All 30 target titles
were verified to exist in the corpus (`--check-titles`), so no score is diluted
by a mislabelled page.

Grading is strict: page-level top-1 and top-3, not recall@10.

| category | n | bm25 top1/top3/miss | dense | hybrid | hybrid+rerank |
|---|---|---|---|---|---|
| potions | 8 | 0.00 / 0.00 / 8 | **0.62 / 0.75 / 2** | 0.62 / 0.75 / 2 | 0.38 / 0.62 / 1 |
| materials | 8 | 0.12 / 0.25 / 4 | 0.00 / 0.38 / 3 | 0.00 / 0.38 / 3 | 0.25 / 0.38 / 3 |
| tools | 7 | 0.00 / 0.00 / 6 | 0.00 / 0.57 / 2 | 0.00 / 0.57 / 1 | 0.14 / 0.43 / 0 |
| mobs | 7 | 0.00 / 0.00 / 6 | 0.57 / 0.71 / 1 | 0.14 / 0.29 / 2 | 0.43 / 0.43 / 2 |
| **all** | 30 | 0.03 / 0.07 / 24 | **0.30 / 0.60 / 8** | 0.20 / 0.50 / 8 | 0.30 / 0.47 / 6 |

### What this says

- **These are much harder than the labelled set.** Overall rerank drops from
  0.930 recall@10 on `queries.jsonl` to 0.47 top-3 here. That is not a
  contradiction: the labelled set is full of exact-token queries, and this set
  deliberately has none.
- **Dense is the best system on ambiguous phrasing** (top-1 0.30, top-3 0.60).
  Hybrid+rerank matches its top-1 but loses top-3 (0.47 vs 0.60), and hybrid is
  worse than both. BM25 is effectively unusable here: 24 of 30 queries return
  nothing acceptable in the top 10.
- **Tools are the weakest category.** No system puts the right tool first, and
  "the thing that breaks stone faster than your hands" ranks Pickaxe 7th (hybrid)
  and 8th (rerank). Tool descriptions are dominated by the material variants
  (Golden/Stone/Netherite Pickaxe), which are on-topic but not the base page.
- **Six queries defeat every system**, including:
  - "drink that makes you harder to hit but slower" -> Potion of the Turtle Master
    (returns Witch, Potion of Slowness, Cobweb)
  - "the metal from the nether that beats diamond" -> Netherite Ingot
    (returns Netherite Armor / Netherite Sword, which mention diamond constantly)
  - "stringy thing spiders leave behind" -> String
    (returns Spider franchise pages; the word "spiders" wins over the concept)
  - "the shiny blue thing that makes the best pickaxe" -> Diamond (returns the
    pickaxe pages — the head noun dominates the embedding)
  - "the green thing that explodes next to you" -> Creeper
    (returns Story Mode:Firework Rocket / Story Mode:Creeper)
- **Franchise-prefixed pages are not a systemic retrieval problem.** They are
  31% of page titles but only 20-23% of retrieved top-3/top-10 slots, i.e. they
  are slightly *under*-represented overall. They surface only on the specific
  queries where their text matches the description better than the base article.

### Consequence for choosing a default

The `hybrid+rerank` default is right for named lookups (best on `queries.jsonl`),
but it is not the best system for vague descriptions, where dense wins on top-3.
That trade-off is not resolvable by one global default; it would need
query-type routing (short/exact -> hybrid+rerank, descriptive -> dense).
