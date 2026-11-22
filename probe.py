"""Qualitative check: vague player descriptions, judged by hand.

Static metrics in eval.py use queries written by someone who knows the corpus.
Real players describe what they half-remember ("the drink that lets you see in the
dark"), so this probe holds a handful of those and records where the correct
page actually lands.

    python probe.py

Reports the rank of the target page per system, >N meaning it fell outside the
pool. Rank is page-level: duplicate chunks of the same page count once, which is
what the UI shows.
"""

import config  # noqa: F401  (imported for the side effect of path setup)
import rerank
import search

# (description, page titles that would be a correct answer)
CASES = [
    ("the drink you get that lets you see in the dark", ["Potion of Night Vision"]),
    ("a brew that lets you breathe underwater for a while", ["Potion of Water Breathing"]),
    ("something that makes you jump really high for a bit", ["Potion of Leaping"]),
    ("shiny blue mineral found deep underground, best pickaxe", ["Diamond"]),
    ("strongest metal from the nether, better than diamond", ["Netherite Ingot"]),
    ("sticky stuff slimes drop, sticky pistons", ["Slimeball"]),
    ("what block makes crops grow instantly", ["Bone Meal"]),
    ("a block that pushes other blocks around", ["Piston"]),
    ("the block that shoots out arrows or items", ["Dispenser"]),
    ("the block that detects changes nearby", ["Observer"]),
    ("the container that moves items between chests automatically", ["Hopper"]),
    ("the block that ejects items when powered", ["Dropper"]),
    ("the strongest melee weapon you can make from a nether metal", ["Netherite Sword"]),
    ("the thing that mines blocks faster than fists", ["Pickaxe"]),
    ("the crop that makes bread", ["Wheat"]),
    ("the food pigs like to eat", ["Potato"]),
    ("the block that turns food scraps into fertilizer", ["Composter"]),
    ("the tall black mob that steals blocks", ["Enderman"]),
    ("the tiny poisonous spider found in mineshafts", ["Cave Spider"]),
    ("the villager that wanders around selling stuff", ["Wandering Trader"]),
]

POOL = 400
SYSTEMS = ["bm25", "dense", "hybrid"]


def page_rank(query, system, targets, pool=POOL):
    """1-based rank of the first target page, or None if outside the pool."""
    all_chunks = search.chunks()
    seen, position = set(), 0
    for doc_idx, _score in search.ranked(query, system, pool):
        page = all_chunks[doc_idx]["page"]
        if page in seen:
            continue
        seen.add(page)
        position += 1
        if page in targets:
            return position
    return None


def rerank_rank(query, targets, pool=200):
    pages = [r["page"] for r in rerank.search_with_rerank(query, k=100, retrieve_k=pool)]
    for position, page in enumerate(pages, start=1):
        if page in targets:
            return position
    return None


def main():
    header = f"{'description':<62}" + "".join(f"{s:>9}" for s in SYSTEMS) + f"{'rerank':>9}"
    print(header)
    print("-" * len(header))

    hits = {system: 0 for system in SYSTEMS + ["rerank"]}
    for query, targets in CASES:
        row = ""
        for system in SYSTEMS:
            rank = page_rank(query, system, targets)
            hits[system] += 1 if (rank and rank <= 10) else 0
            row += f"{(rank if rank else '>400'):>9}"
        rank = rerank_rank(query, targets)
        hits["rerank"] += 1 if (rank and rank <= 10) else 0
        row += f"{(rank if rank else '>100'):>9}"
        print(f"{query[:60]:<62}{row}")

    n = len(CASES)
    print()
    print("correct page in top 10: " + "  ".join(f"{s} {hits[s]}/{n}" for s in SYSTEMS + ["rerank"]))

    print()
    print("By category (vague description probe)")
    print(f"{'category':<18}" + "".join(f"{s:>10}" for s in SYSTEMS) + f"{'rerank':>10}")
    cat_cases = {
        "potions": [0, 1, 2],
        "resources": [3, 4, 5, 6],
        "blocks": [7, 8, 9, 10, 11],
        "tools": [12, 13],
        "farming": [14, 15, 16],
        "mobs": [17, 18],
        "villagers": [19],
    }
    for cat, idxs in cat_cases.items():
        cat_hits = {s: 0 for s in SYSTEMS + ["rerank"]}
        for i in idxs:
            query, targets = CASES[i]
            for system in SYSTEMS:
                rank = page_rank(query, system, targets)
                if rank and rank <= 10:
                    cat_hits[system] += 1
            rank = rerank_rank(query, targets)
            if rank and rank <= 10:
                cat_hits["rerank"] += 1
        print(
            f"{cat:<18}"
            + "".join(f"{cat_hits[s]:>10}" for s in SYSTEMS)
            + f"{cat_hits['rerank']:>10}"
        )


if __name__ == "__main__":
    main()
