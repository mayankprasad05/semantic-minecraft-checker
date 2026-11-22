"""Accuracy check on deliberately ambiguous mixed-category queries.

probe.py and eval.py both lean on queries where the answer is often unique.
That flatters the systems: a vague description that still contains the item's
own name is easy. Here every query is written the way a player actually asks --
by function, colour, ability, or behaviour -- with no lexical overlap with the
target title, and each spans potions / materials / tools / mobs.

Grading is strict: page-level top-1 and top-3 hits, with a list of acceptable
pages where the wording is genuinely open to more than one right answer.

    python ambiguous_eval.py
    python ambiguous_eval.py --check-titles   # validate labels against the corpus
"""

import argparse
import sys

import config
import eval as E
import search

# (category, query, acceptable pages in order of preference)
QUERIES = [
    # --- potions -----------------------------------------------------------
    ("potions", "the drink that makes you run faster", ["Potion of Swiftness"]),
    ("potions", "drink that makes you jump higher", ["Potion of Leaping"]),
    ("potions", "the potion that keeps you alive in lava", ["Potion of Fire Resistance"]),
    ("potions", "drink that lets you see in the dark", ["Potion of Night Vision"]),
    ("potions", "the drink that weakens whatever you hit", ["Potion of Weakness"]),
    ("potions", "potion you throw at someone to hurt them", ["Splash Potion", "Potion of Harming"]),
    ("potions", "drink that makes you harder to hit but slower",
     ["Potion of the Turtle Master"]),
    ("potions", "the brew that lets you breathe underwater", ["Potion of Water Breathing"]),

    # --- materials ---------------------------------------------------------
    ("materials", "the green gem villagers want", ["Emerald"]),
    ("materials", "the blue stone you spend to enchant", ["Lapis Lazuli"]),
    ("materials", "the metal from the nether that beats diamond", ["Netherite Ingot"]),
    ("materials", "the red powder that carries power", ["Redstone Dust"]),
    ("materials", "sticky thing slimes drop", ["Slimeball"]),
    ("materials", "stringy thing spiders leave behind", ["String"]),
    ("materials", "the shiny blue thing that makes the best pickaxe", ["Diamond"]),
    ("materials", "the yellow dust that makes potions last longer", ["Redstone Dust",
                                                                  "Glowstone Dust"]),

    # --- tools -------------------------------------------------------------
    ("tools", "the thing that breaks stone faster than your hands", ["Pickaxe"]),
    ("tools", "the tool for chopping down trees", ["Axe"]),
    ("tools", "the tool for digging up dirt", ["Shovel"]),
    ("tools", "the garden tool that prepares soil for seeds", ["Hoe"]),
    ("tools", "the thing you use to catch fish", ["Fishing Rod"]),
    ("tools", "the weapon that fires arrows from a distance", ["Bow"]),
    ("tools", "the swinging weapon that does the most damage", ["Sword"]),

    # --- mobs --------------------------------------------------------------
    ("mobs", "the green thing that explodes next to you", ["Creeper"]),
    ("mobs", "the tall dark thing that teleports", ["Enderman"]),
    ("mobs", "the flying nether thing that spits fire", ["Ghast", "Blaze"]),
    ("mobs", "the small zombie that lives in the desert", ["Husk"]),
    ("mobs", "the thing that walks up and trades with you", ["Villager", "Wandering Trader"]),
    ("mobs", "the bony thing that shoots arrows", ["Skeleton"]),
    ("mobs", "the burning nether thing that flies at you", ["Blaze", "Ghast"]),
]

CATEGORIES = ["potions", "materials", "tools", "mobs"]


def corpus_titles():
    return {c["page"] for c in search.chunks()}


def check_titles():
    titles = corpus_titles()
    bad = 0
    for cat, q, targets in QUERIES:
        missing = [t for t in targets if t not in titles]
        if missing:
            bad += 1
            print(f"MISSING TARGET  [{cat}] {q!r} -> {missing}")
            for t in missing:
                near = [x for x in titles if t.split()[0].lower() in x.lower()][:6]
                print(f"   similar titles: {near}")
    print(f"\n{len(QUERIES)} queries, {bad} with unknown target titles")
    return bad == 0


def rank_of(pages, targets):
    """1-based rank of the best acceptable page, or None."""
    for i, page in enumerate(pages, start=1):
        if page in targets:
            return i
    return None


def run_system(query, system, k=10):
    if system == "hybrid+rerank":
        import rerank
        return rerank.search_with_rerank(query, k=k)
    return search.search(query, system=system, k=k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-titles", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    if args.check_titles:
        sys.exit(0 if check_titles() else 1)

    systems = ["bm25", "dense", "hybrid", "hybrid+rerank"]
    rows = []
    for cat, query, targets in QUERIES:
        result = {"category": cat, "query": query, "targets": targets}
        for system in systems:
            pages = E.returned_pages(run_system(query, system, args.k))
            result[system] = (pages, rank_of(pages, targets))
        rows.append(result)
        print(f"[{cat}] {query}", flush=True)

    print("\n" + "=" * 100)
    print("Per-query rank of the first acceptable page (1 = right answer first)")
    print("=" * 100)
    header = f"{'query':<56} {'want':<22}" + "".join(f"{s:>12}" for s in systems)
    print(header)
    print("-" * len(header))
    for r in rows:
        want = r["targets"][0]
        want = want if len(want) <= 21 else want[:20] + "."
        q = r["query"] if len(r["query"]) <= 55 else r["query"][:54] + "."
        cells = ""
        for s in systems:
            rank = r[s][1]
            cells += f"{('-' if rank is None else rank):>12}"
        print(f"{q:<56} {want:<22}{cells}")

    print("\nSummary: top-1 / top-3 / not found in top-10")
    print(f"{'category':<12}{'n':>3}" + "".join(f"{s:>22}" for s in systems))
    for cat in CATEGORIES:
        sub = [r for r in rows if r["category"] == cat]
        line = f"{cat:<12}{len(sub):>3}"
        for s in systems:
            top1 = sum(1 for r in sub if r[s][1] == 1) / len(sub)
            top3 = sum(1 for r in sub if r[s][1] and r[s][1] <= 3) / len(sub)
            miss = sum(1 for r in sub if r[s][1] is None)
            line += f"{top1:>8.2f}{top3:>7.2f}{miss:>6}"
        print(line)
    line = f"{'ALL':<12}{len(rows):>3}"
    for s in systems:
        top1 = sum(1 for r in rows if r[s][1] == 1) / len(rows)
        top3 = sum(1 for r in rows if r[s][1] and r[s][1] <= 3) / len(rows)
        miss = sum(1 for r in rows if r[s][1] is None)
        line += f"{top1:>8.2f}{top3:>7.2f}{miss:>6}"
    print(line)

    print("\nMisses (not in top 10 for any system):")
    for r in rows:
        if all(r[s][1] is None for s in systems):
            print(f"  [{r['category']}] {r['query']!r} want={r['targets']}")
            print(f"      dense top3={r['dense'][0][:3]}")
            print(f"      rerank top3={r['hybrid+rerank'][0][:3]}")

    print("\nAMBIG DONE")


if __name__ == "__main__":
    main()
