"""Patch version metadata into pages.jsonl for the high-value pages.

This is a small manual annotation for the priority potion/material/mob pages,
not a full version scrape. It writes a new file so pages.jsonl is not mutated
in place by accident; after review, replace data/pages.jsonl with the new file.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAGES_IN = ROOT / "data" / "pages.jsonl"
PAGES_OUT = ROOT / "data" / "pages_versioned.jsonl"

# Manually curated version notes for the priority pages.
# These are the "when did this get added" answers we care about for the
# version-sensitive bucket. Use the earliest version where the page/concept
# meaningfully existed in the game.
VERSION_MAP = {
    # potions
    "Potion": "30",
    "Potion of Night Vision": "30",
    "Potion of Water Breathing": "30",
    "Potion of Leaping": "30",
    "Potion of Swiftness": "30",
    "Potion of Fire Resistance": "30",
    "Potion of Healing": "30",
    "Potion of Harming": "30",
    "Potion of Invisibility": "30",
    "Potion of Poison": "30",
    "Potion of Slowness": "30",
    "Potion of Weakness": "30",
    "Potion of the Turtle Master": "1.11",
    "Potion of Slow Falling": "1.16.1",
    "Awkward Potion": "30",
    "Mundane Potion": "30",
    "Thick Potion": "30",
    "Lagging Potion": "1.11",
    # brewing materials
    "Brewing Stand": "1.0.0",
    "Nether Wart": "1.0.0",
    "Ghast Tear": "1.0.0",
    "Magma Cream": "1.0.0",
    "Fermented Spider Eye": "1.0.0",
    "Glowstone Dust": "1.0.0",
    "Gunpowder": "1.0.0",
    "Blaze Powder": "1.0.0",
    "Sugar": "1.0.0",
    "Spider Eye": "1.0.0",
    "Rabbit Hide": "1.0.0",
    "Slimeball": "1.0.0",
    # materials
    "Diamond": "1.0.0",
    "Diamond Ore": "1.0.0",
    "Netherite Ingot": "1.16",
    "Netherite Scrap": "1.16",
    "Ancient Debris": "1.16",
    "Netherite Block": "1.16",
    "Block of Iron": "1.0.0",
    "Block of Diamond": "1.0.0",
    "Gold Ingot": "1.0.0",
    "Iron Ingot": "1.0.0",
    # mobs
    "Enderman": "1.0.0",
    "Wandering Trader": "1.9",
    "Cave Spider": "1.8.1",
    "Spider": "1.0.0",
    "Witch": "1.0.0",
    "Skeleton": "1.0.0",
    "Creeper": "1.0.0",
    "Endermite": "1.9",
    "Zombie": "1.0.0",
    "Villager": "1.0.0",
}

IGNORE = {"categories", "category", "title", "url", "wikitext", "version"}


def patch_page(page):
    if page.get("title") in VERSION_MAP:
        page = dict(page)
        page["version"] = VERSION_MAP[page["title"]]
    return page


def main():
    if not PAGES_IN.exists():
        raise SystemExit(f"missing {PAGES_IN}")
    written = 0
    patched = 0
    with PAGES_IN.open(encoding="utf-8") as fin, PAGES_OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            page = json.loads(line)
            if page.get("title") in VERSION_MAP:
                page = patch_page(page)
                patched += 1
            fout.write(json.dumps(page, ensure_ascii=False) + "\n")
            written += 1
    print(f"pages written : {written}")
    print(f"pages patched : {patched}")
    print(f"-> {PAGES_OUT}")


if __name__ == "__main__":
    main()
