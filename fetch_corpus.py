"""Download raw wikitext for Minecraft wiki pages through the MediaWiki API.

Uses the API (action=query) instead of scraping HTML: cleaner text, fewer
requests, and it respects the site's rate limits if you keep --sleep sane.

    python fetch_corpus.py --limit 40                       # quick smoke test
    python fetch_corpus.py                                  # full CATEGORIES list
    python fetch_corpus.py --categories Mobs --depth 2      # follow subcategories
    python fetch_corpus.py --api fandom                     # old host fallback

Most real categories are thin and their members sit in subcategories
(Category:Mobs -> Category:Hostile mobs -> ...), so --depth defaults to 1.
Pages reached by several categories are stored once, with all their categories.
"""

import argparse
import json
import time

import requests

import config


def api_get(api_url, params, session, retries=3):
    query = dict(params)
    query["format"] = "json"
    query["formatversion"] = "2"
    for attempt in range(retries):
        try:
            resp = session.get(api_url, params=query, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries - 1:
                raise
            print(f"  retrying after error: {exc}")
            time.sleep(2 * (attempt + 1))


def category_members(api_url, category, session, limit=None, sleep=0.5):
    """Return (page_titles, subcategory_names) for one category."""
    pages, subcats = [], []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "500",
        "cmtype": "page|subcat",
    }
    while True:
        data = api_get(api_url, params, session)
        for member in data.get("query", {}).get("categorymembers", []):
            if member.get("ns") == 14:  # Category namespace -> subcategory
                subcats.append(member["title"].split(":", 1)[-1])
                continue
            pages.append(member["title"])
            if limit and len(pages) >= limit:
                return pages, subcats
        cont = data.get("continue")
        if not cont:
            return pages, subcats
        params.update(cont)
        time.sleep(sleep)


def crawl_category(api_url, category, session, depth, limit, sleep, seen_categories=None):
    """Collect page titles under a category, recursing into subcategories."""
    seen_categories = seen_categories if seen_categories is not None else set()
    key = category.lower()
    if key in seen_categories:
        return []
    seen_categories.add(key)

    pages, subcats = category_members(api_url, category, session, limit, sleep)
    if depth > 0:
        for sub in subcats:
            pages.extend(
                crawl_category(api_url, sub, session, depth - 1, limit, sleep, seen_categories)
            )
    return pages


def fetch_wikitext(api_url, titles, session, sleep=0.5):
    """Fetch revisions content for up to 50 titles per request."""
    pages = {}
    for start in range(0, len(titles), 50):
        batch = titles[start : start + 50]
        params = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "redirects": "1",
            "titles": "|".join(batch),
        }
        data = api_get(api_url, params, session)
        for page in data.get("query", {}).get("pages", []):
            revisions = page.get("revisions") or []
            if not revisions:
                continue
            content = revisions[0].get("slots", {}).get("main", {}).get("content")
            if content:
                pages[page["title"]] = content
        time.sleep(sleep)
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="*", default=config.CATEGORIES)
    parser.add_argument("--limit", type=int, default=None, help="max pages per category")
    parser.add_argument("--depth", type=int, default=1, help="subcategory depth to follow")
    parser.add_argument("--api", choices=["wiki", "fandom"], default="wiki")
    parser.add_argument("--out", default=str(config.PAGES_PATH))
    parser.add_argument("--sleep", type=float, default=0.5, help="seconds between requests")
    args = parser.parse_args()

    api_url = config.WIKI_API if args.api == "wiki" else config.WIKI_API_FALLBACK
    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT

    print(f"API: {api_url}  depth={args.depth}")

    # title -> categories, so a page reached twice is stored once.
    categories_of = {}
    for category in args.categories:
        print(f"[{category}] crawling...")
        titles = crawl_category(api_url, category, session, args.depth, args.limit, args.sleep)
        unique = sorted(set(titles))
        print(f"[{category}] {len(unique)} pages")
        if not unique:
            print(f"[{category}] WARNING: empty category -- the name may not exist on this wiki")
        for title in unique:
            categories_of.setdefault(title, []).append(category)

    all_titles = sorted(categories_of)
    print(f"fetching wikitext for {len(all_titles)} unique pages...")
    texts = fetch_wikitext(api_url, all_titles, session, args.sleep)

    config.DATA.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.out, "w", encoding="utf-8") as handle:
        for title in all_titles:
            wikitext = texts.get(title)
            if not wikitext:
                continue
            cats = categories_of[title]
            record = {
                "title": title,
                "category": cats[0],
                "categories": cats,
                "url": config.WIKI_BASE + title.replace(" ", "_"),
                "wikitext": wikitext,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nwrote {written} pages -> {args.out}")


if __name__ == "__main__":
    main()
