"""Turn raw wikitext into section-level chunks.

Design notes worth defending in a report:

* Chunk per section, not per fixed token window. Wiki pages already have
  meaningful boundaries (== Obtaining ==, == Usage ==).
* Prefix every chunk with "<page> > <section>: " before indexing. A section body
  often starts with "It can be crafted from..." and has no subject noun of its
  own, so the breadcrumb is doing real work for retrieval.
* Drop Trivia / Gallery / History etc. (see config.SKIP_SECTIONS). Those
  sections are the classic source of "the search found a fun fact" failures.

    python chunk.py
"""

import argparse
import json
import re

import config

COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
REF_RE = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>|<references\s*/>", re.S)
TABLE_RE = re.compile(r"\{\|.*?\|\}", re.S)
FILE_RE = re.compile(r"\[\[(?:File|Image|Category):[^\[\]]*\]\]", re.I)
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
EXT_LINK_RE = re.compile(r"\[https?://[^\]]*\]")
LINK_RE = re.compile(r"\[\[([^\[\]|]+)\|([^\[\]]+)\]\]")
SIMPLE_LINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
QUOTE_RE = re.compile(r"'{2,5}")
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]{0,80}>")
HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)
IMAGE_TOKEN_RE = re.compile(r"\b\w+\.(?:png|jpe?g|gif|webp|svg|ogg|ogv)\b", re.I)
LIST_MARKERS = "*#:;|!"

OVERLOAD_SECTIONS = {
    "mob-spawning",
    "spawning",
    "behavior",
    "usage",
    "drop",
    "drops",
    "mob loot",
    "trading",
    "trades",
}


def looks_like_gallery(text):
    """True for chunks that are mostly leftover image/file listings.

    Catches 'Textures', 'Screenshots', 'Production images' and friends without
    having to enumerate every gallery-style heading on the wiki. Two signals:

    * many image filenames among the tokens, or
    * a surviving '|' next to a filename -- after cleaning, a pipe left in the
      text means an unresolved ''[[File:x.png|Caption]]'' fragment, never prose.
    """
    tokens = text.split()
    hits = len(IMAGE_TOKEN_RE.findall(text))
    if not hits:
        return False
    if hits >= 2 and "|" in text:
        return True
    return len(tokens) >= 8 and hits / len(tokens) > 0.04


def clean_wikitext(text):
    """Strip the markup that hurts retrieval, keep the prose.

    The passes are applied in a loop rather than once each, because wiki markup
    nests: [[File:x.png|thumb|A [[creeper]] in a field]] only resolves once the
    inner link has collapsed. Running to a fixed point is what stops captions
    and file names leaking into the index as "Creeper Explosion.gif|A creeper".
    """
    text = COMMENT_RE.sub(" ", text)
    text = REF_RE.sub(" ", text)
    for _ in range(10):
        before = text
        text = TABLE_RE.sub(" ", text)
        text = TEMPLATE_RE.sub(" ", text)
        text = FILE_RE.sub(" ", text)
        text = LINK_RE.sub(r"\2", text)
        text = SIMPLE_LINK_RE.sub(r"\1", text)
        if text == before:
            break
    text = EXT_LINK_RE.sub(" ", text)
    text = QUOTE_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)

    lines = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        if line[0] in LIST_MARKERS:
            line = line.lstrip(LIST_MARKERS).strip()
            if not line:
                continue
        lines.append(line)
    return "\n".join(lines)


def split_sections(wikitext):
    """Return [(section_name, raw_body), ...] with the intro as 'Overview'."""
    matches = list(HEADING_RE.finditer(wikitext))
    if not matches:
        return [("Overview", wikitext)]
    sections = [("Overview", wikitext[: matches[0].start()])]
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(wikitext)
        sections.append((match.group(2).strip(), wikitext[match.end() : end]))
    return sections


def pack_paragraphs(text, max_chars):
    """Split an over-long section into <= max_chars pieces on line boundaries."""
    if len(text) <= max_chars:
        return [text]

    pieces, current = [], []
    size = 0
    for line in text.splitlines():
        while len(line) > max_chars:  # single monster line: cut on sentence ends
            cut = line.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > 0 else max_chars
            if current:
                pieces.append("\n".join(current))
                current, size = [], 0
            pieces.append(line[:cut].strip())
            line = line[cut:].strip()
        if size + len(line) + 1 > max_chars and current:
            pieces.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        pieces.append("\n".join(current))
    return [p for p in pieces if p.strip()]


def is_usable(piece):
    """Keep real prose even when short; drop only fragmentary markup dust."""
    if len(piece) >= config.MIN_CHUNK_CHARS:
        return True
    return len(piece.split()) >= config.MIN_PROSE_WORDS


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def chunks_for_page(page, stats):
    title = page["title"]
    categories = page.get("categories") or [page.get("category", "")]
    primary = page.get("category") or categories[0]
    chunks = []
    for section, raw_body in split_sections(page["wikitext"]):
        if section.lower() in config.SKIP_SECTIONS:
            stats["skipped_sections"] += 1
            continue
        body = clean_wikitext(raw_body)
        if not body.strip():
            continue
        if looks_like_gallery(body):
            stats["gallery"] += 1
            continue
        for i, piece in enumerate(pack_paragraphs(body, config.MAX_CHUNK_CHARS)):
            if not is_usable(piece):
                stats["too_short"] += 1
                continue
            if len(piece) < config.MIN_CHUNK_CHARS:
                stats["short_prose"] += 1
            suffix = f"-{i}" if i else ""
            chunks.append(
                {
                    "id": f"{slugify(title)}#{slugify(section)}{suffix}",
                    "page": title,
                    "section": section,
                    "category": primary,
                    "categories": categories,
                    "url": page.get("url", ""),
                    "text": piece,
                    "version": page.get("version"),
                    # Indexed and embedded; the breadcrumb is the trick.
                    "search_text": f"{title} > {section}: {piece}",
                }
            )
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", default=str(config.PAGES_PATH))
    parser.add_argument("--out", default=str(config.CHUNKS_PATH))
    args = parser.parse_args()

    stats = {
        "pages": 0,
        "skipped_sections": 0,
        "too_short": 0,
        "no_wikitext": 0,
        "gallery": 0,
        "short_prose": 0,
    }
    chunks = []
    with open(args.pages, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            page = json.loads(line)
            stats["pages"] += 1
            if not page.get("wikitext"):
                stats["no_wikitext"] += 1
                continue
            chunks.extend(chunks_for_page(page, stats))

    # Guarantee unique ids even if two pages slugify identically.
    seen = {}
    for chunk in chunks:
        base = chunk["id"]
        if base in seen:
            seen[base] += 1
            chunk["id"] = f"{base}-{seen[base]}"
        else:
            seen[base] = 0

    config.DATA.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    lens = [len(c["text"]) for c in chunks] or [0]
    print(f"pages read       : {stats['pages']}")
    print(f"chunks written   : {len(chunks)}")
    print(f"unique pages     : {len({c['page'] for c in chunks})}")
    print(f"avg chunk chars  : {sum(lens) / len(lens):.0f}")
    print(f"skipped sections : {stats['skipped_sections']} (Trivia/Gallery/History/...)")
    print(f"kept short prose : {stats['short_prose']} (<{config.MIN_CHUNK_CHARS} chars, real sentences)")
    print(f"dropped <min     : {stats['too_short']} (markup dust)")
    print(f"dropped gallery  : {stats['gallery']}")
    print(f"pages w/o text   : {stats['no_wikitext']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
