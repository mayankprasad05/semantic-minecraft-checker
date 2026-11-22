"""Paths, corpus settings and model names for the Minecraft retrieval project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PAGES_PATH = DATA / "pages.jsonl"
CHUNKS_PATH = DATA / "chunks.jsonl"
BM25_PATH = DATA / "bm25.pkl"
EMB_PATH = DATA / "embeddings.npy"
EMB_META_PATH = DATA / "embeddings_meta.json"
RESULTS_PATH = ROOT / "results.md"

# minecraft.wiki is the community wiki that replaced the old Fandom one.
# If it is unreachable, fetch_corpus.py --api fandom falls back to the old host.
WIKI_API = "https://minecraft.wiki/api.php"
WIKI_API_FALLBACK = "https://minecraft.fandom.com/api.php"
WIKI_BASE = "https://minecraft.wiki/w/"
USER_AGENT = (
    "minecraft-retrieval-course-project/0.1 "
    "(student project; contact: replace-with-your-email@example.com)"
)

# Categories to pull pages from. Narrow this list to shrink the corpus.
# These are real category names on minecraft.wiki (there is no Category:Crafting
# or Category:Enchanting -- those are pages, not categories). fetch_corpus.py
# warns when a category resolves to zero pages.
CATEGORIES = [
    "Blocks",
    "Items",
    "Mobs",
    "Biomes",
    "Food",
    "Villagers",
    "Redstone",
    "Tools",
    "Potions",
]

# Sections that are noise for a search demo. History is temporarily unskipped
# while we evaluate whether the versioned History chunks actually help the
# version-sensitive queries for the priority potion/material/mob pages.
SKIP_SECTIONS = {
    "trivia",
    "gallery",
    "screenshots",
    "in other media",
    "media",
    "references",
    "external links",
    "see also",
    "navigation",
    "notes",
    "issues",
    "sounds",
    "video",
    "data values",
    "loot",
}

MIN_CHUNK_CHARS = 120
MAX_CHUNK_CHARS = 1500
# A piece shorter than MIN_CHUNK_CHARS is still kept if it reads like a sentence
# ("Creepers drop 0-2 gunpowder."). Only markup dust is discarded.
MIN_PROSE_WORDS = 8

# bge-base is a stronger retriever than MiniLM-L6 (768 dims vs 384) at ~4x the
# encode time. The instruction prefix below is how bge was trained: queries get
# the prefix, documents do not. Leaving it off costs a few points of recall.
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Chunks average ~290 characters (under 90 tokens); 512 would pad and encode
# about twice as long for no extra context. Cap it and keep encode time sane.
EMBED_MAX_SEQ_LEN = 256
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Reciprocal Rank Fusion constant (the standard 60 from the original paper).
RRF_K = 60

# How much BM25 counts in the fused score relative to dense (dense weight is 1.0).
# Swept 0.25-1.0 against both the labelled queries and probe.py, at RRF_K of
# 10/30/60/100. Deweighting BM25 never improved the vague-description probe
# (3-4/7 at every setting) and cost labelled MRR (0.684 -> 0.594 at w=0.25), so
# the default stays at parity. The knob is kept because the sweep is a result.
RRF_BM25_WEIGHT = 1.0

# --- Version-aware ranking -------------------------------------------------
# When a query asks *when* something was added ("which version added netherite",
# "when did potions get added"), topical relevance alone cannot answer it: the
# item page and its derivative pages (Netherite Upgrade, Netherite Shovel) are
# equally on-topic, but only the item page carries the version annotation.
# For those queries only, chunks that carry a `version` field get a bonus, and
# so do chunks whose page title overlaps the query. Both bonuses are scaled by
# the score span of the candidate list, so they work for BM25, cosine and RRF
# scores alike. Non-version queries are untouched.
VERSION_QUERY_BOOST = 1.0   # x score-span, for chunks that carry a version
VERSION_TITLE_BOOST = 0.5   # x score-span x query/title word overlap

# --- Rerank blending -------------------------------------------------------
# The cross-encoder is more precise but it also discards the retriever's own
# ordering, which was measurably better than rerank on the materials slice
# (MRR 0.886 hybrid vs 0.783 rerank). Blending the normalised cross-encoder
# score with the normalised pre-rerank hybrid score keeps the cross-encoder's
# judgement while falling back on retrieval order when it is unsure.
RERANK_BLEND_WEIGHT = 0.4   # weight on the cross-encoder; 1.0 = pure rerank
# Chosen by sweep_blend.py over the labelled set: 0.4 had the best overall
# MRR (0.804) and nDCG (0.822), and materials reached 1.000 recall / 0.947 MRR
# (pure rerank managed 0.895 / 0.783; plain hybrid 0.947 / 0.886).
