"""Build the two indexes.

    python index.py               # BM25 + dense (needs sentence-transformers)
    python index.py --skip-dense   # BM25 only, works on a bare install
"""

import argparse
import json
import pickle

import config


def load_chunks(path=None):
    path = path or config.CHUNKS_PATH
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_bm25(chunks):
    from bm25 import BM25Index

    index = BM25Index([chunk["search_text"] for chunk in chunks])
    config.DATA.mkdir(parents=True, exist_ok=True)
    with open(config.BM25_PATH, "wb") as handle:
        pickle.dump({"index": index, "n_docs": len(chunks)}, handle)
    print(f"bm25    : {index.n_docs} docs, {len(index.idf)} unique terms -> {config.BM25_PATH}")


def build_dense(chunks):
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print(f"dense   : SKIPPED ({exc}). Install sentence-transformers to build it.")
        return False

    print(f"dense   : encoding with {config.EMBED_MODEL} (first run downloads the model)")
    model = SentenceTransformer(config.EMBED_MODEL)
    model.max_seq_length = config.EMBED_MAX_SEQ_LEN
    vectors = model.encode(
        [chunk["search_text"] for chunk in chunks],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    np.save(config.EMB_PATH, vectors.astype("float32"))
    with open(config.EMB_META_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": config.EMBED_MODEL,
                "n": len(chunks),
                "dim": int(vectors.shape[1]),
                "ids": [chunk["id"] for chunk in chunks],
            },
            handle,
        )
    print(f"dense   : {vectors.shape[0]}x{vectors.shape[1]} -> {config.EMB_PATH}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default=str(config.CHUNKS_PATH))
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    if not chunks:
        raise SystemExit("No chunks found. Run fetch_corpus.py then chunk.py first.")
    print(f"chunks  : {len(chunks)} from {args.chunks}")

    build_bm25(chunks)
    if args.skip_dense:
        print("dense   : skipped by flag")
    else:
        build_dense(chunks)


if __name__ == "__main__":
    main()
