"""A small, dependency-free Okapi BM25 implementation.

Written out rather than imported so the lexical half of the project runs on a
bare Python install. Roughly 60 lines, and you can read every term of the
scoring formula -- which is the point for a course project.
"""

import math
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be been by can do does for from has have how i if in into is it its
    me my of on or so than that the their them then there these they this to was were what when
    where which who why will with you your""".split()
)


def tokenize(text):
    """Lowercase, split on non-alphanumerics, drop stopwords and single chars."""
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


class BM25Index:
    """Okapi BM25 over a list of documents (here: chunk search_texts)."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        self.doc_len = []
        self.postings = defaultdict(dict)  # term -> {doc_idx: term_frequency}
        df = Counter()

        for doc_idx, doc in enumerate(docs):
            tokens = tokenize(doc)
            self.doc_len.append(len(tokens))
            for term, count in Counter(tokens).items():
                self.postings[term][doc_idx] = count
                df[term] += 1

        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        # Standard BM25 idf, always positive.
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scored(self, query):
        """Return [(doc_idx, score), ...] sorted by score descending."""
        scores = {}
        avgdl = self.avgdl or 1.0
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_idx, tf in self.postings[term].items():
                dl = self.doc_len[doc_idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * tf * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

    def search(self, query, k=10):
        return self.scored(query)[:k]
