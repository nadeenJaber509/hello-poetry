"""Hybrid retrieval: dense (ChromaDB + MiniLM) + lexical (BM25), fused with RRF.

Why: the dense model ranks by general "topic" similarity, so a short chunk
containing the exact command `poetry new` can lose to chunks that merely talk
about Poetry commands. BM25 rewards exact/stemmed word overlap (new, create,
project) and fails on different questions than dense search does, so fusing
the two rankings recovers most of the misses.
"""
import re

from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

_stemmer = PorterStemmer()

STOPWORDS = {
    "which", "what", "how", "does", "do", "can", "a", "an", "the", "is", "are",
    "of", "to", "in", "for", "with", "by", "be", "it", "its", "that", "this",
    "i", "or", "on", "as", "from", "at", "where", "when", "under",
}


def tokenize(text, remove_stopwords=False):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [
        _stemmer.stem(w)
        for w in words
        if not (remove_stopwords and w in STOPWORDS)
    ]


def reciprocal_rank_fusion(rankings, k=60):
    """Fuse several ranked lists of ids. Returns [(id, score)] best first."""
    scores = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class HybridSearcher:
    def __init__(self, collection, model):
        self.collection = collection
        self.model = model

        stored = collection.get(include=["documents", "metadatas"])
        self.ids = stored["ids"]
        self.texts = stored["documents"]
        self.metadatas = stored["metadatas"]
        self.index_by_id = {cid: i for i, cid in enumerate(self.ids)}

        # Title is added to the lexical index only (stored text is unchanged).
        self.bm25 = BM25Okapi([
            tokenize(f"{m['title']} {t}")
            for m, t in zip(self.metadatas, self.texts)
        ])

    def _dense_ranking(self, query, n):
        embedding = self.model.encode([query]).tolist()[0]
        result = self.collection.query(query_embeddings=[embedding], n_results=n)
        return result["ids"][0]

    def _bm25_ranking(self, query, n):
        scores = self.bm25.get_scores(tokenize(query, remove_stopwords=True))
        order = sorted(range(len(self.ids)), key=lambda i: -scores[i])[:n]
        return [self.ids[i] for i in order if scores[i] > 0]

    def search(self, query, top_k=5, candidates=20, rrf_k=60):
        dense = self._dense_ranking(query, candidates)
        lexical = self._bm25_ranking(query, candidates)
        fused = reciprocal_rank_fusion([dense, lexical], k=rrf_k)[:top_k]

        matches = []
        for chunk_id, score in fused:
            i = self.index_by_id[chunk_id]
            matches.append({
                "id": chunk_id,
                "text": self.texts[i],
                "source": self.metadatas[i]["source"],
                "title": self.metadatas[i]["title"],
                "rrf_score": score,
                "dense_rank": dense.index(chunk_id) + 1 if chunk_id in dense else None,
                "bm25_rank": lexical.index(chunk_id) + 1 if chunk_id in lexical else None,
            })
        return matches