"""Local vector store: FAISS (IndexFlatIP, if faiss-cpu is installed) or exact NumPy cosine search.

Both are exact (brute-force) searches over L2-normalised vectors, so results are equivalent; FAISS is
optional. The index (vectors + chunk metadata) is cached on disk so documents are not re-embedded.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # faiss-cpu is optional
    faiss = None


def index_key(doc_hashes, model_name, chunk_cfg) -> str:
    raw = json.dumps([sorted(doc_hashes), model_name, chunk_cfg], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class VectorStore:
    def __init__(self, vectors: np.ndarray, chunks: list, prefer_faiss: bool = True):
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        self.chunks = chunks
        self.backend = "numpy"
        self._faiss = None
        if prefer_faiss and faiss is not None and len(self.vectors):
            idx = faiss.IndexFlatIP(self.vectors.shape[1])
            idx.add(self.vectors)
            self._faiss, self.backend = idx, "faiss"

    def search(self, qvec: np.ndarray, k: int = 5, doc_filter=None):
        """Returns [(chunk, cosine_score)] best first. `doc_filter` = set of filenames or None."""
        if not len(self.vectors):
            return []
        q = np.ascontiguousarray(qvec.reshape(1, -1), dtype="float32")
        n = len(self.vectors) if doc_filter else min(len(self.vectors), max(k, 1))
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, n)
            pairs = list(zip(ids[0].tolist(), scores[0].tolist()))
        else:
            s = (self.vectors @ q[0])
            order = np.argsort(-s)[:n]
            pairs = [(int(i), float(s[i])) for i in order]
        out = []
        for i, sc in pairs:
            if i < 0:
                continue
            c = self.chunks[i]
            if doc_filter and c["doc"] not in doc_filter:
                continue
            out.append((c, float(sc)))
            if len(out) >= k:
                break
        return out

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        (folder / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, prefer_faiss: bool = True):
        vec = np.load(folder / "vectors.npy")
        chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        if len(vec) != len(chunks):
            raise ValueError("corrupt index cache")
        return cls(vec, chunks, prefer_faiss)
