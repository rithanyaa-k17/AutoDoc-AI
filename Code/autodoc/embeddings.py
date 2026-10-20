"""Semantic embeddings via sentence-transformers (default: all-MiniLM-L6-v2, ~90 MB, CPU is fine).

The model is downloaded ONCE from Hugging Face on first use and cached on disk; after that it works offline.
Any object with `.name` and `.encode(list[str]) -> float32 array (L2-normalised)` can be used (tests use a fake).
"""
import os

import numpy as np

from .config import CFG


class EmbeddingUnavailable(RuntimeError):
    pass


class SentenceTransformerEmbedder:
    def __init__(self, model_name=None):
        self.name = model_name or os.environ.get("AUTODOC_EMBED_MODEL") or CFG["rag"]["embedding_model"]
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise EmbeddingUnavailable(
                "sentence-transformers is not installed. Run: python -m pip install sentence-transformers "
                f"({exc})") from exc
        try:
            self.model = SentenceTransformer(self.name, device="cpu")
        except Exception as exc:
            raise EmbeddingUnavailable(
                f"Could not load embedding model '{self.name}'. The first run needs internet to download it "
                f"(about 90 MB). Run: python Code\\check_setup.py --download  ({type(exc).__name__}: {exc})") from exc
        self.dim = int(self.model.encode(["dimension probe"], convert_to_numpy=True).shape[1])

    def encode(self, texts, batch_size=32):
        vecs = self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vecs, dtype="float32")
