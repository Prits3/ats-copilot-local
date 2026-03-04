from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(emb, dtype="float32")


class VectorStore:
    def __init__(self, dim: int = 384, persist_dir: str = ".chroma_local"):
        self.dim = dim
        self.kind = ""
        self.metas: List[Dict] = []
        self._faiss_index = None
        self._chroma_collection = None
        self._init_backend(persist_dir=persist_dir)

    def _init_backend(self, persist_dir: str) -> None:
        try:
            import faiss  # type: ignore

            self._faiss_index = faiss.IndexFlatIP(self.dim)
            self.kind = "faiss"
            return
        except Exception:
            pass

        import chromadb

        client = chromadb.PersistentClient(path=persist_dir)
        self._chroma_collection = client.get_or_create_collection("local_job_hunter")
        self.kind = "chroma"

    def add(self, embeddings: np.ndarray, metas: List[Dict]) -> None:
        if len(metas) == 0:
            return
        if self.kind == "faiss":
            self._faiss_index.add(embeddings)
            self.metas.extend(metas)
            return

        ids = [m.get("chunk_id", f"id_{i}") for i, m in enumerate(metas)]
        docs = [m.get("text", "") for m in metas]
        embs = embeddings.tolist()
        self._chroma_collection.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        self.metas.extend(metas)

    def search(self, query_embedding: np.ndarray, k: int = 6) -> List[Dict]:
        if self.kind == "faiss":
            if self._faiss_index.ntotal == 0:
                return []
            scores, ids = self._faiss_index.search(query_embedding, k)
            out: List[Dict] = []
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0:
                    continue
                m = self.metas[idx].copy()
                m["score"] = float(score)
                out.append(m)
            return out

        result = self._chroma_collection.query(query_embeddings=query_embedding.tolist(), n_results=k)
        out: List[Dict] = []
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for meta, dist in zip(metas, dists):
            m = dict(meta)
            # Convert distance-like value into similarity-like score for consistency.
            m["score"] = float(1.0 / (1.0 + float(dist)))
            out.append(m)
        return out


def build_profile_vector_store(
    chunks: List[Dict],
    embedder: Optional[EmbeddingModel] = None,
    persist_dir: str = ".chroma_local",
) -> Tuple[EmbeddingModel, VectorStore]:
    embedder = embedder or EmbeddingModel()
    store = VectorStore(dim=384, persist_dir=persist_dir)
    texts = [c["text"] for c in chunks]
    embs = embedder.embed(texts)
    store.add(embs, chunks)
    return embedder, store


def retrieve_relevant_cv_chunks(
    store: VectorStore,
    embedder: EmbeddingModel,
    query_text: str,
    k: int = 6,
) -> List[Dict]:
    q = embedder.embed([query_text])
    return store.search(q, k=k)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Embeddings are already normalized, but keep safe behavior.
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
