from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.backend = "lexical_fallback"
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.backend = "sentence_transformers"
        except Exception:
            self.model = None

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            emb = self.model.encode(texts, normalize_embeddings=True)
            return np.asarray(emb, dtype="float32")

        # Fallback: deterministic hashed bag-of-words embedding (local, no extra deps).
        vectors = np.zeros((len(texts), 384), dtype="float32")
        token_re = re.compile(r"[a-zA-Z0-9+#.-]+")
        for i, text in enumerate(texts):
            tokens = token_re.findall((text or "").lower())
            if not tokens:
                continue
            for tok in tokens:
                idx = (hash(tok) % 384 + 384) % 384
                vectors[i, idx] += 1.0
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


class LocalVectorStore:
    def __init__(self, dim: int = 384, chroma_path: str = ".chroma_local"):
        self.dim = dim
        self.kind = ""
        self._faiss_index = None
        self._chroma_collection = None
        self.metas: List[Dict] = []
        self._init_backend(chroma_path)

    def _init_backend(self, chroma_path: str) -> None:
        try:
            import faiss  # type: ignore

            self._faiss_index = faiss.IndexFlatIP(self.dim)
            self.kind = "faiss"
            return
        except Exception:
            pass

        import chromadb

        client = chromadb.PersistentClient(path=chroma_path)
        self._chroma_collection = client.get_or_create_collection("cv_chunks")
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
        self._chroma_collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=docs,
            metadatas=metas,
        )
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

        res = self._chroma_collection.query(query_embeddings=query_embedding.tolist(), n_results=k)
        out: List[Dict] = []
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for meta, dist in zip(metas, dists):
            m = dict(meta)
            m["score"] = float(1.0 / (1.0 + float(dist)))
            out.append(m)
        return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_profile_store(chunks: List[Dict], embedder: Embedder | None = None) -> Tuple[Embedder, LocalVectorStore]:
    embedder = embedder or Embedder()
    store = LocalVectorStore(dim=384)
    texts = [c["text"] for c in chunks]
    embs = embedder.embed(texts) if texts else np.zeros((0, 384), dtype="float32")
    store.add(embs, chunks)
    return embedder, store


def retrieve_cv_evidence(store: LocalVectorStore, embedder: Embedder, query: str, k: int = 6) -> List[Dict]:
    q = embedder.embed([query])
    return store.search(q, k=k)


def _keyword_presence_score(job_text: str, interests_keywords: List[str], profile_skills: List[str]) -> Tuple[int, List[str]]:
    low = (job_text or "").lower()
    matched = []
    for kw in set([*interests_keywords, *profile_skills]):
        if not kw:
            continue
        if re.search(rf"(?<!\w){re.escape(kw.lower())}(?!\w)", low):
            matched.append(kw.lower())
    bonus = min(20, len(set(matched)) * 3)
    return bonus, sorted(set(matched))


def rank_jobs(
    jobs: List[Dict],
    profile_embedding: np.ndarray,
    embedder: Embedder,
    profile_skills: List[str],
    interests_keywords: List[str],
) -> List[Dict]:
    ranked = []
    for idx, job in enumerate(jobs):
        desc = job.get("description_text", "")
        if not desc:
            continue
        j_emb = embedder.embed([desc])[0]
        sim = max(0.0, _cosine(profile_embedding, j_emb))
        sim_score = min(80, int(round(sim * 80)))
        kw_bonus, kw_matched = _keyword_presence_score(desc, interests_keywords, profile_skills)
        total = min(100, sim_score + kw_bonus)
        row = dict(job)
        row.update(
            {
                "job_id": idx,
                "match_score": total,
                "semantic_similarity": round(sim, 4),
                "keyword_bonus": kw_bonus,
                "matched_keywords": kw_matched,
            }
        )
        ranked.append(row)

    ranked.sort(key=lambda x: x["match_score"], reverse=True)
    return ranked
