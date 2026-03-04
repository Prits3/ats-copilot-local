from typing import List, Dict, Tuple
import re
from pypdf import PdfReader
from collections import Counter
import math


def extract_pdf_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        pages.append((p.extract_text() or "").strip())
    return pages


def extract_pdf_text(pdf_path: str) -> str:
    return "\n".join(extract_pdf_pages(pdf_path)).strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


class LocalRAG:
    def __init__(self):
        pass

    def build_chunks_for_source(
        self, source: str, pages: List[str], chunk_size: int = 1000, overlap: int = 150
    ) -> List[Dict]:
        metas: List[Dict] = []
        for page_i, page_text in enumerate(pages):
            for ci, ch in enumerate(chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)):
                metas.append(
                    {
                        "source": source,
                        "page": page_i + 1,
                        "chunk_id": f"{source}_p{page_i+1}_c{ci}",
                        "text": ch,
                    }
                )
        return metas

    def build_index_from_chunks(self, chunks: List[Dict]) -> Tuple[List[Counter], List[Dict]]:
        if not chunks:
            return [], []
        vectors = [self._to_counter(c["text"]) for c in chunks]
        return vectors, chunks

    def build_index(self, pages: List[str]) -> Tuple[List[Counter], List[Dict]]:
        chunks = self.build_chunks_for_source("cv", pages)
        return self.build_index_from_chunks(chunks)

    def retrieve(self, index: List[Counter], metas: List[Dict], query: str, k: int = 6) -> List[Dict]:
        if not index:
            return []
        q_vec = self._to_counter(query)
        scored = []
        for idx, c_vec in enumerate(index):
            score = self._cosine_counter(q_vec, c_vec)
            if score > 0:
                scored.append((score, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, idx in scored[:k]:
            m = metas[idx].copy()
            m["score"] = float(score)
            out.append(m)
        return out

    def _to_counter(self, text: str) -> Counter:
        tokens = re.findall(r"[a-zA-Z0-9+#.-]+", text.lower())
        return Counter(tokens)

    def _cosine_counter(self, a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b.get(t, 0) for t in a)
        if dot == 0:
            return 0.0
        a_norm = math.sqrt(sum(v * v for v in a.values()))
        b_norm = math.sqrt(sum(v * v for v in b.values()))
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return dot / (a_norm * b_norm)
