
from __future__ import annotations

import re
import uuid

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    """Hybrid retrieval using BM25, dense embeddings, and Reciprocal Rank Fusion."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.embedding_model_name = embedding_model
        self.embedder = SentenceTransformer(embedding_model)
        self.client = chromadb.Client()
        self.collection = None
        self.chunks: list[dict] = []
        self.bm25 = None

    @staticmethod
    def _tokens(text: str) -> list[str]:
        """Normalize text while preserving words, numbers, and decimals."""
        return re.findall(
            r"\d+(?:\.\d+)?%?|[a-z]+(?:['-][a-z]+)*",
            str(text).lower(),
        )

    def fit(self, chunks: list[dict]) -> None:
        """Index document chunks for keyword and semantic retrieval."""
        if not chunks:
            raise ValueError("Cannot index an empty set of chunks.")

        for i, chunk in enumerate(chunks):
            if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
                raise ValueError(f"Chunk {i} has missing or empty text.")

        # Tokenize once so BM25 and query processing use the same rules.
        tokenized_chunks = [self._tokens(c["text"]) for c in chunks]

        if not any(tokenized_chunks):
            raise ValueError("No searchable words were found in the chunks.")

        self.chunks = chunks
        self.bm25 = BM25Okapi(tokenized_chunks)

        # Use a fresh collection for each indexing run.
        collection_name = "research_" + uuid.uuid4().hex[:20]
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        embeddings = self.embedder.encode(
            [c["text"] for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        metadatas = [
            {
                "index": i,
                "source": str(c.get("source", "Unknown source")),
                "page": int(c.get("page", 0) or 0),
            }
            for i, c in enumerate(chunks)
        ]

        self.collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=[c["text"] for c in chunks],
            embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
            metadatas=metadatas,
        )

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """Retrieve and rank relevant chunks using reciprocal rank fusion."""
        if self.collection is None or self.bm25 is None or not self.chunks:
            raise RuntimeError("Call fit(chunks) before search(query).")

        if not isinstance(query, str) or not query.strip():
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_tokens = self._tokens(query)
        if not query_tokens:
            return []

        n = len(self.chunks)
        candidate_n = min(max(top_k * 4, top_k), n)

        # 1. Keyword retrieval. Ignore chunks with zero keyword overlap.
        bm25_scores = np.asarray(
            self.bm25.get_scores(query_tokens), dtype=float
        )
        bm25_order = np.argsort(bm25_scores)[::-1]
        bm25_order = [
            int(i)
            for i in bm25_order
            if bm25_scores[i] > 0
        ][:candidate_n]

        # 2. Semantic retrieval using dense embeddings.
        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        dense_result = self.collection.query(
            query_embeddings=[
                np.asarray(query_embedding, dtype=np.float32).tolist()
            ],
            n_results=candidate_n,
            include=["metadatas", "distances"],
        )

        dense_order = [
            int(metadata["index"])
            for metadata in dense_result["metadatas"][0]
        ]

        # 3. Reciprocal Rank Fusion (RRF).
        # A chunk may be retrieved by either method or both.
        rrf_scores: dict[int, float] = {}
        bm25_ranks: dict[int, int] = {}
        dense_ranks: dict[int, int] = {}
        rrf_constant = 60

        for rank, idx in enumerate(bm25_order, start=1):
            bm25_ranks[idx] = rank
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (
                1.0 / (rrf_constant + rank)
            )

        for rank, idx in enumerate(dense_order, start=1):
            dense_ranks[idx] = rank
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (
                1.0 / (rrf_constant + rank)
            )

        selected = sorted(
            rrf_scores,
            key=lambda idx: rrf_scores[idx],
            reverse=True,
        )[:top_k]

        if not selected:
            return []

        max_score = rrf_scores[selected[0]]

        # Return the original chunk fields to preserve app compatibility.
        results = []
        seen_texts = set()

        for idx in selected:
            chunk = self.chunks[idx]
            normalized_text = " ".join(
                str(chunk["text"]).lower().split()
            )

            # Avoid returning the same passage more than once.
            if normalized_text in seen_texts:
                continue
            seen_texts.add(normalized_text)

            results.append(
                {
                    **chunk,
                    "retrieval_score": float(
                        rrf_scores[idx] / max_score
                    ),
                    "rrf_score": float(rrf_scores[idx]),
                    "bm25_rank": bm25_ranks.get(idx),
                    "dense_rank": dense_ranks.get(idx),
                }
            )

        return results
