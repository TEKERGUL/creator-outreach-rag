"""Vector embedding index for semantic creator search.

Uses sentence-transformers (all-MiniLM-L6-v2) to encode creator profiles
and NumPy-based cosine similarity for retrieval — lightweight, no external
vector-DB dependency required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src import config

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with similarity score and creator metadata."""

    score: float
    creator: dict[str, Any]


@dataclass
class CreatorIndex:
    """Builds and queries a semantic embedding index over creator profiles.

    Attributes:
        model_name: HuggingFace model identifier for the sentence encoder.
        embeddings: (N, D) matrix of L2-normalised creator embeddings.
        metadata: List of creator record dicts aligned with *embeddings*.
    """

    model_name: str = config.EMBEDDING_MODEL
    embeddings: np.ndarray | None = None
    metadata: list[dict[str, Any]] = field(default_factory=list)
    _model: SentenceTransformer | None = field(default=None, repr=False)

    # ── helpers ──────────────────────────────────────────────────────────
    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            logger.info("Loading sentence-transformer model '%s' …", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _build_text(row: pd.Series) -> str:
        """Combine the most semantically-rich fields into one passage."""
        parts = [
            str(row.get("bio_summary", "")),
            f"niche: {row.get('niche', '')}",
            f"sub-niche: {row.get('sub_niche', '')}",
            f"platform: {row.get('platform', '')}",
            f"past brands: {row.get('past_brands', '')}",
            f"content style: {row.get('content_style', '')}",
            f"country: {row.get('country', '')}",
        ]
        return " | ".join(p for p in parts if p)

    # ── build ────────────────────────────────────────────────────────────
    def build_index(self, csv_path: str | Path) -> None:
        """Read a CSV of creator profiles and build the embedding matrix.

        Args:
            csv_path: Path to the creators CSV file.

        Raises:
            FileNotFoundError: If *csv_path* does not exist.
            ValueError: If the CSV is empty or missing required columns.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        logger.info("Reading creator data from %s", csv_path)
        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError("CSV file is empty.")

        required = {"username", "bio_summary", "niche"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        texts = df.apply(self._build_text, axis=1).tolist()
        logger.info("Encoding %d creator profiles …", len(texts))
        vectors = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        # L2-normalise so dot-product == cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # guard against zero vectors
        self.embeddings = vectors / norms

        self.metadata = df.to_dict(orient="records")
        logger.info("Index built — %d creators, dim=%d", len(self.metadata), self.embeddings.shape[1])

    # ── search ───────────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        candidate_indices: list[int] | None = None,
    ) -> list[SearchResult]:
        """Return the *top_k* creators most semantically similar to *query*.

        Args:
            query: Natural-language search string.
            top_k: Number of results to return.
            candidate_indices: If provided, restrict search to these row indices
                (useful when combining with pre-filters).

        Returns:
            A list of :class:`SearchResult` sorted by descending similarity.

        Raises:
            RuntimeError: If the index has not been built or loaded yet.
        """
        if self.embeddings is None or not self.metadata:
            raise RuntimeError("Index is empty — call build_index() or load() first.")

        query_vec = self.model.encode([query], convert_to_numpy=True)
        query_vec = query_vec / np.linalg.norm(query_vec)

        if candidate_indices is not None:
            emb_subset = self.embeddings[candidate_indices]
            scores = (emb_subset @ query_vec.T).flatten()
            top_local = np.argsort(scores)[::-1][:top_k]
            results = [
                SearchResult(score=float(scores[i]), creator=self.metadata[candidate_indices[i]])
                for i in top_local
            ]
        else:
            scores = (self.embeddings @ query_vec.T).flatten()
            top_idx = np.argsort(scores)[::-1][:top_k]
            results = [
                SearchResult(score=float(scores[i]), creator=self.metadata[i])
                for i in top_idx
            ]

        return results

    # ── persistence ──────────────────────────────────────────────────────
    def save(self, directory: str | Path | None = None) -> None:
        """Persist embeddings and metadata to disk as .npy files.

        Args:
            directory: Target directory (created if needed). Defaults to
                ``config.INDEX_DIR``.
        """
        if self.embeddings is None:
            raise RuntimeError("Nothing to save — build the index first.")

        directory = Path(directory or config.INDEX_DIR)
        directory.mkdir(parents=True, exist_ok=True)

        emb_path = directory / "embeddings.npy"
        meta_path = directory / "metadata.npy"

        np.save(emb_path, self.embeddings)
        np.save(meta_path, np.array(self.metadata, dtype=object))
        logger.info("Index saved to %s", directory)

    def load(self, directory: str | Path | None = None) -> None:
        """Load a previously-saved index from disk.

        Args:
            directory: Directory containing ``embeddings.npy`` and
                ``metadata.npy``. Defaults to ``config.INDEX_DIR``.

        Raises:
            FileNotFoundError: If the expected files are missing.
        """
        directory = Path(directory or config.INDEX_DIR)

        emb_path = directory / "embeddings.npy"
        meta_path = directory / "metadata.npy"

        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Index files not found in {directory}. Run 'build-index' first."
            )

        self.embeddings = np.load(emb_path)
        self.metadata = np.load(meta_path, allow_pickle=True).tolist()
        logger.info(
            "Index loaded — %d creators, dim=%d",
            len(self.metadata),
            self.embeddings.shape[1],
        )
