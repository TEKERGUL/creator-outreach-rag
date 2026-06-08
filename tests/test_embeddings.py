"""Unit tests for the CreatorIndex embedding / search system."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.embeddings import CreatorIndex


# ── fixtures ─────────────────────────────────────────────────────────────────

MINI_CSV_CONTENT = """\
username,display_name,platform,niche,sub_niche,followers,engagement_rate,past_brands,content_style,country,bio_summary
@skincare_pro,Skin Pro,tiktok,beauty,skincare,200000,10.0,"CeraVe,Tatcha",tutorial,US,"Skincare expert who reviews serums and moisturizers with ingredient breakdowns."
@lift_heavy,Lift Heavy,youtube,fitness,weightlifting,300000,7.0,"Gymshark,MyProtein",educational,US,"Personal trainer sharing workout splits and form correction videos for strength training."
@pasta_king,Pasta King,tiktok,food,recipe,150000,11.0,"HelloFresh,KitchenAid",recipe,UK,"Italian home cook making pasta from scratch with simple weeknight recipes."
"""


@pytest.fixture
def mini_csv(tmp_path: Path) -> Path:
    """Write a small CSV and return its path."""
    csv_path = tmp_path / "creators.csv"
    csv_path.write_text(MINI_CSV_CONTENT)
    return csv_path


@pytest.fixture
def built_index(mini_csv: Path) -> CreatorIndex:
    """Return a CreatorIndex that has already been built."""
    idx = CreatorIndex()
    idx.build_index(mini_csv)
    return idx


# ── build_index ──────────────────────────────────────────────────────────────


class TestBuildIndex:
    def test_loads_correct_count(self, built_index: CreatorIndex) -> None:
        assert len(built_index.metadata) == 3

    def test_embeddings_shape(self, built_index: CreatorIndex) -> None:
        assert built_index.embeddings is not None
        assert built_index.embeddings.shape == (3, 384)

    def test_embeddings_are_normalized(self, built_index: CreatorIndex) -> None:
        norms = np.linalg.norm(built_index.embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_missing_file_raises(self) -> None:
        idx = CreatorIndex()
        with pytest.raises(FileNotFoundError):
            idx.build_index("/nonexistent/path.csv")

    def test_empty_csv_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.csv"
        empty.write_text("username,bio_summary,niche\n")
        idx = CreatorIndex()
        with pytest.raises(ValueError, match="empty"):
            idx.build_index(empty)


# ── search ───────────────────────────────────────────────────────────────────


class TestSearch:
    def test_returns_correct_top_k(self, built_index: CreatorIndex) -> None:
        results = built_index.search("skincare routine", top_k=2)
        assert len(results) == 2

    def test_skincare_query_ranks_skincare_first(self, built_index: CreatorIndex) -> None:
        results = built_index.search("skincare serums and moisturizer reviews", top_k=3)
        assert results[0].creator["username"] == "@skincare_pro"

    def test_fitness_query_ranks_fitness_first(self, built_index: CreatorIndex) -> None:
        results = built_index.search("weightlifting workout form", top_k=3)
        assert results[0].creator["username"] == "@lift_heavy"

    def test_scores_are_descending(self, built_index: CreatorIndex) -> None:
        results = built_index.search("cooking recipes", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_candidate_indices_restricts_search(self, built_index: CreatorIndex) -> None:
        results = built_index.search("skincare", top_k=3, candidate_indices=[1, 2])
        usernames = [r.creator["username"] for r in results]
        assert "@skincare_pro" not in usernames

    def test_empty_index_raises(self) -> None:
        idx = CreatorIndex()
        with pytest.raises(RuntimeError, match="empty"):
            idx.search("anything")


# ── save / load ──────────────────────────────────────────────────────────────


class TestPersistence:
    def test_roundtrip(self, built_index: CreatorIndex, tmp_path: Path) -> None:
        built_index.save(tmp_path)

        loaded = CreatorIndex()
        loaded.load(tmp_path)

        assert len(loaded.metadata) == len(built_index.metadata)
        np.testing.assert_array_almost_equal(loaded.embeddings, built_index.embeddings)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        idx = CreatorIndex()
        with pytest.raises(FileNotFoundError):
            idx.load(tmp_path / "nonexistent")
