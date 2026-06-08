"""Unit tests for the FilterEngine."""

from __future__ import annotations

import pandas as pd
import pytest

from src.filters import FilterEngine

# ── fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_DATA = pd.DataFrame(
    [
        {
            "username": "@beauty_a",
            "niche": "beauty",
            "sub_niche": "skincare",
            "platform": "tiktok",
            "followers": 500_000,
            "engagement_rate": 9.5,
            "country": "US",
            "tiktok_shop_enabled": "yes",
        },
        {
            "username": "@beauty_b",
            "niche": "beauty",
            "sub_niche": "makeup",
            "platform": "instagram",
            "followers": 120_000,
            "engagement_rate": 12.0,
            "country": "UK",
            "tiktok_shop_enabled": "no",
        },
        {
            "username": "@fitness_a",
            "niche": "fitness",
            "sub_niche": "yoga",
            "platform": "youtube",
            "followers": 300_000,
            "engagement_rate": 7.0,
            "country": "US",
            "tiktok_shop_enabled": "no",
        },
        {
            "username": "@tech_a",
            "niche": "tech",
            "sub_niche": "gadgets",
            "platform": "youtube",
            "followers": 1_000_000,
            "engagement_rate": 5.0,
            "country": "IN",
            "tiktok_shop_enabled": "no",
        },
        {
            "username": "@food_a",
            "niche": "food",
            "sub_niche": "recipe",
            "platform": "tiktok",
            "followers": 250_000,
            "engagement_rate": 11.0,
            "country": "US",
            "tiktok_shop_enabled": "yes",
        },
    ]
)


@pytest.fixture
def engine() -> FilterEngine:
    """Return a fresh FilterEngine with sample data."""
    return FilterEngine(SAMPLE_DATA)


# ── niche ────────────────────────────────────────────────────────────────────


class TestFilterByNiche:
    def test_filters_correct_niche(self, engine: FilterEngine) -> None:
        result = engine.filter_by_niche("beauty")
        assert result.count == 2
        usernames = result.filtered_df["username"].tolist()
        assert "@beauty_a" in usernames
        assert "@beauty_b" in usernames

    def test_case_insensitive(self, engine: FilterEngine) -> None:
        assert engine.filter_by_niche("BEAUTY").count == 2

    def test_no_matches_returns_empty(self, engine: FilterEngine) -> None:
        assert engine.filter_by_niche("gaming").count == 0


# ── sub_niche ────────────────────────────────────────────────────────────────


class TestFilterBySubNiche:
    def test_filters_sub_niche(self, engine: FilterEngine) -> None:
        result = engine.filter_by_sub_niche("skincare")
        assert result.count == 1
        assert result.filtered_df.iloc[0]["username"] == "@beauty_a"


# ── platform ─────────────────────────────────────────────────────────────────


class TestFilterByPlatform:
    def test_filters_platform(self, engine: FilterEngine) -> None:
        result = engine.filter_by_platform("youtube")
        assert result.count == 2

    def test_case_insensitive(self, engine: FilterEngine) -> None:
        assert engine.filter_by_platform("TikTok").count == 2


# ── engagement ───────────────────────────────────────────────────────────────


class TestFilterByEngagement:
    def test_min_rate(self, engine: FilterEngine) -> None:
        result = engine.filter_by_engagement(min_rate=10.0)
        assert result.count == 2  # beauty_b (12.0) and food_a (11.0)

    def test_range(self, engine: FilterEngine) -> None:
        result = engine.filter_by_engagement(min_rate=6.0, max_rate=8.0)
        assert result.count == 1  # fitness_a (7.0)

    def test_no_matches(self, engine: FilterEngine) -> None:
        result = engine.filter_by_engagement(min_rate=50.0)
        assert result.count == 0


# ── followers ────────────────────────────────────────────────────────────────


class TestFilterByFollowers:
    def test_min_only(self, engine: FilterEngine) -> None:
        result = engine.filter_by_followers(min_followers=400_000)
        assert result.count == 2  # beauty_a, tech_a

    def test_min_and_max(self, engine: FilterEngine) -> None:
        result = engine.filter_by_followers(min_followers=200_000, max_followers=500_000)
        assert result.count == 3  # beauty_a, fitness_a, food_a


# ── country ──────────────────────────────────────────────────────────────────


class TestFilterByCountry:
    def test_filters_country(self, engine: FilterEngine) -> None:
        result = engine.filter_by_country("US")
        assert result.count == 3

    def test_case_insensitive(self, engine: FilterEngine) -> None:
        assert engine.filter_by_country("us").count == 3


# ── tiktok shop ──────────────────────────────────────────────────────────────


class TestFilterByTikTokShop:
    def test_enabled(self, engine: FilterEngine) -> None:
        result = engine.filter_by_tiktok_shop(enabled=True)
        assert result.count == 2

    def test_disabled(self, engine: FilterEngine) -> None:
        result = engine.filter_by_tiktok_shop(enabled=False)
        assert result.count == 3


# ── chaining ─────────────────────────────────────────────────────────────────


class TestChaining:
    def test_niche_and_platform(self, engine: FilterEngine) -> None:
        result = engine.filter_by_niche("beauty").filter_by_platform("tiktok")
        assert result.count == 1
        assert result.filtered_df.iloc[0]["username"] == "@beauty_a"

    def test_chain_with_engagement(self, engine: FilterEngine) -> None:
        result = (
            engine
            .filter_by_country("US")
            .filter_by_engagement(min_rate=9.0)
        )
        assert result.count == 2  # beauty_a (9.5), food_a (11.0)

    def test_chain_to_empty(self, engine: FilterEngine) -> None:
        result = (
            engine
            .filter_by_niche("tech")
            .filter_by_country("US")
        )
        assert result.count == 0


# ── reset ────────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_filters(self, engine: FilterEngine) -> None:
        engine.filter_by_niche("beauty")
        assert engine.count == 2
        engine.reset()
        assert engine.count == 5


# ── indices property ─────────────────────────────────────────────────────────


class TestIndices:
    def test_returns_correct_indices(self, engine: FilterEngine) -> None:
        indices = engine.filter_by_niche("fitness").indices
        assert indices == [2]

    def test_all_indices_when_no_filter(self, engine: FilterEngine) -> None:
        assert engine.indices == [0, 1, 2, 3, 4]
