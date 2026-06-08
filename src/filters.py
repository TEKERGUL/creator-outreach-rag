"""Structured pre-filters for creator metadata.

Filters are applied *before* vector search to narrow the candidate set,
improving both relevance and speed.  Filters are chainable::

    engine = FilterEngine(df)
    indices = (
        engine
        .filter_by_niche("beauty")
        .filter_by_platform("tiktok")
        .filter_by_engagement(min_rate=8.0)
        .indices
    )
"""

from __future__ import annotations

import logging
from typing import Self

import pandas as pd

logger = logging.getLogger(__name__)


class FilterEngine:
    """Chainable filter interface over a creator DataFrame.

    Parameters:
        df: The full creator DataFrame.
        mask: Boolean mask matching ``df`` rows.  Defaults to all-True.
    """

    def __init__(self, df: pd.DataFrame, mask: pd.Series | None = None) -> None:
        self._df = df
        self._mask: pd.Series = mask if mask is not None else pd.Series(True, index=df.index)

    # ── properties ───────────────────────────────────────────────────────
    @property
    def indices(self) -> list[int]:
        """Row indices of creators that pass all applied filters."""
        return self._mask[self._mask].index.tolist()

    @property
    def filtered_df(self) -> pd.DataFrame:
        """Return the filtered DataFrame subset."""
        return self._df.loc[self._mask].copy()

    @property
    def count(self) -> int:
        """Number of creators that pass all applied filters."""
        return int(self._mask.sum())

    # ── filter methods (each returns self for chaining) ──────────────────
    def filter_by_niche(self, niche: str) -> Self:
        """Keep only creators whose *niche* matches (case-insensitive)."""
        col = self._df["niche"].str.lower()
        self._mask &= col == niche.strip().lower()
        logger.debug("filter_by_niche('%s') → %d remaining", niche, self.count)
        return self

    def filter_by_sub_niche(self, sub_niche: str) -> Self:
        """Keep only creators whose *sub_niche* matches (case-insensitive)."""
        col = self._df["sub_niche"].str.lower()
        self._mask &= col == sub_niche.strip().lower()
        logger.debug("filter_by_sub_niche('%s') → %d remaining", sub_niche, self.count)
        return self

    def filter_by_platform(self, platform: str) -> Self:
        """Keep only creators on the given *platform*."""
        col = self._df["platform"].str.lower()
        self._mask &= col == platform.strip().lower()
        logger.debug("filter_by_platform('%s') → %d remaining", platform, self.count)
        return self

    def filter_by_engagement(self, min_rate: float = 0.0, max_rate: float = 100.0) -> Self:
        """Keep creators whose engagement rate falls within [min_rate, max_rate]."""
        col = pd.to_numeric(self._df["engagement_rate"], errors="coerce").fillna(0.0)
        self._mask &= (col >= min_rate) & (col <= max_rate)
        logger.debug(
            "filter_by_engagement(%.1f–%.1f) → %d remaining",
            min_rate,
            max_rate,
            self.count,
        )
        return self

    def filter_by_followers(
        self,
        min_followers: int = 0,
        max_followers: int | None = None,
    ) -> Self:
        """Keep creators whose follower count is within the given range."""
        col = pd.to_numeric(self._df["followers"], errors="coerce").fillna(0)
        self._mask &= col >= min_followers
        if max_followers is not None:
            self._mask &= col <= max_followers
        logger.debug(
            "filter_by_followers(%d–%s) → %d remaining",
            min_followers,
            max_followers or "∞",
            self.count,
        )
        return self

    def filter_by_country(self, country: str) -> Self:
        """Keep only creators from a specific country code (e.g. 'US')."""
        col = self._df["country"].str.upper()
        self._mask &= col == country.strip().upper()
        logger.debug("filter_by_country('%s') → %d remaining", country, self.count)
        return self

    def filter_by_tiktok_shop(self, enabled: bool = True) -> Self:
        """Keep creators based on TikTok Shop status."""
        col = self._df["tiktok_shop_enabled"].str.lower()
        target = "yes" if enabled else "no"
        self._mask &= col == target
        logger.debug("filter_by_tiktok_shop(%s) → %d remaining", enabled, self.count)
        return self

    def reset(self) -> Self:
        """Clear all filters."""
        self._mask = pd.Series(True, index=self._df.index)
        return self
