"""LLM-powered outreach email generation.

Uses the Anthropic Claude API to draft personalised creator outreach
emails.  Supports multiple templates and tone options.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any

from anthropic import APIError, Anthropic

from src import config

logger = logging.getLogger(__name__)


class Tone(str, Enum):
    """Outreach email tone."""

    FORMAL = "formal"
    CASUAL = "casual"
    FRIENDLY = "friendly"


# ── prompt templates ─────────────────────────────────────────────────────────

_AFFILIATE_PROMPT = """\
You are an influencer marketing manager.
Write a personalized affiliate-program invitation email to the creator below.

Creator profile
────────────────
Username: {username}
Display name: {display_name}
Platform: {platform}
Niche: {niche} / {sub_niche}
Followers: {followers:,}
Engagement rate: {engagement_rate}%
Past brand partners: {past_brands}
Content style: {content_style}
Bio: {bio_summary}

Campaign details
────────────────
Campaign: {campaign}

Guidelines
──────────
• Tone: {tone}
• Reference specific details from their content to show you actually watch them.
• Keep it under 200 words.
• Include a clear call-to-action.
• Do NOT use generic flattery.
"""

_COLLABORATION_PROMPT = """\
You are an influencer marketing manager.
Write a collaboration proposal email to the creator below.

Creator profile
────────────────
Username: {username}
Display name: {display_name}
Platform: {platform}
Niche: {niche} / {sub_niche}
Followers: {followers:,}
Engagement rate: {engagement_rate}%
Past brand partners: {past_brands}
Content style: {content_style}
Bio: {bio_summary}

Brand brief
───────────
{brand_brief}

Guidelines
──────────
• Tone: {tone}
• Reference specific details from their content.
• Outline deliverables, timeline expectations, and compensation structure.
• Keep it under 250 words.
"""

_FOLLOWUP_PROMPT = """\
You are an influencer marketing manager.
Write a polite follow-up email to a creator you previously reached out to.

Creator profile
────────────────
Username: {username}
Display name: {display_name}
Platform: {platform}
Niche: {niche} / {sub_niche}
Bio: {bio_summary}

Previous outreach date: {previous_date}

Guidelines
──────────
• Tone: {tone}
• Be respectful of their time.
• Add a new incentive or piece of information to re-engage.
• Keep it under 150 words.
"""


class OutreachGenerator:
    """Generate personalised outreach emails via the Claude API.

    Parameters:
        api_key: Anthropic API key. Falls back to ``config.ANTHROPIC_API_KEY``.
        model: Model identifier. Falls back to ``config.ANTHROPIC_MODEL``.
        max_retries: Number of retries on transient API errors.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key or config.ANTHROPIC_API_KEY
        if not self._api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY in your .env file."
            )
        self._client = Anthropic(api_key=self._api_key)
        self._model = model or config.ANTHROPIC_MODEL
        self._max_retries = max_retries

    # ── internal ─────────────────────────────────────────────────────────
    def _call_llm(self, prompt: str) -> str:
        """Send a prompt to Claude with exponential-backoff retry."""
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=config.MAX_TOKENS,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except APIError as exc:
                wait = 2**attempt
                logger.warning(
                    "Anthropic API error (attempt %d/%d): %s — retrying in %ds",
                    attempt,
                    self._max_retries,
                    exc,
                    wait,
                )
                if attempt == self._max_retries:
                    raise
                time.sleep(wait)
        # unreachable, but keeps mypy happy
        raise RuntimeError("LLM call failed after retries")  # pragma: no cover

    @staticmethod
    def _creator_kwargs(creator: dict[str, Any]) -> dict[str, Any]:
        """Normalise a creator dict into template-friendly kwargs."""
        return {
            "username": creator.get("username", "unknown"),
            "display_name": creator.get("display_name", ""),
            "platform": creator.get("platform", ""),
            "niche": creator.get("niche", ""),
            "sub_niche": creator.get("sub_niche", ""),
            "followers": int(creator.get("followers", 0)),
            "engagement_rate": creator.get("engagement_rate", ""),
            "past_brands": creator.get("past_brands", ""),
            "content_style": creator.get("content_style", ""),
            "bio_summary": creator.get("bio_summary", ""),
        }

    # ── public API ───────────────────────────────────────────────────────
    def affiliate_invitation(
        self,
        creator: dict[str, Any],
        campaign: str,
        tone: Tone = Tone.FRIENDLY,
    ) -> str:
        """Generate an affiliate-program invitation email.

        Args:
            creator: Creator metadata dict (from CSV / search result).
            campaign: Short campaign name or description.
            tone: Desired email tone.

        Returns:
            The generated email body as a string.
        """
        prompt = _AFFILIATE_PROMPT.format(
            **self._creator_kwargs(creator),
            campaign=campaign,
            tone=tone.value,
        )
        logger.info("Generating affiliate invitation for %s", creator.get("username"))
        return self._call_llm(prompt)

    def collaboration_proposal(
        self,
        creator: dict[str, Any],
        brand_brief: str,
        tone: Tone = Tone.FORMAL,
    ) -> str:
        """Generate a brand-collaboration proposal email.

        Args:
            creator: Creator metadata dict.
            brand_brief: Detailed brand/campaign brief.
            tone: Desired email tone.

        Returns:
            The generated email body as a string.
        """
        prompt = _COLLABORATION_PROMPT.format(
            **self._creator_kwargs(creator),
            brand_brief=brand_brief,
            tone=tone.value,
        )
        logger.info("Generating collaboration proposal for %s", creator.get("username"))
        return self._call_llm(prompt)

    def follow_up(
        self,
        creator: dict[str, Any],
        previous_date: str,
        tone: Tone = Tone.FRIENDLY,
    ) -> str:
        """Generate a follow-up email referencing a prior outreach.

        Args:
            creator: Creator metadata dict.
            previous_date: Date string of the original outreach.
            tone: Desired email tone.

        Returns:
            The generated email body as a string.
        """
        prompt = _FOLLOWUP_PROMPT.format(
            **self._creator_kwargs(creator),
            previous_date=previous_date,
            tone=tone.value,
        )
        logger.info("Generating follow-up for %s", creator.get("username"))
        return self._call_llm(prompt)
