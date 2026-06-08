#!/usr/bin/env python3
"""Creator Outreach RAG — CLI interface.

Semantic search over creator profiles and AI-powered outreach email generation.

Usage examples:
    python app.py build-index
    python app.py search "beauty creators who do skincare tutorials" --top-k 5
    python app.py search "fitness" --min-engagement 8 --platform tiktok
    python app.py outreach --creator @glowwithrae --campaign "Summer Launch" --tone casual
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from src import config
from src.embeddings import CreatorIndex
from src.filters import FilterEngine
from src.outreach import OutreachGenerator, Tone

logger = logging.getLogger("creator_rag")


# ── helpers ──────────────────────────────────────────────────────────────────


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _format_number(n: int | float) -> str:
    """Human-readable number: 1234567 → 1.2M, 45000 → 45.0K."""
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n))


def _load_index() -> CreatorIndex:
    """Load the pre-built index or fail with a helpful message."""
    idx = CreatorIndex()
    try:
        idx.load()
    except FileNotFoundError:
        logger.error(
            "No index found.  Run  `python app.py build-index`  first."
        )
        sys.exit(1)
    return idx


# ── commands ─────────────────────────────────────────────────────────────────


def cmd_build_index(args: argparse.Namespace) -> None:
    """Build (or rebuild) the embedding index from the CSV data."""
    csv_path = Path(args.data)
    logger.info("Building index from %s …", csv_path)

    idx = CreatorIndex()
    idx.build_index(csv_path)
    idx.save()

    logger.info("✓ Index saved to %s  (%d creators)", config.INDEX_DIR, len(idx.metadata))


def cmd_search(args: argparse.Namespace) -> None:
    """Semantic search with optional structured pre-filters."""
    idx = _load_index()

    # ── apply pre-filters if any were specified ──────────────────────────
    df = pd.DataFrame(idx.metadata)
    engine = FilterEngine(df)

    if args.niche:
        engine.filter_by_niche(args.niche)
    if args.platform:
        engine.filter_by_platform(args.platform)
    if args.country:
        engine.filter_by_country(args.country)
    if args.min_engagement is not None:
        engine.filter_by_engagement(min_rate=args.min_engagement)
    if args.min_followers is not None or args.max_followers is not None:
        engine.filter_by_followers(
            min_followers=args.min_followers or 0,
            max_followers=args.max_followers,
        )

    candidates = engine.indices if engine.count < len(df) else None
    if candidates is not None:
        logger.info("Pre-filter narrowed pool to %d creators", len(candidates))
        if not candidates:
            logger.warning("No creators match the applied filters.")
            return

    # ── semantic search ──────────────────────────────────────────────────
    results = idx.search(args.query, top_k=args.top_k, candidate_indices=candidates)

    # ── display ──────────────────────────────────────────────────────────
    rows = []
    for r in results:
        c = r.creator
        rows.append(
            {
                "Score": f"{r.score:.3f}",
                "Username": c.get("username", ""),
                "Platform": c.get("platform", ""),
                "Niche": c.get("niche", ""),
                "Followers": _format_number(c.get("followers", 0)),
                "Eng %": f"{c.get('engagement_rate', 0)}%",
                "Country": c.get("country", ""),
            }
        )

    print(f"\n🔍  Top {len(rows)} results for: \"{args.query}\"\n")
    print(tabulate(rows, headers="keys", tablefmt="rounded_outline"))
    print()


def cmd_outreach(args: argparse.Namespace) -> None:
    """Generate a personalised outreach email for a creator."""
    idx = _load_index()

    # Find the creator by username
    target = args.creator.lstrip("@")
    creator = None
    for meta in idx.metadata:
        if meta.get("username", "").lstrip("@").lower() == target.lower():
            creator = meta
            break

    if creator is None:
        logger.error("Creator '@%s' not found in the index.", target)
        sys.exit(1)

    tone = Tone(args.tone)

    try:
        gen = OutreachGenerator()
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if args.template == "affiliate":
        email = gen.affiliate_invitation(creator, campaign=args.campaign, tone=tone)
    elif args.template == "collaboration":
        email = gen.collaboration_proposal(
            creator,
            brand_brief=args.campaign,
            tone=tone,
        )
    elif args.template == "follow_up":
        email = gen.follow_up(
            creator,
            previous_date=args.previous_date or "last week",
            tone=tone,
        )
    else:
        email = gen.affiliate_invitation(creator, campaign=args.campaign, tone=tone)

    print(f"\n📧  Outreach email for {creator.get('username')}\n")
    print(email)
    print()


# ── argument parser ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="creator-rag",
        description="Semantic search and AI-powered outreach for influencer marketing.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # build-index ─────────────────────────────────────────────────────────
    p_build = sub.add_parser("build-index", help="Build the embedding index from CSV data")
    p_build.add_argument(
        "--data",
        default=str(config.DEFAULT_CSV),
        help="Path to creator CSV (default: data/creators.csv)",
    )

    # search ──────────────────────────────────────────────────────────────
    p_search = sub.add_parser("search", help="Semantic search for creators")
    p_search.add_argument("query", help="Natural-language search query")
    p_search.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K, help="Number of results")
    p_search.add_argument("--niche", help="Filter by niche (beauty, fitness, food, …)")
    p_search.add_argument("--platform", help="Filter by platform (tiktok, instagram, youtube, …)")
    p_search.add_argument("--country", help="Filter by country code (US, UK, …)")
    p_search.add_argument("--min-engagement", type=float, help="Minimum engagement rate (%%)")
    p_search.add_argument("--min-followers", type=int, help="Minimum follower count")
    p_search.add_argument("--max-followers", type=int, help="Maximum follower count")

    # outreach ────────────────────────────────────────────────────────────
    p_out = sub.add_parser("outreach", help="Generate an outreach email for a creator")
    p_out.add_argument("--creator", required=True, help="Creator username (e.g. @glowwithrae)")
    p_out.add_argument("--campaign", required=True, help="Campaign name or brief")
    p_out.add_argument(
        "--tone",
        choices=["formal", "casual", "friendly"],
        default="friendly",
        help="Email tone (default: friendly)",
    )
    p_out.add_argument(
        "--template",
        choices=["affiliate", "collaboration", "follow_up"],
        default="affiliate",
        help="Email template type (default: affiliate)",
    )
    p_out.add_argument("--previous-date", help="Date of previous outreach (for follow_up template)")

    return parser


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    commands = {
        "build-index": cmd_build_index,
        "search": cmd_search,
        "outreach": cmd_outreach,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
