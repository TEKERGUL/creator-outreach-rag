"""Centralized configuration for the Creator Outreach RAG system."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
INDEX_DIR: Path = PROJECT_ROOT / "index"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

DEFAULT_CSV: Path = DATA_DIR / "creators.csv"
EMBEDDINGS_FILE: Path = INDEX_DIR / "embeddings.npy"
METADATA_FILE: Path = INDEX_DIR / "metadata.npy"

# ── Embedding model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384  # output dimension for all-MiniLM-L6-v2

# ── LLM / Outreach ──────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
MAX_TOKENS: int = 1024

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K: int = 5
