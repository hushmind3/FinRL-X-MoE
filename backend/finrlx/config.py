from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
MODEL_DIR = Path(os.getenv("FINRLX_MODEL_DIR", Path.home() / "Desktop" / "모델")).expanduser()
CHAMPION_PATH = Path(os.getenv("FINRLX_CHAMPION_PATH", MODEL_DIR / "champion.pt")).expanduser()
EXPERTS_DIR = Path(os.getenv("FINRLX_EXPERTS_DIR", MODEL_DIR / "experts")).expanduser()
RUNTIME_DIR = Path(os.getenv("FINRLX_RUNTIME_DIR", ROOT / "runtime")).expanduser()
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
INITIAL_CAPITAL = float(os.getenv("FINRLX_INITIAL_CAPITAL", "10000000"))
TRANSACTION_COST = max(0.0, float(os.getenv("FINRLX_TRANSACTION_COST", "0.001")))
FEED_POLL_SECONDS = max(2.0, float(os.getenv("FINRLX_FEED_POLL_SECONDS", "5")))
