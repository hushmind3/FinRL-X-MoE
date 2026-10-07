from __future__ import annotations

import math
import os
from typing import Any

from .finrl_imports import ensure_finrl_source_layout


class FinRLAlpacaPaperExecutor:
    """Optional FinRL-X Alpaca paper execution through its weight API."""

    def __init__(self) -> None:
        self.mode = os.getenv("FINRLX_PAPER_EXECUTOR", "local").strip().lower()
        self._manager = None
        self._account_name: str | None = None
        self.last_result: dict[str, Any] | None = None
        self.last_error: str | None = None

    def status(self) -> dict[str, Any]:
        if self.mode == "local":
            return {"mode": "local_paper", "available": True, "last_error": self.last_error,
                    "last_result": self.last_result}
        if self.mode != "alpaca_paper":
            return {"mode": self.mode, "available": False,
                    "last_error": "FINRLX_PAPER_EXECUTOR must be local or alpaca_paper.",
                    "last_result": self.last_result}
        base_url = (os.getenv("APCA_BASE_URL") or "https://paper-api.alpaca.markets").lower().rstrip("/")
        configured = bool((os.getenv("APCA_API_KEY") or os.getenv("ALPACA_API_KEY")) and
                          (os.getenv("APCA_API_SECRET") or os.getenv("ALPACA_API_SECRET")))
        paper_url = "paper-api.alpaca.markets" in base_url
        reason = (None if configured and paper_url else
                  "APCA_API_KEY and APCA_API_SECRET are required." if not configured else
                  "APCA_BASE_URL must use https://paper-api.alpaca.markets.")
        return {"mode": "alpaca_paper", "available": configured and paper_url,
                "account_type": "paper" if paper_url else "blocked_live_url",
                "last_error": self.last_error or reason, "last_result": self.last_result}

    def execute_strategy_result(self, strategy_result) -> dict[str, Any] | None:
        if self.mode == "local":
            return None
        status = self.status()
        if not status["available"]:
            raise RuntimeError(status["last_error"] or
                               "FinRL Alpaca paper execution needs APCA paper credentials and the paper API URL.")

        weights = {}
        for row in strategy_result.weights.itertuples(index=False):
            symbol, weight = str(row.gvkey).upper(), float(row.weight)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("FinRL StrategyResult contains an invalid target weight.")
            weights[symbol] = weight
        if sum(weights.values()) > 1 + 1e-8:
            raise ValueError("FinRL StrategyResult ticker weights exceed account equity.")

        manager = self._get_manager()
        result = manager.execute_portfolio_rebalance(
            target_weights=weights,
            account_name=self._account_name,
            dry_run=False,
        )
        self.last_result = result
        self.last_error = None
        return result

    def _get_manager(self):
        if self._manager is not None:
            return self._manager
        ensure_finrl_source_layout()
        from trading.alpaca_manager import AlpacaManager, create_alpaca_account_from_env

        account = create_alpaca_account_from_env(name="finrlx-paper")
        account.base_url = account.base_url.rstrip("/")
        if account.base_url.endswith("/v2"):
            account.base_url = account.base_url[:-3]
        if "paper-api.alpaca.markets" not in account.base_url.lower():
            raise RuntimeError("실계좌 URL은 paper executor에서 허용하지 않습니다.")
        self._account_name = account.name
        self._manager = AlpacaManager([account])
        return self._manager
