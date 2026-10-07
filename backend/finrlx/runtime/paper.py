from __future__ import annotations

import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from ..config import INITIAL_CAPITAL, RUNTIME_DIR, TRANSACTION_COST

class PaperAccount:
    """Fractional-share paper ledger; unallocated equity remains cash."""
    def __init__(self, path: Path = RUNTIME_DIR / "paper_account.json") -> None:
        self.path = path
        self.lock = threading.RLock()
        self.cash = INITIAL_CAPITAL
        self.positions: dict[str, float] = {}
        self.prices: dict[str, float] = {}
        self.fills: list[dict] = []
        self.enabled = False
        self.error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            self.cash = float(state.get("cash", INITIAL_CAPITAL))
            self.positions = {str(k): float(v) for k, v in state.get("positions", {}).items()}
            self.prices = {str(k): float(v) for k, v in state.get("prices", {}).items()}
            self.fills = list(state.get("fills", []))[-2000:]
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.error = f"paper 계좌 파일 읽기 실패: {exc}"

    def mark(self, symbol: str, price: float) -> None:
        if price > 0:
            self.prices[symbol] = price

    def rebalance(self, weights: dict[str, float], timestamp: str) -> dict:
        with self.lock:
            equity_before = self.equity()
            target = {s: float(w) for s, w in weights.items()}
            if any(not math.isfinite(weight) or weight < 0 for weight in target.values()):
                raise ValueError("paper 목표 비중은 유한한 0 이상 값이어야 합니다.")
            if sum(target.values()) > 1 + 1e-8:
                raise ValueError("실제 종목 목표 비중 합계는 1 이하여야 합니다.")
            created = []
            for symbol in set(self.positions) | set(target):
                price = self.prices.get(symbol, 0.0)
                if price <= 0:
                    if self.positions.get(symbol, 0.0) != 0 or target.get(symbol, 0.0) > 0:
                        raise ValueError(f"{symbol} 목표 주문을 낼 현재 가격이 없습니다.")
                    continue
                old_value = self.positions.get(symbol, 0.0) * price
                target_value = equity_before * target.get(symbol, 0.0)
                delta = target_value - old_value
                if abs(delta) <= 1e-10:
                    continue
                fee = abs(delta) * TRANSACTION_COST
                self.cash -= delta + fee
                self.positions[symbol] = target_value / price
                fill = {"timestamp": timestamp, "symbol": symbol, "side": "BUY" if delta > 0 else "SELL",
                        "price": price, "notional": abs(delta), "fee": fee, "slippage": 0.0}
                self.fills.append(fill)
                created.append(fill)
            self.fills = self.fills[-2000:]
            self._save()
            return {"fills": created, "equity": self.equity(), "cash": self.cash}

    def rebalance_strategy_result(self, result, timestamp: str) -> dict:
        """Execute the official FinRL-X StrategyResult weight contract locally."""
        rows = getattr(result, "weights", None)
        if rows is None or not {"gvkey", "weight"}.issubset(rows.columns):
            raise ValueError("FinRL StrategyResult에는 gvkey와 weight 열이 필요합니다.")
        weights = {str(row.gvkey): float(row.weight) for row in rows.itertuples(index=False)}
        return self.rebalance(weights, timestamp)

    def equity(self) -> float:
        return self.cash + sum(qty * self.prices.get(symbol, 0.0) for symbol, qty in self.positions.items())

    def snapshot(self) -> dict:
        equity = self.equity()
        holdings = {symbol: qty * self.prices.get(symbol, 0.0) for symbol, qty in self.positions.items() if qty != 0}
        return {"enabled": self.enabled, "cash": self.cash, "equity": equity,
                "return_pct": (equity / INITIAL_CAPITAL - 1) * 100 if INITIAL_CAPITAL else 0,
                "positions": holdings,
                "weights": {k: v / equity for k, v in holdings.items()} if equity > 0 else {},
                "fills": self.fills[-50:], "error": self.error}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"cash": self.cash, "positions": self.positions,
                                   "prices": self.prices, "fills": self.fills,
                                   "updated_at": datetime.now(timezone.utc).isoformat()},
                                  ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
