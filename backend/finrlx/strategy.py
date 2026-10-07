from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

try:
    from strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult
except ImportError as exc:
    _FINRL_IMPORT_ERROR = exc
    BaseStrategy = object
    StrategyConfig = StrategyResult = None
else:
    _FINRL_IMPORT_ERROR = None

TARGET_WEIGHT_STRATEGY_NAME = "TorchRL Target Weight Policy"

class TargetWeightStrategy(BaseStrategy):
    """FinRL-X strategy adapter; target_weights is the strategy/execution contract."""
    def __init__(self) -> None:
        if _FINRL_IMPORT_ERROR:
            raise RuntimeError("FinRL-Trading 전략 API를 불러오지 못했습니다.") from _FINRL_IMPORT_ERROR
        super().__init__(StrategyConfig(name=TARGET_WEIGHT_STRATEGY_NAME))
        self._weights = pd.DataFrame(columns=["gvkey", "weight"])
        self._metadata: dict[str, Any] = {}

    def set_target_weights(self, weights: dict[str, float], metadata: dict[str, Any] | None = None) -> None:
        clean = {str(k).upper(): float(v) for k, v in weights.items()}
        if any(not np.isfinite(v) or v < 0 for v in clean.values()) or sum(clean.values()) > 1 + 1e-8:
            raise ValueError("실제 종목 weight는 유한한 0 이상 값이며 합계는 1 이하여야 합니다.")
        # Cash is the unallocated residual and never appears as a ticker row.
        # A zero-weight row closes an existing holding; with no holding it
        # produces no order in the execution layer.
        self._weights = pd.DataFrame(
            [{"gvkey": k, "weight": v} for k, v in clean.items()],
            columns=["gvkey", "weight"],
        )
        self._metadata = {**(metadata or {}), "cash_weight": max(0.0, 1.0 - sum(clean.values()))}

    def generate_weights(self, data: dict[str, pd.DataFrame] | None = None,
                         target_date: str | None = None) -> StrategyResult:
        return StrategyResult(self.config.name, self._weights.copy(),
                              {**self._metadata, "target_date": target_date})
