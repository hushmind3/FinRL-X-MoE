from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_KEYS = ("momentum_1", "momentum_5", "momentum_20", "volatility_20", "trend_20",
                "trend_60", "range_pct", "volume_z", "drawdown_60")

def add_market_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Create causal, point-in-time features; no row uses a future observation."""
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"필수 시세 열 누락: {', '.join(sorted(missing))}")
    result = []
    for symbol, frame in bars.groupby("symbol", sort=False):
        g = frame.sort_values("timestamp").copy()
        close = pd.to_numeric(g.close, errors="coerce").astype(float)
        volume = pd.to_numeric(g.volume, errors="coerce").fillna(0).astype(float)
        log_ret = np.log(close.where(close > 0)).diff().replace([np.inf, -np.inf], np.nan).fillna(0)
        g["momentum_1"] = close.pct_change().fillna(0).clip(-0.5, 0.5)
        g["momentum_5"] = close.pct_change(5).fillna(0).clip(-1, 1)
        g["momentum_20"] = close.pct_change(20).fillna(0).clip(-1, 1)
        g["volatility_20"] = log_ret.rolling(20, min_periods=5).std().fillna(0).clip(0, 1)
        g["trend_20"] = (close / close.rolling(20, min_periods=1).mean() - 1).fillna(0).clip(-1, 1)
        g["trend_60"] = (close / close.rolling(60, min_periods=1).mean() - 1).fillna(0).clip(-1, 1)
        g["range_pct"] = ((g.high - g.low) / close).replace([np.inf, -np.inf], 0).fillna(0).clip(0, 1)
        vol_mean = volume.rolling(20, min_periods=5).mean()
        vol_std = volume.rolling(20, min_periods=5).std().replace(0, np.nan)
        g["volume_z"] = ((volume - vol_mean) / vol_std).replace([np.inf, -np.inf], 0).fillna(0).clip(-5, 5)
        g["drawdown_60"] = (close / close.rolling(60, min_periods=1).max() - 1).fillna(0).clip(-1, 0)
        g["symbol"] = symbol
        result.append(g)
    return pd.concat(result, ignore_index=True) if result else pd.DataFrame()
