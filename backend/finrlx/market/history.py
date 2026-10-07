from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

import pandas as pd
import yfinance as yf

from .features import add_market_features

def fetch_history(symbols: Sequence[str], start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    clean = list(dict.fromkeys(str(s).strip().upper() for s in symbols if str(s).strip()))
    if not clean:
        raise ValueError("종목 코드를 입력하세요.")
    data = yf.download(clean, start=start, end=end, interval=interval, auto_adjust=True,
                       progress=False, group_by="ticker", threads=True)
    if data.empty:
        raise ValueError("요청 기간에 시세가 없습니다.")
    frames = []
    for symbol in clean:
        if isinstance(data.columns, pd.MultiIndex):
            if symbol not in data.columns.get_level_values(0):
                continue
            rows = data[symbol].copy()
        else:
            rows = data.copy()
        rows.columns = [str(c).lower() for c in rows.columns]
        rows["timestamp"] = pd.to_datetime(rows.index).tz_localize(None)
        rows["symbol"] = symbol
        frames.append(rows.reset_index(drop=True))
    if not frames:
        raise ValueError("시세 데이터에서 종목 열을 만들지 못했습니다.")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.rename(columns={"adj close": "close"})
    frame["close"] = frame["close"].astype(float)
    frame["volume"] = frame["volume"].fillna(0).astype(float)
    return add_market_features(frame.sort_values(["timestamp", "symbol"]))

def default_history_range() -> tuple[str, str]:
    return (date.today() - timedelta(days=365 * 3)).isoformat(), date.today().isoformat()
