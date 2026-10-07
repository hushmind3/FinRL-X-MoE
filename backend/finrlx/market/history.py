from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

import pandas as pd
import yfinance as yf

from ..config import RUNTIME_DIR
from ..finrl_imports import finrl_data_manager
from .features import add_market_features

def fetch_history(symbols: Sequence[str], start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    clean = list(dict.fromkeys(str(s).strip().upper() for s in symbols if str(s).strip()))
    if not clean:
        raise ValueError("종목 코드를 입력하세요.")
    # FinRL-X owns the historical data contract and cache. Yahoo is a source
    # adapter for installations without an FMP key or a populated local cache.
    manager = finrl_data_manager(str(RUNTIME_DIR / "finrl_data"))
    ticker_table = pd.DataFrame({"tickers": clean, "sectors": [None] * len(clean),
                                 "dateFirstAdded": [None] * len(clean)})
    data = manager.get_price_data(ticker_table, start, end)
    if data.empty:
        data = _fetch_yahoo_into_finrl_cache(clean, start, end, interval, manager)
    if data.empty:
        raise ValueError("FinRL-X 데이터 관리자와 Yahoo adapter 모두 요청 기간의 시세를 반환하지 않았습니다.")

    frame = data.copy()
    if "tic" not in frame.columns:
        frame["tic"] = frame.get("gvkey", "")
    if "gvkey" not in frame.columns:
        frame["gvkey"] = frame["tic"]
    if "datadate" not in frame.columns:
        raise ValueError("FinRL-X price data is missing datadate.")
    frame["timestamp"] = pd.to_datetime(frame["datadate"], errors="coerce").dt.tz_localize(None)
    frame["symbol"] = frame["tic"].astype(str).str.upper()
    frame = frame[frame["symbol"].isin(clean)].copy()
    frame["close"] = pd.to_numeric(frame.get("adj_close", frame.get("prccd")), errors="coerce")
    frame["open"] = pd.to_numeric(frame.get("prcod", frame["close"]), errors="coerce")
    frame["high"] = pd.to_numeric(frame.get("prchd", frame["close"]), errors="coerce")
    frame["low"] = pd.to_numeric(frame.get("prcld", frame["close"]), errors="coerce")
    frame["volume"] = pd.to_numeric(
        frame.get("cshtrd", pd.Series(0, index=frame.index)), errors="coerce"
    ).fillna(0)
    frame = frame[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
    frame = frame.dropna(subset=["timestamp", "close"]).sort_values(["timestamp", "symbol"])
    if frame.empty:
        raise ValueError("FinRL-X price data contains no usable ticker rows.")
    return add_market_features(frame.sort_values(["timestamp", "symbol"]))


def _fetch_yahoo_into_finrl_cache(symbols: list[str], start: str, end: str,
                                  interval: str, manager) -> pd.DataFrame:
    data = yf.download(symbols, start=start, end=end, interval=interval, auto_adjust=True,
                       progress=False, group_by="ticker", threads=True)
    if data.empty:
        return pd.DataFrame()
    frames = []
    for symbol in symbols:
        if isinstance(data.columns, pd.MultiIndex):
            if symbol in data.columns.get_level_values(0):
                rows = data[symbol].copy()
            elif symbol in data.columns.get_level_values(1):
                rows = data.xs(symbol, level=1, axis=1).copy()
            else:
                continue
        else:
            rows = data.copy()
        rows.columns = [str(column).lower() for column in rows.columns]
        rows["datadate"] = pd.to_datetime(rows.index).tz_localize(None).strftime("%Y-%m-%d")
        rows["gvkey"] = symbol
        rows["tic"] = symbol
        rows["prccd"] = rows.get("close")
        rows["prcod"] = rows.get("open", rows["prccd"])
        rows["prchd"] = rows.get("high", rows["prccd"])
        rows["prcld"] = rows.get("low", rows["prccd"])
        rows["cshtrd"] = rows.get("volume", 0)
        rows["adj_close"] = rows["prccd"]
        frames.append(rows[["gvkey", "datadate", "tic", "prccd", "prcod", "prchd", "prcld", "cshtrd", "adj_close"]]
                      .reset_index(drop=True))
    if not frames:
        return pd.DataFrame()
    canonical = pd.concat(frames, ignore_index=True)
    # Persist source-adapted rows in FinRL-X's own normalized store so future
    # calls share its caching and canonical price schema.
    manager.current_source.data_store.save_price_data(canonical)
    return manager.get_price_data(pd.DataFrame({"tickers": symbols}), start, end)

def default_history_range() -> tuple[str, str]:
    return (date.today() - timedelta(days=365 * 3)).isoformat(), date.today().isoformat()
