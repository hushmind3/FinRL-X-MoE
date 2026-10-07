from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import websockets
import yfinance as yf

from ..config import FEED_POLL_SECONDS
from .features import add_market_features
from ..models.champion import champion

class MarketFeed:
    """One market connection fan-outs normalized quote/bar events to API clients and RL."""
    def __init__(self) -> None:
        self.symbols: list[str] = []
        self.provider = "stopped"
        self.timeframe = "1Min"
        self.running = False
        self.last_error: str | None = None
        self.last_event: dict[str, Any] | None = None
        self.history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=2048))
        self.latest_enriched: dict[str, dict[str, Any]] = {}
        self.training_events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=10_000)
        self.inference_events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=10_000)
        self.clients: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None
        self._lock = threading.RLock()

    async def start(self, symbols: list[str], provider: str, timeframe: str) -> dict[str, Any]:
        clean = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
        if not clean:
            raise ValueError("구독할 종목 코드를 입력하세요.")
        await self.stop()
        for events in (self.training_events, self.inference_events):
            while True:
                try:
                    events.get_nowait()
                except queue.Empty:
                    break
        with self._lock:
            self.history = defaultdict(lambda: deque(maxlen=2048))
            self.latest_enriched = {}
            self.last_event = None
        self.symbols, self.provider, self.timeframe = clean, provider, timeframe
        self.last_error = None
        self.running = True
        if provider == "alpaca":
            if not (os.getenv("APCA_API_KEY") or os.getenv("ALPACA_API_KEY")) or not (
                os.getenv("APCA_API_SECRET") or os.getenv("ALPACA_API_SECRET")
            ):
                self.running = False
                raise ValueError("Alpaca 실시간 시세를 쓰려면 .env에 FinRL-X 규격 APCA_API_KEY와 APCA_API_SECRET을 설정하세요.")
        if provider == "alpaca":
            self._task = asyncio.create_task(self._alpaca_loop(), name="finrlx-alpaca-stream")
        else:
            self._task = asyncio.create_task(self._yahoo_loop(), name="finrlx-yahoo-poll")
        await self._broadcast({"type": "feed_status", "status": self.status()})
        return self.status()

    async def stop(self) -> dict[str, Any]:
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        self.provider = "stopped"
        await self._broadcast({"type": "feed_status", "status": self.status()})
        return self.status()

    def status(self) -> dict[str, Any]:
        return {"running": self.running, "provider": self.provider, "timeframe": self.timeframe,
                "symbols": self.symbols, "last_error": self.last_error,
                "last_update": self.last_event.get("timestamp") if self.last_event else None,
                "data_quality": "live_stream" if self.provider == "alpaca" else
                                "polled_delayed" if self.provider == "yahoo_poll" else "none"}

    def snapshot(self, limit: int = 300) -> dict[str, Any]:
        with self._lock:
            return {"status": self.status(), "quotes": [list(v)[-1] for v in self.history.values() if v],
                    "bars": {symbol: list(rows)[-limit:] for symbol, rows in self.history.items()}}

    async def subscribe(self) -> asyncio.Queue:
        channel: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.clients.add(channel)
        return channel

    def unsubscribe(self, channel: asyncio.Queue) -> None:
        self.clients.discard(channel)

    async def _publish(self, symbol: str, timestamp: Any, open_: float, high: float, low: float,
                       close: float, volume: float, *, kind: str = "bar") -> None:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        event = {"type": kind, "symbol": symbol.upper(), "timestamp": ts.isoformat(),
                 "open": float(open_), "high": float(high), "low": float(low),
                 "close": float(close), "volume": float(volume), "provider": self.provider}
        with self._lock:
            rows = self.history[symbol.upper()]
            if rows and rows[-1]["timestamp"] == event["timestamp"]:
                # Replace the forming minute so chart values stay current; learning only sees closed bars.
                previous = rows[-1]
                rows[-1] = event
                is_new_bar = False
                is_updated = any(previous.get(key) != event.get(key)
                                 for key in ("open", "high", "low", "close", "volume"))
            else:
                rows.append(event)
                is_new_bar = True
                is_updated = True
            self.last_event = event
            rows_copy = list(rows)
            cached_enriched = self.latest_enriched.get(symbol.upper())
        if not is_updated and cached_enriched is not None:
            enriched = cached_enriched.copy()
        else:
            enriched = event.copy()
            try:
                features = add_market_features(pd.DataFrame(rows_copy))
                if not features.empty:
                    for key in ("momentum_1", "momentum_5", "momentum_20", "volatility_20", "trend_20",
                                "trend_60", "range_pct", "volume_z", "drawdown_60"):
                        enriched[key] = float(features.iloc[-1][key])
            except Exception as exc:
                enriched["feature_error"] = f"{type(exc).__name__}: {exc}"
            try:
                fused = champion.extract_live(enriched)
                if fused is not None:
                    enriched["expert_latent"], enriched["expert_available"] = fused
            except Exception as exc:
                enriched["expert_error"] = f"{type(exc).__name__}: {exc}"
            self.latest_enriched[symbol.upper()] = enriched.copy()
        await self._broadcast({"type": "market_event", "data": enriched, "status": self.status()})
        # Alpaca emits a bar update as it forms; only enqueue it after a later
        # timestamp proves the previous minute is closed. Yahoo polling also
        # benefits from one transition per timestamp rather than repeated polls.
        if is_new_bar:
            for events, label in ((self.training_events, "학습"), (self.inference_events, "추론")):
                try:
                    events.put_nowait(enriched)
                except queue.Full:
                    self.last_error = f"{label} 입력 대기열이 찼습니다. 오래된 bar를 건너뜁니다."

    async def _broadcast(self, message: dict[str, Any]) -> None:
        stale = []
        for client in tuple(self.clients):
            try:
                client.put_nowait(message)
            except asyncio.QueueFull:
                stale.append(client)
        for client in stale:
            self.clients.discard(client)

    async def broadcast_external(self, message: dict[str, Any]) -> None:
        await self._broadcast(message)

    async def _yahoo_loop(self) -> None:
        while self.running:
            try:
                results = await asyncio.gather(*(asyncio.to_thread(self._yahoo_latest, s) for s in self.symbols),
                                               return_exceptions=True)
                for symbol, row in zip(self.symbols, results):
                    if isinstance(row, Exception):
                        self.last_error = f"{symbol}: {type(row).__name__}: {row}"
                    elif row is not None:
                        await self._publish(symbol, **row)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(FEED_POLL_SECONDS)

    @staticmethod
    def _yahoo_latest(symbol: str) -> dict[str, Any] | None:
        frame = yf.Ticker(symbol).history(period="1d", interval="1m", prepost=True, auto_adjust=True)
        if frame.empty:
            return None
        row = frame.iloc[-1]
        idx = frame.index[-1]
        return {"timestamp": idx.to_pydatetime(), "open_": row["Open"], "high": row["High"],
                "low": row["Low"], "close": row["Close"], "volume": row["Volume"]}

    async def _alpaca_loop(self) -> None:
        feed = os.getenv("ALPACA_DATA_FEED", "iex").lower()
        url = f"wss://stream.data.alpaca.markets/v2/{feed}"
        key = os.getenv("APCA_API_KEY") or os.environ["ALPACA_API_KEY"]
        secret = os.getenv("APCA_API_SECRET") or os.environ["ALPACA_API_SECRET"]
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_size=4_000_000) as ws:
                    await ws.send(json.dumps({"action": "auth", "key": key, "secret": secret}))
                    auth = json.loads(await ws.recv())
                    if not any(m.get("T") == "success" and m.get("msg") == "authenticated" for m in auth):
                        raise RuntimeError(f"Alpaca 인증 실패: {auth}")
                    await ws.send(json.dumps({"action": "subscribe", "bars": self.symbols}))
                    async for raw in ws:
                        for message in json.loads(raw):
                            if message.get("T") != "b":
                                continue
                            await self._publish(message["S"], message["t"], message["o"], message["h"],
                                                message["l"], message["c"], message["v"])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                await self._broadcast({"type": "feed_status", "status": self.status()})
                await asyncio.sleep(3)

feed = MarketFeed()
