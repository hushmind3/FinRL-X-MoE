from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from .backtest import run_finrl_backtest
from .config import INITIAL_CAPITAL
from .market.feed import feed
from .market.history import fetch_history
from .models.champion import champion
from .runtime.service import runtime
from .schemas import BacktestRequest, FeedStartRequest, TrainRequest

API_PREFIX = "/api/v1"

@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.set_loop(asyncio.get_running_loop())
    yield
    runtime.stop_training()
    runtime.stop_paper()
    await runtime.stop_feed()

app = FastAPI(title="FinRL-X Trading Lab API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5178", "http://localhost:5178"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])

@app.get(f"{API_PREFIX}/health")
def health() -> dict:
    return {"ok": True, "service": "finrlx-backend", "api_version": "v1",
            "project_id": "finrlx-trading-lab", "runtime_contract": 5}

@app.get(f"{API_PREFIX}/state")
def state() -> dict:
    return runtime.state()

@app.get(f"{API_PREFIX}/models/champion")
def champion_status() -> dict:
    return champion.inspect()

@app.post(f"{API_PREFIX}/models/champion/inspect")
def inspect_champion() -> dict:
    result = champion.inspect()
    if not result.get("available"):
        raise HTTPException(400, result.get("error", "체크포인트 형식이 지원되지 않습니다."))
    return result

@app.post(f"{API_PREFIX}/feed/start")
async def start_feed(request: FeedStartRequest) -> dict:
    try:
        return await runtime.start_feed(request.symbols, request.provider, request.timeframe)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

@app.post(f"{API_PREFIX}/feed/stop")
async def stop_feed() -> dict:
    return await runtime.stop_feed()

@app.get(f"{API_PREFIX}/feed/snapshot")
def feed_snapshot() -> dict:
    return feed.snapshot()

@app.post(f"{API_PREFIX}/training/start")
def start_training(request: TrainRequest) -> dict:
    try:
        return runtime.start_training(algorithm=request.algorithm,
                                      frames_per_batch=request.steps_per_iteration,
                                      learning_rate=request.learning_rate)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc

@app.post(f"{API_PREFIX}/training/stop")
def stop_training() -> dict:
    return runtime.stop_training()

@app.post(f"{API_PREFIX}/paper/start")
def start_paper() -> dict:
    try:
        return runtime.start_paper()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc

@app.post(f"{API_PREFIX}/paper/stop")
def stop_paper() -> dict:
    return runtime.stop_paper()

@app.get(f"{API_PREFIX}/paper/account")
def paper_account() -> dict:
    return runtime.paper.snapshot()

@app.post(f"{API_PREFIX}/backtest")
async def backtest(request: BacktestRequest) -> dict:
    try:
        features = await run_in_threadpool(fetch_history, request.symbols, request.start, request.end)
        result = await run_in_threadpool(run_finrl_backtest, features, request.initial_capital)
        result["symbol_count"] = len(set(request.symbols))
        return result
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"백테스트 실패: {type(exc).__name__}: {exc}") from exc

@app.post(f"{API_PREFIX}/trading/live/start")
def live_trading_gate() -> dict:
    state = runtime.state()["live_gate"]
    raise HTTPException(409, {"message": "실계좌 주문 게이트가 닫혀 있습니다.", **state})

@app.websocket("/ws/market")
async def market_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    channel = await feed.subscribe()
    await websocket.send_json({"type": "snapshot", "data": feed.snapshot(), "runtime": runtime.state()})
    try:
        while True:
            event = await channel.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        feed.unsubscribe(channel)
