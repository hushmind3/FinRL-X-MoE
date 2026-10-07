from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any

from ..market.feed import feed
from ..models.champion import champion
from ..rl.trainer import TorchRLTrainer
from ..rl.inference import LiveInferenceRunner
from .paper import PaperAccount

class TradingRuntime:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.paper = PaperAccount()
        self.trainer = TorchRLTrainer(feed.training_events, self._publish_from_worker)
        self.inference = LiveInferenceRunner(feed.inference_events, self.trainer.act,
                                             self._publish_from_worker)
        self.latest_target: dict[str, Any] | None = None
        self.last_decision: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.strategy = None
        self._lock = threading.RLock()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def state(self) -> dict:
        model_info = champion.inspect()
        return {"mode": "research_paper", "live_orders_enabled": False,
                "live_gate": {"status": "closed", "reason": "broker execution adapter and user risk approval are not configured"},
                "feed": feed.status(), "champion": model_info, "training": self.trainer.info(),
                "inference": {"running": self.inference.running, "error": self.inference.error},
                "paper": self.paper.snapshot(), "target_weights": self.latest_target,
                "last_decision": self.last_decision, "error": self.last_error}

    async def start_feed(self, symbols: list[str], provider: str, timeframe: str) -> dict:
        self.set_loop(asyncio.get_running_loop())
        self.inference.stop()
        if self.trainer.running:
            training = self.trainer.stop()
            if training["running"]:
                raise ValueError("기존 PPO collector를 정리하는 중입니다. 잠시 뒤 시세를 다시 시작하세요.")
        self.paper.enabled = False
        return await feed.start(symbols, provider, timeframe)

    async def stop_feed(self) -> dict:
        self.inference.stop()
        self.trainer.stop()
        self.paper.enabled = False
        return await feed.stop()

    def start_training(self, *, algorithm: str = "PPO", frames_per_batch: int = 32,
                       learning_rate: float = 3e-4) -> dict:
        if not feed.running:
            raise ValueError("먼저 실시간 시세를 시작하세요.")
        if not self.trainer.running:
            # Frozen expert predictions are included in observations only when
            # their native adapter has produced a valid, symbol-matched output.
            self.trainer.start(feed.symbols, feed.training_events, algorithm=algorithm,
                               frames_per_batch=frames_per_batch, learning_rate=learning_rate)
        return self.trainer.info()

    def stop_training(self) -> dict:
        return self.trainer.stop()

    def start_paper(self) -> dict:
        if not feed.running:
            raise ValueError("시세를 먼저 시작하세요.")
        if not self.trainer.policy_ready or self.trainer.updates < 1:
            raise ValueError("TorchRL learner의 첫 업데이트가 끝난 뒤 paper 실행을 켜세요.")
        while True:
            try:
                feed.inference_events.get_nowait()
            except queue.Empty:
                break
        self.paper.enabled = True
        self.inference.start(feed.symbols)
        return self.paper.snapshot()

    def stop_paper(self) -> dict:
        self.inference.stop()
        self.paper.enabled = False
        self.paper._save()
        return self.paper.snapshot()

    def _publish_from_worker(self, event: dict) -> None:
        data = event.get("data", {})
        if event.get("type") == "policy_decision":
            weights = data.get("weights", {})
            target = {"as_of": data.get("execution_time", data.get("timestamp")),
                      "signal_time": data.get("signal_time"),
                      "execution_time": data.get("execution_time", data.get("timestamp")),
                      "weights": weights,
                      "cash_weight": data.get("cash_weight", 1.0), "source": "policy",
                      "observation_source": "causal_market_features+available_champion_latent",
                      "expert_checkpoint_status": champion.inspect().get("status")}
            with self._lock:
                self.latest_target = target
                self.last_decision = data
            # Publish through FinRL-Trading's BaseStrategy contract.
            if self.paper.enabled:
                for symbol, price in data.get("prices", {}).items():
                    self.paper.mark(symbol, float(price))
                try:
                    self.paper.rebalance(weights, str(target["execution_time"]))
                except Exception as exc:
                    self.paper.error = f"paper rebalance rejected: {exc}"
            try:
                if self.strategy is None:
                    from ..strategy import TargetWeightStrategy
                    self.strategy = TargetWeightStrategy()
                self.strategy.set_target_weights(weights, {"signal_time": data.get("signal_time"),
                                                           "execution_time": target["execution_time"],
                                                           "source": "torchrl_policy"})
            except Exception as exc:
                self.last_error = f"FinRL-X weight contract 준비 실패: {type(exc).__name__}: {exc}"
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self._enqueue_event, event)

    @staticmethod
    def _enqueue_event(event: dict) -> None:
        # Rebound by API startup to the market WebSocket fan-out.
        from ..market.feed import feed
        task = asyncio.create_task(feed.broadcast_external(event))
        task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)

runtime = TradingRuntime()
