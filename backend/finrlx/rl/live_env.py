from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Binary, Composite, Unbounded
from torchrl.envs import EnvBase

from ..config import TRANSACTION_COST
from ..market.features import FEATURE_KEYS


class LivePortfolioEnv(EnvBase):
    """TorchRL environment over synchronized closed bars and portfolio state."""

    def __init__(self, symbols: list[str], event_queue: queue.Queue, *,
                 on_transition: Callable[[dict], None] | None = None,
                 stop_event: threading.Event | None = None) -> None:
        self.symbols = list(dict.fromkeys(str(s).upper() for s in symbols))
        if not self.symbols:
            raise ValueError("LivePortfolioEnv에는 종목 시세가 필요합니다.")
        self.event_queue = event_queue
        self.on_transition = on_transition
        self.stop_event = stop_event or threading.Event()
        self.latest: dict[str, dict] = {}
        # The final component is the cash target. It is an action dimension,
        # never a ticker or a price series.
        self.last_weights = np.r_[np.zeros(len(self.symbols), dtype=np.float32), 1.0]
        self.last_prices: dict[str, float] = {}
        self.equity = 1.0
        self.peak_equity = 1.0
        self.step_count = 0
        self.last_turnover = 0.0
        super().__init__(device=torch.device("cpu"), batch_size=torch.Size([]))
        count = len(self.symbols)
        self.observation_spec = Composite(
            asset_features=Unbounded(shape=(count, len(FEATURE_KEYS) + 1), dtype=torch.float32),
            expert_latent=Unbounded(shape=(count, 64), dtype=torch.float32),
            expert_mask=Binary(shape=(count,), dtype=torch.bool),
            asset_mask=Binary(shape=(count,), dtype=torch.bool),
            account_state=Unbounded(shape=(4,), dtype=torch.float32),
            device=self.device,
        )
        self.action_spec = Unbounded(shape=(count + 1,), dtype=torch.float32, device=self.device)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32, device=self.device)
        self.full_done_spec = Composite(
            done=Binary(shape=(1,), dtype=torch.bool, device=self.device),
            terminated=Binary(shape=(1,), dtype=torch.bool, device=self.device),
            truncated=Binary(shape=(1,), dtype=torch.bool, device=self.device),
            device=self.device,
        )

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        events = self._next_batch()
        for symbol, event in events.items():
            self.latest[symbol] = event
            self.last_prices[symbol] = float(event["close"])
        self.last_weights = np.r_[np.zeros(len(self.symbols), dtype=np.float32), 1.0]
        self.equity = self.peak_equity = 1.0
        self.step_count = 0
        self.last_turnover = 0.0
        return TensorDict({**self._observation(), "done": torch.zeros(1, dtype=torch.bool),
                           "terminated": torch.zeros(1, dtype=torch.bool),
                           "truncated": torch.zeros(1, dtype=torch.bool)}, batch_size=[])

    def _step(self, tensordict: TensorDict) -> TensorDict:
        target = self._to_weights(tensordict["action"].detach().cpu().numpy().reshape(-1))
        policy_observation = self._observation()
        signal_time = max(row["timestamp"] for row in self.latest.values())
        events = self._next_batch()
        execution_time = max(event["timestamp"] for event in events.values())

        old = self.last_weights.astype(np.float64)
        returns = np.zeros(len(self.symbols), dtype=np.float64)
        for idx, symbol in enumerate(self.symbols):
            event = events[symbol]
            new_price = float(event["close"])
            old_price = self.last_prices.get(symbol, new_price)
            returns[idx] = new_price / old_price - 1.0 if old_price > 0 else 0.0
            self.latest[symbol] = event
            self.last_prices[symbol] = new_price

        # Existing holdings earn the signal-to-execution interval. The new
        # allocation is executed at execution_time, matching the historical
        # backtest's next-available-close convention.
        gross_return = float(np.dot(old[:-1], returns))
        gross_factor = 1.0 + gross_return
        drifted = np.r_[old[:-1] * (1.0 + returns), old[-1]] / max(gross_factor, 1e-12)
        turnover = float(np.abs(target[:-1].astype(np.float64) - drifted[:-1]).sum())
        net_factor = gross_factor * (1.0 - TRANSACTION_COST * turnover)
        net_return = net_factor - 1.0
        self.equity *= max(1e-8, net_factor)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.last_weights = target
        self.last_turnover = turnover
        self.step_count += 1
        reward = float(np.clip(np.log(max(1e-8, 1 + net_return)) * 100, -10, 10))

        snapshot = {
            "as_of": execution_time,
            "symbols": self.symbols,
            "signal_time": signal_time,
            "execution_time": execution_time,
            "timestamp": execution_time,
            "weights": self.weights(),
            "cash_weight": float(target[-1]),
            "source": "torchrl_policy",
            "observation_source": "causal_market_features+available_champion_latent",
            "equity_return": self.equity - 1,
            "portfolio_return": net_return,
            "reward": reward,
            "turnover": turnover,
            "prices": self.last_prices.copy(),
            "policy_observation": policy_observation,
        }
        if self.on_transition:
            self.on_transition(snapshot)
        return TensorDict({**self._observation(), "reward": torch.tensor([reward], dtype=torch.float32),
                           "done": torch.zeros(1, dtype=torch.bool),
                           "terminated": torch.zeros(1, dtype=torch.bool),
                           "truncated": torch.zeros(1, dtype=torch.bool)}, batch_size=[])

    def _next_batch(self) -> dict[str, dict]:
        deadline = time.monotonic() + 180
        batch: dict[str, dict] = {}
        while True:
            if self.stop_event.is_set():
                raise InterruptedError("TorchRL collector stopped by request.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("새 market bar를 180초 동안 받지 못했습니다.")
            try:
                event = self.event_queue.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            symbol = event.get("symbol")
            if symbol in self.symbols and float(event.get("close", 0)) > 0:
                batch[symbol] = event
                if len(batch) == len(self.symbols):
                    return batch

    def _observation(self) -> TensorDict:
        market_rows: list[list[float]] = []
        latent_rows: list[list[float]] = []
        expert_mask: list[bool] = []
        for idx, symbol in enumerate(self.symbols):
            row = self.latest.get(symbol, {})
            market_rows.append([*(float(row.get(key, 0) or 0) for key in FEATURE_KEYS),
                                float(self.last_weights[idx])])
            latent = row.get("expert_latent")
            available = bool(row.get("expert_available", False)) and latent is not None
            latent_values = np.asarray(latent if available else np.zeros(64), dtype=np.float32).reshape(-1)
            if latent_values.size != 64 or not np.isfinite(latent_values).all():
                available = False
                latent_values = np.zeros(64, dtype=np.float32)
            latent_rows.append(latent_values.tolist())
            expert_mask.append(available)
        cash_weight = max(0.0, 1.0 - float(self.last_weights[:-1].sum()))
        account = [float(self.equity - 1), float(self.equity / self.peak_equity - 1),
                   cash_weight, float(self.last_turnover)]
        return TensorDict({
            "asset_features": torch.as_tensor(np.nan_to_num(market_rows), dtype=torch.float32),
            "expert_latent": torch.as_tensor(np.nan_to_num(latent_rows), dtype=torch.float32),
            "expert_mask": torch.as_tensor(expert_mask, dtype=torch.bool),
            "asset_mask": torch.ones(len(self.symbols), dtype=torch.bool),
            "account_state": torch.as_tensor(np.nan_to_num(np.clip(account, -5, 5)), dtype=torch.float32),
        }, batch_size=[])

    def _to_weights(self, action: np.ndarray) -> np.ndarray:
        allocation = np.asarray(action, dtype=np.float64).reshape(-1)
        if allocation.size != len(self.symbols) + 1:
            raise ValueError("정책 action 차원이 종목 수와 현금 action을 반영하지 않습니다.")
        if not np.isfinite(allocation).all() or np.any(allocation < 0):
            raise ValueError("정책 목표 비중은 유한한 음수 아닌 값이어야 합니다.")
        if not np.isclose(float(allocation.sum()), 1.0, rtol=1e-5, atol=1e-5):
            raise ValueError("정책 목표 종목 비중과 cash 비중 합계가 1이 아닙니다.")
        return allocation.astype(np.float32)

    def _set_seed(self, seed: int | None) -> int:
        actual_seed = int(seed if seed is not None else torch.seed())
        torch.manual_seed(actual_seed)
        np.random.seed(actual_seed % (2**32))
        return actual_seed

    def weights(self) -> dict[str, float]:
        return {symbol: float(self.last_weights[i]) for i, symbol in enumerate(self.symbols)}
