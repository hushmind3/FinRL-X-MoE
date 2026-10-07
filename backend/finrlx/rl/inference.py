from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import torch
from tensordict import TensorDict

from .live_env import LivePortfolioEnv


class LiveInferenceRunner:
    """Deterministic live runner sharing the training EnvBase and actor helper."""

    def __init__(self, event_queue: queue.Queue, action: Callable[[TensorDict], list[float] | None],
                 publish: Callable[[dict], None]) -> None:
        self.event_queue = event_queue
        self.action = action
        self.publish = publish
        self.running = False
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, symbols: list[str]) -> None:
        if self.running:
            return
        self._stop.clear()
        self.error = None
        self.running = True
        self._thread = threading.Thread(
            target=self._run, args=(list(symbols),), name="finrlx-live-inference", daemon=True
        )
        self._thread.start()
        self.publish({"type": "inference_status", "data": {"running": True, "error": None}})

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if not thread or not thread.is_alive():
            self.running = False

    def _run(self, symbols: list[str]) -> None:
        env = LivePortfolioEnv(symbols, self.event_queue, on_transition=self._on_transition,
                               stop_event=self._stop)
        try:
            observation = env.reset()
            keys = ("asset_features", "expert_latent", "expert_mask", "asset_mask", "account_state")
            while not self._stop.is_set():
                values = self.action(observation.select(*keys))
                if values is None:
                    # Keep the current state until the training actor is ready.
                    time.sleep(0.1)
                    continue
                action = torch.as_tensor(values, dtype=torch.float32, device=env.device)
                transition = env.step(TensorDict({"action": action}, batch_size=[]))
                observation = transition.get("next").select(*keys)
        except InterruptedError:
            pass
        except Exception as exc:
            if not self._stop.is_set():
                self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.running = False
            self.publish({"type": "inference_status", "data": {"running": False, "error": self.error}})

    def _on_transition(self, snapshot: dict) -> None:
        snapshot.pop("policy_observation", None)
        self.publish({"type": "policy_decision", "data": snapshot})
