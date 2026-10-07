from __future__ import annotations

import os
import threading
from typing import Callable

import torch
from tensordict import TensorDict
from torchrl.collectors import Collector
from torchrl.modules import ProbabilisticActor, ValueOperator

from ..config import RUNTIME_DIR
from .learners import available_learners, create_learner
from .live_env import LivePortfolioEnv
from .policy import (
    ARCHITECTURE_VERSION,
    PolicyContract,
    SharedPortfolioValue,
    build_torchrl_policy,
    deterministic_action,
)


class TorchRLTrainer:
    """TorchRL collector around a replaceable learner and a shared policy actor."""

    def __init__(self, event_queue, publish: Callable[[dict], None]) -> None:
        self.event_queue, self.publish = event_queue, publish
        self.running = False
        self.status = "idle"
        self.error: str | None = None
        self.steps = 0
        self.updates = 0
        self.mean_reward: float | None = None
        self.last_loss: float | None = None
        self.algorithm = "PPO"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = RUNTIME_DIR / "policy_torchrl.pt"
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._policy_lock = threading.RLock()
        self._policy: ProbabilisticActor | None = None
        self._critic: ValueOperator | None = None
        self._active_symbols: list[str] = []
        self.frames_per_batch = 32
        self.learning_rate = 3e-4

    def info(self) -> dict:
        return {"status": self.status, "running": self.running, "algorithm": self.algorithm,
                "available_algorithms": list(available_learners()), "steps": self.steps,
                "updates": self.updates, "mean_reward": self.mean_reward,
                "last_loss": self.last_loss, "device": str(self.device), "error": self.error,
                "checkpoint": str(self.checkpoint), "symbols": self._active_symbols,
                "rollout": "TorchRL Collector; learner consumes each rollout once"}

    @property
    def policy_ready(self) -> bool:
        with self._policy_lock:
            return self._policy is not None

    def start(self, symbols: list[str], event_queue=None, *, algorithm: str = "PPO",
              frames_per_batch: int = 32, learning_rate: float = 3e-4) -> dict:
        with self._lock:
            if self.running:
                return self.info()
            if not symbols:
                raise ValueError("학습할 종목 시세를 먼저 시작하세요.")
            # Resolve the learner before launching the worker, so unsupported
            # algorithms fail at the API boundary instead of in a background thread.
            if algorithm.strip().upper() not in available_learners():
                raise ValueError(f"지원하지 않는 learner입니다. 사용 가능: {', '.join(available_learners())}")
            self.algorithm = algorithm.strip().upper()
            self._active_symbols = list(dict.fromkeys(str(s).upper() for s in symbols))
            self.event_queue = event_queue or self.event_queue
            self.frames_per_batch = frames_per_batch
            self.learning_rate = learning_rate
            self._stop.clear()
            self.running, self.status, self.error = True, "starting", None
            self._thread = threading.Thread(target=self._run, name="finrlx-torchrl-learner", daemon=True)
            self._thread.start()
            return self.info()

    def stop(self) -> dict:
        self._stop.set()
        self.status = "stopping"
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if not thread or not thread.is_alive():
            self.running = False
        return self.info()

    def act(self, observation: TensorDict) -> list[float] | None:
        with self._policy_lock:
            policy = self._policy
            if policy is None:
                return None
            with torch.no_grad():
                action = deterministic_action(policy, observation.to(self.device))
                return action.detach().cpu().reshape(-1).tolist()

    def _run(self) -> None:
        collector = None
        try:
            env = LivePortfolioEnv(self._active_symbols, self.event_queue, stop_event=self._stop)
            policy = build_torchrl_policy(len(self._active_symbols), self.device)
            critic = ValueOperator(
                SharedPortfolioValue().to(self.device),
                in_keys=["asset_features", "expert_latent", "expert_mask", "asset_mask", "account_state"],
            ).to(self.device)
            learner = create_learner(
                self.algorithm, actor=policy, critic=critic, learning_rate=self.learning_rate,
                minibatch_size=max(1, min(64, self.frames_per_batch)),
            )
            self._load_checkpoint(policy, critic, learner)
            with self._policy_lock:
                self._policy, self._critic = policy, critic
            policy.eval()
            critic.eval()
            collector = Collector(env, policy, frames_per_batch=self.frames_per_batch, total_frames=-1,
                                  device=self.device, storing_device="cpu", max_frames_per_traj=4096)
            self.status = "running"
            self.publish({"type": "training_status", "data": self.info()})
            for rollout in collector:
                if self._stop.is_set():
                    break
                rollout = rollout.to(self.device)
                with self._policy_lock:
                    metrics = learner.update(rollout)
                self.last_loss = metrics.get("loss")
                collector.update_policy_weights_()
                reward = float(rollout["next", "reward"].mean().item())
                self.mean_reward = reward
                self.steps += int(rollout.numel())
                self.updates += 1
                if self.updates % 10 == 0:
                    self._save(policy, critic, learner)
                self.publish({"type": "training_status", "data": self.info()})
            self.status = "stopped"
        except Exception as exc:
            if self._stop.is_set():
                self.status = "stopped"
            else:
                self.error = f"{type(exc).__name__}: {exc}"
                self.status = "error"
        finally:
            self.running = False
            if self.updates > 0 and self._policy is not None and self._critic is not None:
                try:
                    self._save(self._policy, self._critic, learner if "learner" in locals() else None)
                except Exception as exc:
                    self.error = self.error or f"정책 저장 실패: {type(exc).__name__}: {exc}"
            if collector is not None:
                try:
                    collector.shutdown(raise_on_error=False)
                except Exception:
                    pass
            self.publish({"type": "training_status", "data": self.info()})

    def _load_checkpoint(self, policy: ProbabilisticActor, critic: ValueOperator, learner=None) -> None:
        if not self.checkpoint.is_file():
            return
        try:
            state = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
            if (state.get("architecture_version") != ARCHITECTURE_VERSION
                    or state.get("policy_contract") != PolicyContract().as_dict()):
                return
            if not (_state_compatible(policy.module, state.get("policy", {}))
                    and _state_compatible(critic.module, state.get("critic", {}))):
                return
            policy.module.load_state_dict(state["policy"], strict=True)
            critic.module.load_state_dict(state["critic"], strict=True)
            self.updates = int(state.get("updates", 0))
            self.steps = int(state.get("steps", 0))
            optimizer = getattr(learner, "optimizer", None)
            saved_optimizer = state.get("optimizer")
            if (optimizer is not None and saved_optimizer is not None
                    and state.get("algorithm", "").upper() == self.algorithm):
                optimizer.load_state_dict(saved_optimizer)
        except (OSError, RuntimeError, ValueError, TypeError):
            # An incompatible or damaged checkpoint is not partially loaded.
            return

    def _save(self, policy, critic, learner=None) -> None:
        # Never publish a randomly initialized policy when collection or the
        # first learner update fails before producing a trained checkpoint.
        if self.updates < 1:
            return
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint.with_suffix(".tmp")
        optimizer = getattr(learner, "optimizer", None)
        torch.save({"algorithm": self.algorithm, "architecture_version": ARCHITECTURE_VERSION,
                    "policy_contract": PolicyContract().as_dict(),
                    "policy": policy.module.state_dict(), "critic": critic.module.state_dict(),
                    "optimizer": optimizer.state_dict() if optimizer is not None else None,
                    "updates": self.updates, "steps": self.steps}, tmp)
        os.replace(tmp, self.checkpoint)

def _state_compatible(module, saved: dict[str, torch.Tensor]) -> bool:
    current = module.state_dict()
    return current.keys() == saved.keys() and all(current[k].shape == saved[k].shape for k in current)
