from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE


class TorchRLLearner(Protocol):
    """Algorithm boundary: learners consume the same actor, critic and rollout."""

    name: str

    def update(self, rollout: TensorDictBase) -> dict[str, float]: ...


class PPOLearner:
    """On-policy PPO baseline; each rollout is consumed once across epochs."""

    name = "PPO"

    def __init__(
        self,
        actor: ProbabilisticActor,
        critic: ValueOperator,
        *,
        learning_rate: float,
        minibatch_size: int,
        epochs: int = 4,
    ) -> None:
        self.loss = ClipPPOLoss(
            actor_network=actor,
            critic_network=critic,
            clip_epsilon=0.2,
            entropy_bonus=True,
            entropy_coeff=0.01,
            critic_coeff=1.0,
            loss_critic_type="smooth_l1",
        )
        self.advantage = GAE(gamma=0.99, lmbda=0.95, value_network=critic, average_gae=True)
        self.optimizer = torch.optim.Adam(self.loss.parameters(), lr=learning_rate)
        self.minibatch_size = minibatch_size
        self.epochs = epochs

    def update(self, rollout: TensorDictBase) -> dict[str, float]:
        rollout = rollout.reshape(-1)
        self.advantage(rollout)
        count = rollout.numel()
        values: list[float] = []
        for _ in range(self.epochs):
            order = torch.randperm(count, device=rollout.device)
            for start in range(0, count, self.minibatch_size):
                indices = order[start:start + self.minibatch_size]
                sample = rollout[indices]
                losses = self.loss(sample)
                total = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                nn.utils.clip_grad_norm_(self.loss.parameters(), 1.0)
                self.optimizer.step()
                values.append(float(total.detach().cpu()))
        return {"loss": sum(values) / max(1, len(values))}


_LEARNERS: dict[str, Callable[..., TorchRLLearner]] = {"PPO": PPOLearner}


def register_learner(name: str, factory: Callable[..., TorchRLLearner]) -> None:
    key = name.strip().upper()
    if not key:
        raise ValueError("learner 이름이 비어 있습니다.")
    _LEARNERS[key] = factory


def create_learner(name: str, **kwargs) -> TorchRLLearner:
    key = name.strip().upper()
    try:
        factory = _LEARNERS[key]
    except KeyError as exc:
        raise ValueError(f"등록되지 않은 TorchRL learner입니다: {name}") from exc
    return factory(**kwargs)


def available_learners() -> tuple[str, ...]:
    return tuple(sorted(_LEARNERS))
