from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import Unbounded
from torchrl.modules import ProbabilisticActor

from ..market.features import FEATURE_KEYS
from ..models.contracts import EXPERT_LATENT_SCHEMA, EXPERT_LATENT_SIZE


# The actor's parameters do not depend on the number or identity of assets.
ARCHITECTURE_VERSION = "shared_asset_pool_v2"
ASSET_FEATURE_SCHEMA = "ohlcv_causal_features_v2"
ASSET_FEATURE_SIZE = len(FEATURE_KEYS) + 1  # market features and current exposure
ACCOUNT_FEATURE_SCHEMA = "portfolio_account_v1"
ACCOUNT_FEATURE_SIZE = 4


@dataclass(frozen=True)
class PolicyContract:
    architecture_version: str = ARCHITECTURE_VERSION
    asset_feature_schema: str = ASSET_FEATURE_SCHEMA
    asset_feature_size: int = ASSET_FEATURE_SIZE
    expert_latent_schema: str = EXPERT_LATENT_SCHEMA
    expert_latent_size: int = EXPERT_LATENT_SIZE
    account_feature_schema: str = ACCOUNT_FEATURE_SCHEMA
    account_feature_size: int = ACCOUNT_FEATURE_SIZE

    def as_dict(self) -> dict[str, str | int]:
        return self.__dict__.copy()


class SharedAssetAllocationActor(nn.Module):
    """Shared per-asset encoder, O(N) cross-asset pooling, and shared heads."""

    def __init__(self, hidden_size: int = 128, attention_heads: int = 4) -> None:
        super().__init__()
        self.contract = PolicyContract()
        input_size = ASSET_FEATURE_SIZE + EXPERT_LATENT_SIZE + 1
        self.asset_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        # One learned query attends over any number of valid assets. This avoids
        # the quadratic memory and compute cost of self-attention over all assets.
        self.asset_pool_query = nn.Parameter(torch.empty(1, 1, hidden_size))
        nn.init.normal_(self.asset_pool_query, std=hidden_size ** -0.5)
        self.cross_asset_pool = nn.MultiheadAttention(hidden_size, attention_heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(hidden_size)
        self.account_encoder = nn.Sequential(nn.Linear(ACCOUNT_FEATURE_SIZE, hidden_size), nn.Tanh())
        self.allocation_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1)
        )
        self.cash_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1)
        )

    def forward(
        self,
        asset_features: torch.Tensor,
        expert_latent: torch.Tensor,
        expert_mask: torch.Tensor,
        asset_mask: torch.Tensor,
        account_state: torch.Tensor,
    ) -> torch.Tensor:
        unbatched = asset_features.ndim == 2
        if unbatched:
            asset_features = asset_features.unsqueeze(0)
            expert_latent = expert_latent.unsqueeze(0)
            expert_mask = expert_mask.unsqueeze(0)
            asset_mask = asset_mask.unsqueeze(0)
            account_state = account_state.unsqueeze(0)

        valid_assets = asset_mask.to(torch.bool)
        if not valid_assets.any(dim=-1).all():
            raise ValueError("정책 입력에는 최소 한 개의 유효 종목이 필요합니다.")

        available = expert_mask.to(torch.bool)
        latent = torch.where(available.unsqueeze(-1), expert_latent, torch.zeros_like(expert_latent))
        encoded_assets = self.asset_encoder(torch.cat((
            asset_features, latent, available.to(asset_features.dtype).unsqueeze(-1)
        ), dim=-1))
        query = self.asset_pool_query.expand(asset_features.shape[0], -1, -1)
        pooled, _ = self.cross_asset_pool(
            query, encoded_assets, encoded_assets,
            key_padding_mask=~valid_assets, need_weights=False,
        )
        pooled = self.pool_norm(pooled.squeeze(1))
        account = self.account_encoder(account_state)

        per_asset = torch.cat((
            encoded_assets,
            pooled.unsqueeze(1).expand_as(encoded_assets),
            account.unsqueeze(1).expand_as(encoded_assets),
        ), dim=-1)
        asset_alpha = F.softplus(self.allocation_head(per_asset).squeeze(-1)) + 1e-4
        cash_alpha = F.softplus(self.cash_head(torch.cat((pooled, account), dim=-1))) + 1e-4
        # The mask is consumed by MaskedDirichlet. Padded positions never enter
        # the distribution's normalization or log probability.
        concentration = torch.cat((asset_alpha, cash_alpha), dim=-1)
        return concentration.squeeze(0) if unbatched else concentration


class SharedPortfolioValue(nn.Module):
    """Value baseline with masked mean pooling over the variable asset set."""

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        input_size = ASSET_FEATURE_SIZE + EXPERT_LATENT_SIZE + 1
        self.asset_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.account_encoder = nn.Sequential(nn.Linear(ACCOUNT_FEATURE_SIZE, hidden_size), nn.Tanh())
        self.value_head = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1))

    def forward(
        self,
        asset_features: torch.Tensor,
        expert_latent: torch.Tensor,
        expert_mask: torch.Tensor,
        asset_mask: torch.Tensor,
        account_state: torch.Tensor,
    ) -> torch.Tensor:
        unbatched = asset_features.ndim == 2
        if unbatched:
            asset_features = asset_features.unsqueeze(0)
            expert_latent = expert_latent.unsqueeze(0)
            expert_mask = expert_mask.unsqueeze(0)
            asset_mask = asset_mask.unsqueeze(0)
            account_state = account_state.unsqueeze(0)
        available = expert_mask.to(torch.bool)
        latent = torch.where(available.unsqueeze(-1), expert_latent, torch.zeros_like(expert_latent))
        encoded = self.asset_encoder(torch.cat((
            asset_features, latent, available.to(asset_features.dtype).unsqueeze(-1)
        ), dim=-1))
        mask = asset_mask.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        result = self.value_head(torch.cat((pooled, self.account_encoder(account_state)), dim=-1))
        return result.squeeze(0) if unbatched else result


class MaskedDirichlet(torch.distributions.Distribution):
    """Dirichlet simplex distribution whose padded asset actions are exactly zero."""

    arg_constraints = {}
    has_rsample = False

    def __init__(self, concentration: torch.Tensor, asset_mask: torch.Tensor, validate_args=None):
        if concentration.shape[-1] != asset_mask.shape[-1] + 1:
            raise ValueError("concentration은 종목 수에 현금 비중 1개를 더한 차원이어야 합니다.")
        self.asset_mask = asset_mask.to(torch.bool)
        self.valid = torch.cat((self.asset_mask, torch.ones_like(self.asset_mask[..., :1])), dim=-1)
        self.concentration = torch.where(self.valid, concentration.clamp_min(1e-6), torch.ones_like(concentration))
        self._batch_shape = self.concentration.shape[:-1]
        self._event_shape = self.concentration.shape[-1:]
        super().__init__(self._batch_shape, self._event_shape, validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        return deterministic_allocation(self.concentration, self.asset_mask)

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        alpha = self.concentration.expand(shape)
        valid = self.valid.expand(shape)
        gamma = torch._standard_gamma(alpha)
        gamma = gamma.masked_fill(~valid, 0)
        return gamma / gamma.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(gamma.dtype).tiny)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        valid = self.valid.expand_as(value)
        alpha = self.concentration.expand_as(value)
        total = alpha.masked_fill(~valid, 0).sum(dim=-1)
        normalizer = torch.lgamma(total) - torch.where(valid, torch.lgamma(alpha), 0).sum(dim=-1)
        log_terms = torch.where(valid, (alpha - 1) * value.clamp_min(1e-12).log(), 0).sum(dim=-1)
        return normalizer + log_terms

    def entropy(self) -> torch.Tensor:
        valid = self.valid
        alpha = self.concentration
        total = alpha.masked_fill(~valid, 0).sum(dim=-1)
        count = valid.sum(dim=-1)
        return (
            torch.lgamma(alpha.masked_fill(~valid, 1)).sum(dim=-1) - torch.lgamma(total)
            + (total - count) * torch.digamma(total)
            - torch.where(valid, (alpha - 1) * torch.digamma(alpha), 0).sum(dim=-1)
        )


def deterministic_allocation(concentration: torch.Tensor, asset_mask: torch.Tensor) -> torch.Tensor:
    """Masked Dirichlet expectation, shared by historical and live inference."""
    if concentration.shape[-1] != asset_mask.shape[-1] + 1:
        raise ValueError("정책 concentration과 asset mask 차원이 다릅니다.")
    valid = torch.cat((asset_mask.to(torch.bool), torch.ones_like(asset_mask[..., :1], dtype=torch.bool)), dim=-1)
    values = concentration.masked_fill(~valid, 0)
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def build_torchrl_policy(asset_count: int, device: torch.device | str = "cpu") -> ProbabilisticActor:
    """Build the same parameter-shared actor for any runtime asset count."""
    if asset_count < 1:
        raise ValueError("정책 actor에는 유효 종목이 최소 하나 필요합니다.")
    target = torch.device(device)
    actor = TensorDictModule(
        SharedAssetAllocationActor().to(target),
        in_keys=["asset_features", "expert_latent", "expert_mask", "asset_mask", "account_state"],
        out_keys=["concentration"],
    )
    spec = Unbounded(shape=(asset_count + 1,), dtype=torch.float32, device=target)
    return ProbabilisticActor(
        module=actor,
        spec=spec,
        in_keys={"concentration": "concentration", "asset_mask": "asset_mask"},
        distribution_class=MaskedDirichlet,
        return_log_prob=True,
    ).to(target)


def deterministic_action(policy: ProbabilisticActor, observation: TensorDict) -> torch.Tensor:
    """Shared deterministic inference helper for historical and live decisions."""
    _, params = policy.get_dist_params(observation)
    return deterministic_allocation(params["concentration"], observation["asset_mask"])
