from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict

from ..config import RUNTIME_DIR, TRANSACTION_COST
from ..market.features import FEATURE_KEYS
from ..models.latent_cache import historical_latents
from .policy import ACCOUNT_FEATURE_SIZE, ARCHITECTURE_VERSION, PolicyContract
from .policy import build_torchrl_policy, deterministic_action


def generate_policy_weights(features: pd.DataFrame, universe: list[str] | None = None) -> pd.DataFrame:
    """Historical rollout through the exact actor and deterministic helper used live."""
    symbols = list(dict.fromkeys(str(s).upper() for s in (universe or features["symbol"].unique())))
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(features["timestamp"].unique())))
    state = _load_compatible_checkpoint()
    if state is None:
        raise ValueError("호환 가능한 variable-universe TorchRL policy checkpoint가 없습니다.")
    policy = build_torchrl_policy(len(symbols), "cpu")
    policy.module.load_state_dict(state["policy"], strict=True)
    policy.eval()
    latent, expert_mask = historical_latents(features, symbols, dates)
    lookup = {(pd.Timestamp(row.timestamp), str(row.symbol).upper()): row._asdict()
              for row in features.itertuples(index=False)}
    values = np.full((len(dates), len(symbols)), np.nan, dtype=np.float32)
    current = np.zeros(len(symbols), dtype=np.float32)
    current_cash = 1.0
    pending: np.ndarray | None = None
    equity = peak = 1.0
    last_turnover = 0.0
    for date_idx, date in enumerate(dates):
        market = np.zeros((len(symbols), len(FEATURE_KEYS) + 1), dtype=np.float32)
        mask = np.zeros(len(symbols), dtype=bool)
        for asset_idx, symbol in enumerate(symbols):
            row = lookup.get((pd.Timestamp(date), symbol))
            if row is None:
                continue
            market[asset_idx, :-1] = [float(row.get(key, 0) or 0) for key in FEATURE_KEYS]
            market[asset_idx, -1] = current[asset_idx]
            mask[asset_idx] = True
        if not mask.any():
            continue
        # Match the live environment: the previous close's action executes
        # now, after current holdings earned the completed interval return.
        interval_returns = np.nan_to_num(market[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
        gross_return = float(np.dot(current, interval_returns))
        drifted_value = np.r_[current * (1.0 + interval_returns), current_cash]
        drifted = drifted_value / max(1.0 + gross_return, 1e-12)
        equity *= max(1e-8, 1.0 + gross_return)
        if pending is not None:
            last_turnover = float(np.abs(pending[:-1] - drifted[:-1]).sum())
            equity *= max(1e-8, 1.0 - TRANSACTION_COST * last_turnover)
            current = pending[:-1].copy()
            current_cash = float(pending[-1])
        peak = max(peak, equity)
        market[:, -1] = current
        account = np.asarray([equity - 1.0, equity / peak - 1.0, current_cash,
                              last_turnover], dtype=np.float32)
        account = np.clip(account, -5, 5)
        obs = TensorDict({
            "asset_features": torch.from_numpy(np.nan_to_num(market)),
            "expert_latent": torch.from_numpy(latent[date_idx]),
            "expert_mask": torch.from_numpy(expert_mask[date_idx]),
            "asset_mask": torch.from_numpy(mask),
            "account_state": torch.from_numpy(account.copy()),
        }, batch_size=[])
        with torch.inference_mode():
            allocation = deterministic_action(policy, obs)
        pending = allocation.cpu().numpy().astype(np.float32)
        values[date_idx] = pending[:-1]
    result = pd.DataFrame(values, index=dates, columns=symbols)
    return result


def policy_source(features: pd.DataFrame) -> str:
    state = _load_compatible_checkpoint()
    if state is None:
        return "torchrl_policy_checkpoint_unavailable"
    return "torchrl_shared_asset_policy"


def _load_compatible_checkpoint() -> dict | None:
    path = RUNTIME_DIR / "policy_torchrl.pt"
    if not path.is_file():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("architecture_version") != ARCHITECTURE_VERSION:
            return None
        if state.get("policy_contract") != PolicyContract().as_dict():
            return None
        return state
    except (OSError, RuntimeError, ValueError, TypeError):
        return None
