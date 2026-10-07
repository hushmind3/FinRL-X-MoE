from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from ..config import CHAMPION_PATH, RUNTIME_DIR
from .contracts import EXPERT_LATENT_SIZE
from .champion import champion


LATENT_SIZE = EXPERT_LATENT_SIZE
CACHE_SCHEMA = "champion_historical_latents_v1"


def historical_latents(
    features: pd.DataFrame,
    symbols: list[str],
    dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or batch-compute fused features once before historical rollout.

    The TorchRL rollout never calls an Expert. If the checkpoint has no fused
    forward adapter, return an unavailable mask and do not cache fake vectors.
    """
    shape = (len(dates), len(symbols), LATENT_SIZE)
    missing = np.zeros(shape, dtype=np.float32), np.zeros(shape[:2], dtype=bool)
    fingerprint = _fingerprint(features, symbols, dates)
    path = RUNTIME_DIR / "expert_latents" / f"{fingerprint}.npz"
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as cached:
                latent = cached["latent"]
                mask = cached["mask"]
            if latent.shape == shape and mask.shape == shape[:2]:
                return latent.astype(np.float32, copy=False), mask.astype(bool, copy=False)
        except (OSError, ValueError, KeyError):
            pass

    computed = champion.extract_historical(features, symbols, dates)
    if computed is None:
        return missing
    latent, mask = computed
    latent = np.asarray(latent, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if latent.shape != shape or mask.shape != shape[:2]:
        raise ValueError(f"fused Expert batch 출력 shape가 계약과 다릅니다: {latent.shape}, {mask.shape}")
    finite_rows = np.isfinite(latent).all(axis=-1)
    mask &= finite_rows
    latent = np.where(mask[..., None], latent, 0).astype(np.float32, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, latent=latent, mask=mask)
    tmp.replace(path)
    return latent, mask


def _fingerprint(features: pd.DataFrame, symbols: list[str], dates: pd.DatetimeIndex) -> str:
    digest = hashlib.sha256()
    digest.update(CACHE_SCHEMA.encode())
    digest.update("|".join(symbols).encode())
    digest.update(dates.asi8.tobytes())
    if not CHAMPION_PATH.is_file():
        digest.update(b"checkpoint-missing")
    else:
        stat = CHAMPION_PATH.stat()
        digest.update(str((stat.st_size, stat.st_mtime_ns)).encode())
    if not features.empty:
        digest.update(pd.util.hash_pandas_object(features, index=True).values.tobytes())
    return digest.hexdigest()
