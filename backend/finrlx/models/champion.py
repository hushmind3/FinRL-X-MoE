from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any, Protocol

import torch

from ..config import CHAMPION_PATH, EXPERTS_DIR
from .contracts import EXPERT_LATENT_SCHEMA, EXPERT_LATENT_SIZE


SUPPORTED_FORMAT = "registered_vertical_trading_moe_v1"
CONTROLLER_PREFIX = "controller."


class FusedFeatureAdapter(Protocol):
    latent_schema: str
    latent_size: int

    def extract_live(self, event: dict[str, Any]): ...

    def extract_historical(self, features, symbols, dates): ...


class ChampionCheckpoint:
    """Read-only checkpoint manifest and optional fused-feature adapter boundary."""

    def __init__(self, path: Path = CHAMPION_PATH, experts_dir: Path = EXPERTS_DIR) -> None:
        self.path, self.experts_dir = path, experts_dir
        self._state: dict[str, Any] | None = None
        self._lock = RLock()
        self._feature_adapter: FusedFeatureAdapter | None = None

    def register_feature_adapter(self, adapter: FusedFeatureAdapter) -> None:
        if (adapter.latent_schema != EXPERT_LATENT_SCHEMA
                or adapter.latent_size != EXPERT_LATENT_SIZE):
            raise ValueError("합체 Expert feature adapter의 schema가 RL policy 계약과 다릅니다.")
        self._feature_adapter = adapter

    def _load(self) -> dict[str, Any]:
        with self._lock:
            if self._state is None:
                self._state = torch.load(self.path, map_location="cpu", weights_only=True, mmap=True)
            return self._state

    def inspect(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"status": "missing", "available": False, "inference_ready": False,
                    "path": str(self.path), "error": "champion.pt 파일을 찾지 못했습니다."}
        try:
            state = self._load()
            state_dict = state.get("state_dict", {})
            mapping = state.get("expert_mapping", {})
            config = state.get("config", {})
            experts = []
            for key, entry in mapping.items():
                backend = str(entry.get("backend", "unknown"))
                experts.append({
                    "id": key,
                    "name": entry.get("name", key),
                    "backend": backend,
                    "input_features": config.get("feature_sizes", {}).get(key),
                    "asset_status": self._asset_status(key, backend),
                })
            controller_keys = [key for key in state_dict if key.startswith(CONTROLLER_PREFIX)]
            has_market_fusion = any("controller.market_fusion." in key for key in controller_keys)
            metadata = state.get("metadata", {})
            has_native_expert_runner = isinstance(metadata.get("native_runner_source"), str)
            has_fusion_runner = any(
                isinstance(metadata.get(key), str) and "market_fusion" in metadata[key]
                for key in ("fusion_runner_source", "controller_source", "inference_runner_source")
            )
            supported = state.get("format") == SUPPORTED_FORMAT
            inference_ready = bool(supported and has_market_fusion and self._feature_adapter is not None)
            return {
                "status": "checkpoint_ready" if supported else "unsupported_format",
                "available": supported,
                "path": str(self.path),
                "size_bytes": self.path.stat().st_size,
                "format": state.get("format"),
                "optimizer_updates": int(state.get("optimizer_updates", 0)),
                "state_tensor_count": len(state_dict),
                "expert_count": len(experts),
                "experts": experts,
                "controller_tensor_count": len(controller_keys),
                "market_fusion_weights_present": has_market_fusion,
                "native_expert_runner_present": has_native_expert_runner,
                "fused_inference_runner_present": has_fusion_runner,
                "feature_adapter_registered": self._feature_adapter is not None,
                "expert_latent_schema": EXPERT_LATENT_SCHEMA,
                "inference_ready": inference_ready,
                "inference_status": "ready" if inference_ready else "weights_without_fused_forward",
                "note": (
                    "체크포인트의 controller 가중치는 읽었습니다. 개별 Expert runner 소스는 있지만 합체 Controller의 forward 구현이 없어 latent를 생성하지 않습니다."
                    if not inference_ready else "합체 Controller의 native forward adapter를 사용할 수 있습니다."
                ),
            }
        except Exception as exc:
            with self._lock:
                self._state = None
            return {"status": "load_error", "available": False, "inference_ready": False,
                    "path": str(self.path), "error": f"{type(exc).__name__}: {exc}"}

    def extract_live(self, event: dict[str, Any]) -> tuple[list[float], bool] | None:
        """Return a per-asset fused latent only when a real forward adapter exists."""
        adapter = self._feature_adapter
        if adapter is None:
            return None
        latent = adapter.extract_live(event)
        if latent is None:
            return None
        values = torch.as_tensor(latent, dtype=torch.float32).reshape(-1)
        if values.numel() != EXPERT_LATENT_SIZE or not torch.isfinite(values).all():
            raise ValueError(f"합체 Expert latent은 유한한 {EXPERT_LATENT_SIZE}차원 벡터여야 합니다.")
        return values.tolist(), True

    def extract_historical(self, features, symbols, dates):
        """Batch hook used by the latent cache; absent forward code means unavailable."""
        adapter = self._feature_adapter
        return None if adapter is None else adapter.extract_historical(features, symbols, dates)

    def _asset_status(self, key: str, backend: str) -> str:
        if not self.experts_dir.is_dir():
            return "missing"
        # Names are read from the checkpoint mapping. Do not invent role groups.
        locations = {
            "fincast": ("market", "FinCast"),
            "exaone": ("market", "EXAONE-Forecast-for-Finance-1.0"),
            "kronos": ("market", "Kronos-base"),
            "marketgpt": ("market", "MarketGPT-100m"),
            "chronos": ("market", "Chronos_Small_2023_Global"),
            "timesfm": ("market", "TimesFM_20M_2023_Global"),
            "timemoe": ("market", "TimeMoE-200M"),
            "toto": ("market", "Toto-2.0-313m"),
            "macrophft": ("action", "MacroHFT"),
            "stock_policy": ("action", "stock"),
        }
        location = locations.get(backend)
        if location is None:
            return "unknown"
        return "present" if self.experts_dir.joinpath(*location).exists() else "missing"


champion = ChampionCheckpoint()
