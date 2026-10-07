from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

class FeedStartRequest(BaseModel):
    symbols: list[str] = Field(min_length=1)
    provider: Literal["yahoo_poll", "alpaca"] = "yahoo_poll"
    timeframe: Literal["1Min"] = "1Min"

class TrainRequest(BaseModel):
    algorithm: str = Field(default="PPO", min_length=1)
    steps_per_iteration: int = Field(default=32, ge=32, le=512)
    learning_rate: float = Field(default=3e-4, gt=1e-6, le=1e-2)
    paper: bool = True

class BacktestRequest(BaseModel):
    symbols: list[str] = Field(min_length=1)
    start: str
    end: str
    initial_capital: float = Field(default=10_000_000, gt=0)

class WeightTarget(BaseModel):
    as_of: str
    weights: dict[str, float]
    cash_weight: float = Field(ge=0.0, le=1.0)
    source: Literal["policy", "baseline"]
