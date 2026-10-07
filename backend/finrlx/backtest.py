from __future__ import annotations

import bt
import pandas as pd

from .finrl_imports import backtest_engine_types
from .config import TRANSACTION_COST
from .rl.backtest_policy import generate_policy_weights, policy_source
from .strategy import TARGET_WEIGHT_STRATEGY_NAME


def make_weight_signals(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return real ticker prices and causal, next-close target weights."""
    prices = features.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    signal_weights = generate_policy_weights(features, prices.columns.tolist())
    execution_weights = signal_weights.reindex(prices.index).ffill().fillna(0.0)
    # Signal is formed after close[t]; execute at the next available close[t+1].
    execution_weights = execution_weights.shift(1).fillna(0.0)
    return prices, execution_weights


class CashPreservingBacktestEngine:
    """FinRL-Trading adapter that preserves uninvested capital as bt cash.

    The pinned BacktestEngine normalizes every weight row to 1 before calling
    bt. This adapter retains the official engine's price prep, fee model, and
    metric functions while bypassing only that normalization step.
    """

    def __init__(self, config) -> None:
        _, engine_type = backtest_engine_types()
        self.engine = engine_type(config)

    def run_backtest(self, strategy_name: str, price_data: pd.DataFrame,
                     weight_signals: pd.DataFrame):
        # Reuse the installed FinRL-Trading engine's data preparation,
        # commission and metrics paths. Bypass only its two sum-to-one steps.
        price_wide = self.engine._prepare_price_data_for_bt(price_data)
        signals = weight_signals.copy()
        signals.index = pd.to_datetime(signals.index)
        signals = signals.sort_index()
        signals = signals.reindex(columns=price_wide.columns).reindex(price_wide.index).ffill().fillna(0.0)
        if (signals < -1e-10).any().any() or (signals.sum(axis=1) > 1 + 1e-8).any():
            raise ValueError("백테스트 ticker weight 합이 1을 넘거나 음수입니다.")
        # bt's WeighTarget leaves the unallocated NAV in its native cash account.
        strategy = bt.Strategy(strategy_name, [
            bt.algos.RunAfterDate(signals.index.min()),
            bt.algos.RunOnDate(*signals.index.tolist()),
            bt.algos.SelectThese(list(signals.columns)),
            bt.algos.WeighTarget(signals),
            bt.algos.Rebalance(),
        ])
        backtest = self.engine._build_bt_backtest(strategy, price_wide, price_data)
        output = bt.run(backtest)[strategy_name]
        portfolio_values = output.prices
        returns = portfolio_values.pct_change().dropna()
        if len(portfolio_values):
            scale = self.engine.config.initial_capital / float(portfolio_values.iloc[0])
            portfolio_values = portfolio_values * scale
        metrics = self.engine._calculate_comprehensive_metrics(output, returns)
        return portfolio_values, metrics


def run_finrl_backtest(features: pd.DataFrame, initial_capital: float = 10_000_000) -> dict:
    try:
        BacktestConfig, _ = backtest_engine_types()
    except ImportError as exc:
        raise RuntimeError("FinRL-Trading backtest 엔진을 불러오지 못했습니다. 설치 로그를 확인하세요.") from exc
    prices, weights = make_weight_signals(features)
    if len(prices) < 30:
        raise ValueError("백테스트에는 최소 30개 시세 구간이 필요합니다.")
    long_prices = features[["timestamp", "symbol", "close", "volume"]].rename(
        columns={"timestamp": "datadate", "symbol": "tic", "close": "adj_close", "volume": "cshtrd"})
    config = BacktestConfig(start_date=str(prices.index.min().date()), end_date=str(prices.index.max().date()),
                            initial_capital=initial_capital, transaction_cost=TRANSACTION_COST,
                            benchmark_tickers=[], integer_positions=False)
    engine = CashPreservingBacktestEngine(config)
    last_weights = weights.iloc[-1].to_dict()
    source = policy_source(features)
    portfolio_values, metrics = engine.run_backtest(TARGET_WEIGHT_STRATEGY_NAME, long_prices, weights)
    cash_weight = float(1.0 - sum(last_weights.values()))
    return {"strategy": TARGET_WEIGHT_STRATEGY_NAME, "source": source,
            "signal_time": "bar_close[t]", "execution_time": "next_available_close[t+1]",
            "metrics": {str(k): _float(v) for k, v in metrics.items()},
            "dates": [pd.Timestamp(x).isoformat() for x in portfolio_values.index],
            "equity": [_float(v) for v in portfolio_values.to_numpy()],
            "weights": {str(k): _float(v) for k, v in last_weights.items()},
            "cash_weight": _float(cash_weight), "initial_capital": initial_capital,
            "transaction_cost": TRANSACTION_COST}


def _float(value) -> float | None:
    try:
        number = float(value)
        return number if pd.notna(number) else None
    except (TypeError, ValueError):
        return None
