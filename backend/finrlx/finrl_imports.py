from __future__ import annotations

import importlib
import sys
import types

def backtest_engine_types():
    """Load the installed FinRL-Trading engine despite its source/install layout."""
    try:
        importlib.import_module("src.data.data_fetcher")
    except ImportError:
        # The wheel installs ``src/data`` as top-level ``data`` while the
        # checked source engine retains ``src.data`` imports.
        bridge = sys.modules.get("src") or types.ModuleType("src")
        bridge.__path__ = list(getattr(bridge, "__path__", []))
        sys.modules["src"] = bridge
        data_package = importlib.import_module("data")
        data_fetcher = importlib.import_module("data.data_fetcher")
        sys.modules["src.data"] = data_package
        sys.modules["src.data.data_fetcher"] = data_fetcher
        bridge.data = data_package
    module = importlib.import_module("backtest.backtest_engine")
    return module.BacktestConfig, module.BacktestEngine
