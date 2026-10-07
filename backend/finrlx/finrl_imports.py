from __future__ import annotations

import importlib
import sys
import types

def ensure_finrl_source_layout():
    """Bridge the installed wheel's top-level modules to its src.* imports."""
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
    bridge = sys.modules.get("src")
    if bridge is not None:
        try:
            config_package = importlib.import_module("config")
            sys.modules["src.config"] = config_package
            bridge.config = config_package
        except ImportError:
            pass


def finrl_data_manager(cache_dir: str):
    """Return FinRL-X's unified data manager using this app's cache directory."""
    ensure_finrl_source_layout()
    module = importlib.import_module("data.data_fetcher")
    return module.get_data_manager(cache_dir=cache_dir)

def backtest_engine_types():
    """Load the installed FinRL-Trading engine despite its source/install layout."""
    ensure_finrl_source_layout()
    module = importlib.import_module("backtest.backtest_engine")
    return module.BacktestConfig, module.BacktestEngine
