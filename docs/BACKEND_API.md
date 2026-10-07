# Backend contract v1

The React client consumes this contract. The backend owns provider status, expert-checkpoint status, policy state, account state, and target weights.

## REST

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Backend reachability and API version |
| `GET` | `/api/v1/state` | Unified feed, champion, training, paper, weight, and live-gate state |
| `GET` | `/api/v1/models/champion` | Lazy checkpoint and expert manifest inspection |
| `POST` | `/api/v1/models/champion/inspect` | Explicit checkpoint inspection |
| `POST` | `/api/v1/feed/start` | Subscribe to 1-minute market bars |
| `POST` | `/api/v1/feed/stop` | Stop feed subscription |
| `GET` | `/api/v1/feed/snapshot` | Recent bars and current quote per symbol |
| `POST` | `/api/v1/training/start` | Start a registered TorchRL learner on the current feed |
| `POST` | `/api/v1/training/stop` | Stop collector and persist policy |
| `POST` | `/api/v1/paper/start` | Enable the local fee-aware paper ledger |
| `POST` | `/api/v1/paper/stop` | Disable paper rebalancing |
| `GET` | `/api/v1/paper/account` | Equity, cash, positions, and recent fills |
| `POST` | `/api/v1/backtest` | Fetch history and run FinRL-Trading BacktestEngine |
| `POST` | `/api/v1/trading/live/start` | Closed until a broker adapter and risk gate exist |

Feed request:

```json
{"symbols":["AAPL","MSFT"],"provider":"alpaca","timeframe":"1Min"}
```

Set `ALPACA_API_KEY`, `ALPACA_API_SECRET`, and optionally `ALPACA_DATA_FEED=iex` in `.env` for Alpaca market streaming. `yahoo_poll` is a polling fallback and the status explicitly labels it delayed/polled.

Training request:

```json
{"algorithm":"PPO","steps_per_iteration":256,"learning_rate":0.0003,"paper":true}
```

Backtest request:

```json
{"symbols":["AAPL","MSFT"],"start":"2023-01-01","end":"2026-01-01","initial_capital":10000000}
```

## WebSocket

Connect to `/ws/market`. The first frame is `snapshot`; subsequent frames are one of `market_event`, `feed_status`, `training_status`, or `policy_decision`.

```json
{"type":"market_event","data":{"symbol":"AAPL","timestamp":"2026-10-08T14:30:00+00:00","close":225.4,"provider":"alpaca","momentum_1":0.0012},"status":{"running":true}}
```

```json
{"type":"policy_decision","data":{"as_of":"2026-10-08T14:31:00+00:00","signal_time":"2026-10-08T14:30:00+00:00","execution_time":"2026-10-08T14:31:00+00:00","weights":{"AAPL":0.2,"MSFT":0.25},"cash_weight":0.55,"reward":0.04}}
```

Policy decisions carry `signal_time` and `execution_time`. The signal uses the bar close available at time `t`; the environment applies it at the next available close. `weights` contains real tickers only, and the cash residual is `cash_weight`. An all-cash target has no ticker rows in FinRL `StrategyResult` and creates no order.

`algorithm` selects a registered learner. PPO is currently the included baseline. Other learner names are rejected until a corresponding TorchRL learner implementation is registered. `champion.available` means the checkpoint manifest was read; `champion.inference_ready` means a real fused forward adapter is installed. The current checkpoint has model weights and individual runner source, but not the fused Controller forward implementation, so the Expert latent mask remains unavailable. No technical-score allocation fallback is used.
