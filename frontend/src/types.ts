export type FeedState = {
  running: boolean
  provider: string
  timeframe: string
  symbols: string[]
  last_error?: string | null
  last_update?: string | null
  data_quality: 'live_stream' | 'polled_delayed' | 'none' | string
}

export type ExpertEntry = {
  id: string
  name: string
  backend: string
  input_features?: number
  asset_status: string
}

export type ChampionState = {
  status: string
  available: boolean
  path?: string
  size_bytes?: number
  format?: string
  optimizer_updates?: number
  state_tensor_count?: number
  controller_tensor_count?: number
  expert_count?: number
  experts?: ExpertEntry[]
  inference_ready: boolean
  inference_status?: string
  note?: string
  error?: string
}

export type TrainingState = {
  status: string
  running: boolean
  algorithm: string
  steps: number
  updates: number
  mean_reward?: number | null
  last_loss?: number | null
  device: string
  error?: string | null
  checkpoint: string
  symbols: string[]
  rollout: string
  available_algorithms?: string[]
}

export type PaperState = {
  enabled: boolean
  cash: number
  equity: number
  return_pct: number
  positions: Record<string, number>
  weights: Record<string, number>
  fills: Array<{ timestamp: string; symbol: string; side: string; price: number; notional: number; fee: number; slippage: number }>
  error?: string | null
}

export type TargetWeights = {
  as_of: string
  signal_time?: string
  execution_time?: string
  weights: Record<string, number>
  cash_weight: number
  source: string
  observation_source?: string
  expert_checkpoint_status?: string
}

export type AppState = {
  mode: string
  live_orders_enabled: boolean
  live_gate: { status: string; reason: string }
  feed: FeedState
  champion: ChampionState
  training: TrainingState
  inference: { running: boolean; error?: string | null }
  paper: PaperState
  target_weights: TargetWeights | null
  last_decision: Record<string, unknown> | null
  error?: string | null
}

export type Bar = {
  type?: string
  symbol: string
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  provider: string
  momentum_1?: number
  momentum_5?: number
  momentum_20?: number
  volatility_20?: number
  trend_20?: number
  trend_60?: number
  range_pct?: number
  volume_z?: number
  drawdown_60?: number
}

export type MarketSnapshot = {
  status: FeedState
  quotes: Bar[]
  bars: Record<string, Bar[]>
}

export type BacktestResult = {
  strategy: string
  source: string
  metrics: Record<string, number | null>
  dates: string[]
  equity: Array<number | null>
  weights: Record<string, number | null>
  cash_weight?: number | null
  signal_time?: string
  execution_time?: string
  initial_capital: number
  transaction_cost: number
  symbol_count: number
}

export type SocketEvent =
  | { type: 'snapshot'; data: MarketSnapshot; runtime: AppState }
  | { type: 'market_event'; data: Bar; status: FeedState }
  | { type: 'feed_status'; status: FeedState }
  | { type: 'inference_status'; data: { running: boolean; error?: string | null } }
  | { type: 'training_status'; data: TrainingState }
  | { type: 'policy_decision'; data: TargetWeights & { reward: number; equity_return: number } }
