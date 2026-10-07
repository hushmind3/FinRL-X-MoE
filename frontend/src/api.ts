import type { AppState, BacktestResult, FeedState, MarketSnapshot, TrainingState } from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'content-type': 'application/json', ...init?.headers },
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = typeof body.detail === 'string' ? body.detail : body.detail?.message
    throw new Error(detail || `요청 실패 (${response.status})`)
  }
  return body as T
}

export const api = {
  state: () => request<AppState>('/api/v1/state'),
  snapshot: () => request<MarketSnapshot>('/api/v1/feed/snapshot'),
  inspectChampion: () => request('/api/v1/models/champion/inspect', { method: 'POST' }),
  startFeed: (symbols: string[], provider: string) => request<FeedState>('/api/v1/feed/start', {
    method: 'POST', body: JSON.stringify({ symbols, provider, timeframe: '1Min' }),
  }),
  stopFeed: () => request<FeedState>('/api/v1/feed/stop', { method: 'POST' }),
  startTraining: (algorithm = 'PPO') => request<TrainingState>('/api/v1/training/start', {
    method: 'POST', body: JSON.stringify({ algorithm, steps_per_iteration: 32, learning_rate: 0.0003, paper: true }),
  }),
  stopTraining: () => request<TrainingState>('/api/v1/training/stop', { method: 'POST' }),
  startPaper: () => request('/api/v1/paper/start', { method: 'POST' }),
  stopPaper: () => request('/api/v1/paper/stop', { method: 'POST' }),
  backtest: (input: { symbols: string[]; start: string; end: string; initial_capital: number }) =>
    request<BacktestResult>('/api/v1/backtest', { method: 'POST', body: JSON.stringify(input) }),
}

export function marketSocketUrl() {
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${scheme}//${location.host}/ws/market`
}
