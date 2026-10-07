import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowDownRight, ArrowUpRight, BarChart3, BrainCircuit, Check, CircleAlert,
  Database, Gauge, Layers3, LoaderCircle, LockKeyhole, Pause, Play, Radio, RefreshCw,
  Sparkles, TrendingUp, Wallet, Wifi, WifiOff,
} from 'lucide-react'
import type { FormEvent, ReactNode } from 'react'
import { api, marketSocketUrl } from './api'
import type { AppState, BacktestResult, Bar, SocketEvent } from './types'

const initialSymbols = 'AAPL, MSFT, NVDA'

export function App() {
  const [state, setState] = useState<AppState | null>(null)
  const [bars, setBars] = useState<Record<string, Bar[]>>({})
  const [selected, setSelected] = useState('AAPL')
  const [symbolsInput, setSymbolsInput] = useState(initialSymbols)
  const [provider, setProvider] = useState<'alpaca' | 'yahoo_poll'>('yahoo_poll')
  const [socketConnected, setSocketConnected] = useState(false)
  const [busy, setBusy] = useState('')
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const [backtest, setBacktest] = useState<BacktestResult | null>(null)
  const [learner, setLearner] = useState('PPO')
  const [backtestRange, setBacktestRange] = useState(() => defaultDateRange())
  const [capital, setCapital] = useState(10_000_000)
  const socketRef = useRef<WebSocket | null>(null)

  const refresh = useCallback(async () => {
    try {
      const [nextState, snapshot] = await Promise.all([api.state(), api.snapshot()])
      setState(nextState)
      setBars(snapshot.bars || {})
      setSymbolsInput((current) => current === initialSymbols && nextState.feed.symbols.length
        ? nextState.feed.symbols.join(', ') : current)
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '백엔드에 연결하지 못했습니다.')
    }
  }, [])

  useEffect(() => {
    let reconnect: number | undefined
    let disposed = false
    const connect = () => {
      if (disposed) return
      const socket = new WebSocket(marketSocketUrl())
      socketRef.current = socket
      socket.onopen = () => setSocketConnected(true)
      socket.onclose = () => {
        setSocketConnected(false)
        if (!disposed) reconnect = window.setTimeout(connect, 2500)
      }
      socket.onerror = () => socket.close()
      socket.onmessage = (message) => {
        let event: SocketEvent
        try { event = JSON.parse(message.data) as SocketEvent } catch { return }
        if (event.type === 'snapshot') {
          setBars(event.data.bars || {})
          setState(event.runtime)
        } else if (event.type === 'market_event') {
          setState((previous) => previous ? { ...previous, feed: event.status } : previous)
          setBars((previous) => {
            const history = previous[event.data.symbol] || []
            const next = [...history]
            const last = next.at(-1)
            if (last?.timestamp === event.data.timestamp) next[next.length - 1] = event.data
            else next.push(event.data)
            return { ...previous, [event.data.symbol]: next.slice(-500) }
          })
        } else if (event.type === 'feed_status') {
          setState((previous) => previous ? { ...previous, feed: event.status } : previous)
        } else if (event.type === 'inference_status') {
          setState((previous) => previous ? { ...previous, inference: event.data } : previous)
        } else if (event.type === 'training_status') {
          setState((previous) => previous ? { ...previous, training: event.data } : previous)
        } else if (event.type === 'policy_decision') {
          setState((previous) => previous ? { ...previous, target_weights: event.data } : previous)
        }
      }
    }
    void refresh()
    connect()
    const timer = window.setInterval(() => void refresh(), 20_000)
    return () => {
      disposed = true
      window.clearInterval(timer)
      if (reconnect) window.clearTimeout(reconnect)
      socketRef.current?.close()
    }
  }, [refresh])

  const selectedBars = bars[selected] || []
  const currentQuote = selectedBars.at(-1)
  const previousQuote = selectedBars.at(-2)
  const change = currentQuote && previousQuote ? currentQuote.close / previousQuote.close - 1 : 0
  const sortedQuotes = useMemo(() => Object.entries(bars).map(([symbol, rows]) => ({ symbol, bar: rows.at(-1) }))
    .filter((item): item is { symbol: string; bar: Bar } => Boolean(item.bar))
    .sort((a, b) => (b.bar.momentum_1 || 0) - (a.bar.momentum_1 || 0)), [bars])
  const training = state?.training
  const feed = state?.feed
  const champion = state?.champion
  const paper = state?.paper
  const paperExecution = state?.paper_execution

  const perform = async (key: string, action: () => Promise<unknown>, success: string) => {
    setBusy(key); setNotice(''); setError('')
    try { await action(); setNotice(success); await refresh() }
    catch (reason) { setError(reason instanceof Error ? reason.message : '요청을 완료하지 못했습니다.') }
    finally { setBusy('') }
  }

  const startFeed = () => {
    const symbols = symbolsInput.split(',').map((symbol) => symbol.trim().toUpperCase()).filter(Boolean)
    void perform('feed', () => api.startFeed(symbols, provider), `${symbols.join(', ')} 시세 구독을 시작했습니다.`)
  }

  const runBacktest = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault(); setBusy('backtest'); setBacktest(null); setError(''); setNotice('')
    try {
      const symbols = symbolsInput.split(',').map((symbol) => symbol.trim().toUpperCase()).filter(Boolean)
      const result = await api.backtest({ symbols, start: backtestRange.start, end: backtestRange.end,
        initial_capital: capital })
      setBacktest(result)
    } catch (reason) { setError(reason instanceof Error ? reason.message : '백테스트에 실패했습니다.') }
    finally { setBusy('') }
  }

  const busyAction = Boolean(busy)
  const feedTone = feed?.running ? 'good' : 'idle'
  const dataLabel = feed?.data_quality === 'live_stream' ? 'LIVE STREAM'
    : feed?.data_quality === 'polled_delayed' ? 'POLLED · DELAYED' : 'FEED OFF'

  return <div className="min-h-screen bg-[#080d17] text-slate-100">
    <header className="sticky top-0 z-20 border-b border-white/[0.07] bg-[#080d17]/90 backdrop-blur-xl">
      <div className="mx-auto flex h-[68px] max-w-[1500px] items-center justify-between px-4 sm:px-7">
        <div className="flex items-center gap-3">
          <div className="grid size-9 place-items-center rounded-xl border border-cyan-300/20 bg-cyan-300/[0.08] text-cyan-200"><Layers3 size={18} /></div>
          <div><div className="text-sm font-bold tracking-[0.12em]">FINRL<span className="text-cyan-300">·</span>X</div><div className="mt-0.5 text-[9px] tracking-[0.2em] text-slate-500">TRADING LAB</div></div>
          <span className="ml-4 hidden rounded-full border border-emerald-300/20 bg-emerald-300/[0.06] px-2.5 py-1 text-[9px] font-semibold tracking-[0.12em] text-emerald-200 sm:inline-flex"><i className="mr-2 mt-1 size-1.5 rounded-full bg-emerald-300" />RESEARCH · PAPER</span>
        </div>
        <div className="flex items-center gap-4 text-[10px] text-slate-500">
          <span className="hidden sm:inline">{new Date().toLocaleDateString('ko-KR', { dateStyle: 'medium' })}</span>
          <span className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 ${socketConnected ? 'border-emerald-300/20 text-emerald-200' : 'border-slate-700 text-slate-500'}`}>
            {socketConnected ? <Wifi size={12} /> : <WifiOff size={12} />}{socketConnected ? 'API 연결' : 'API 연결 대기'}
          </span>
          <button onClick={() => void refresh()} className="rounded-lg p-2 text-slate-400 transition hover:bg-white/[0.06] hover:text-white" aria-label="새로고침"><RefreshCw size={15} /></button>
        </div>
      </div>
    </header>

    <main className="mx-auto max-w-[1500px] px-4 pb-12 pt-7 sm:px-7">
      <section className="mb-6 flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
        <div>
          <div className="eyebrow">LIVE MARKET · ONLINE LEARNING</div>
          <h1 className="mt-2 text-3xl font-semibold tracking-[-0.045em] sm:text-[38px]">시세를 보면서 <span className="text-cyan-200">정책을 학습합니다.</span></h1>
          <p className="mt-2 max-w-2xl text-xs leading-6 text-slate-400">시장 feature와 연결 가능한 Expert latent → TorchRL learner 목표 비중 → FinRL-X 백테스트와 Paper. 화면은 백엔드의 실제 feed, policy, account 상태를 표시합니다.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <ActionButton tone="neutral" disabled={busyAction} onClick={() => void perform('inspect', () => api.inspectChampion(), 'champion.pt 구조를 확인했습니다.')} icon={<Database size={15} />}>
            {busy === 'inspect' ? '체크포인트 확인 중…' : 'Champion 확인'}
          </ActionButton>
          {feed?.running
            ? <ActionButton tone="neutral" disabled={busyAction} onClick={() => void perform('feed-stop', api.stopFeed, '시세 구독을 중지했습니다.')} icon={<Pause size={15} />}>시세 중지</ActionButton>
            : <ActionButton tone="primary" disabled={busyAction} onClick={startFeed} icon={<Play size={15} />}>{busy === 'feed' ? '연결 중…' : '실시간 시세 시작'}</ActionButton>}
          {training?.running
            ? <ActionButton tone="violet" disabled={busyAction} onClick={() => void perform('train-stop', api.stopTraining, 'TorchRL learner를 중지했습니다.')} icon={<Pause size={15} />}>학습 중지</ActionButton>
              : <ActionButton tone="violet" disabled={busyAction || !feed?.running} onClick={() => void perform('train', () => api.startTraining(learner), `TorchRL ${learner} 학습을 시작했습니다.`)} icon={<BrainCircuit size={15} />}>{busy === 'train' ? '시작 중…' : `${learner} 학습 시작`}</ActionButton>}
        </div>
      </section>

      {(error || notice) && <div className={`mb-5 flex items-start gap-2 rounded-xl border px-4 py-3 text-xs ${error ? 'border-rose-400/20 bg-rose-400/[0.06] text-rose-200' : 'border-emerald-300/20 bg-emerald-300/[0.05] text-emerald-200'}`}>
        {error ? <CircleAlert className="mt-0.5 shrink-0" size={14} /> : <Check className="mt-0.5 shrink-0" size={14} />}{error || notice}
      </div>}

      <section className="mb-5 grid grid-cols-2 gap-3 xl:grid-cols-4">
        <MetricCard icon={<Radio size={16} />} label="MARKET FEED" value={dataLabel} detail={feed?.symbols.join(' · ') || '종목 구독 전'} tone={feedTone} />
        <MetricCard icon={<BrainCircuit size={16} />} label={`LEARNER ${training?.algorithm || learner}`} value={training?.running ? 'LEARNING' : (training?.status || 'IDLE').toUpperCase()} detail={`${fmt(training?.steps || 0, 0)} transitions · ${fmt(training?.updates || 0, 0)} updates`} tone={training?.running ? 'violet' : 'idle'} />
        <MetricCard icon={<TrendingUp size={16} />} label="MEAN REWARD" value={training?.mean_reward == null ? '—' : fmt(training.mean_reward, 4)} detail={training?.last_loss == null ? '첫 rollout 대기' : `loss ${fmt(training.last_loss, 5)}`} tone={training?.mean_reward != null && training.mean_reward >= 0 ? 'good' : 'idle'} />
        <MetricCard icon={<Wallet size={16} />} label="PAPER EQUITY" value={paper ? fmt(paper.equity, 0) : '—'} detail={paper ? `${paper.return_pct >= 0 ? '+' : ''}${fmt(paper.return_pct, 2)}% · ${Object.keys(paper.positions).length} positions` : '계좌 준비 중'} tone={paper?.return_pct != null && paper.return_pct >= 0 ? 'good' : 'idle'} />
      </section>

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        <article className="panel xl:col-span-8">
          <div className="flex flex-col justify-between gap-4 md:flex-row md:items-start">
            <div>
              <div className="eyebrow">MARKET MONITOR <span className="ml-2 text-slate-600">/</span> {dataLabel}</div>
              <div className="mt-2 flex flex-wrap items-baseline gap-3">
                <h2 className="text-xl font-semibold">{selected}</h2>
                <span className="font-mono text-[22px] font-semibold tabular-nums">{currentQuote ? fmt(currentQuote.close, 2) : '—'}</span>
                {currentQuote && <span className={`inline-flex items-center gap-1 text-xs ${change >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                  {change >= 0 ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}{change >= 0 ? '+' : ''}{fmt(change * 100, 2)}%
                </span>}
              </div>
              <div className="mt-1 text-[10px] text-slate-500">{currentQuote ? `업데이트 ${formatTime(currentQuote.timestamp)} · ${currentQuote.provider}` : '시세 구독을 시작하면 가격이 표시됩니다.'}</div>
            </div>
            <div className="flex min-w-0 flex-wrap items-center gap-2">
              <input value={symbolsInput} onChange={(event) => setSymbolsInput(event.target.value)} placeholder="AAPL, MSFT, NVDA" className="h-9 w-full min-w-[220px] max-w-[280px] rounded-lg border border-white/10 bg-[#0a111d] px-3 font-mono text-[11px] text-slate-200 outline-none placeholder:text-slate-600 focus:border-cyan-300/40" aria-label="구독 종목" />
              <select value={provider} onChange={(event) => setProvider(event.target.value as 'alpaca' | 'yahoo_poll')} className="h-9 rounded-lg border border-white/10 bg-[#0a111d] px-2 text-[10px] text-slate-300 outline-none" aria-label="시세 공급자">
                <option value="yahoo_poll">Yahoo · 폴링</option><option value="alpaca">Alpaca · 실시간</option>
              </select>
              <button onClick={startFeed} disabled={busyAction} className="h-9 rounded-lg border border-cyan-200/20 bg-cyan-200/[0.08] px-3 text-[10px] font-semibold text-cyan-100 transition hover:bg-cyan-200/[0.14] disabled:opacity-50">적용</button>
            </div>
          </div>
          <div className="mt-5 h-[250px] rounded-xl border border-white/[0.04] bg-[#09111c] p-2 sm:h-[300px]">
            <MarketChart bars={selectedBars} />
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <MiniStat label="OPEN" value={currentQuote ? fmt(currentQuote.open, 2) : '—'} />
            <MiniStat label="HIGH" value={currentQuote ? fmt(currentQuote.high, 2) : '—'} />
            <MiniStat label="LOW" value={currentQuote ? fmt(currentQuote.low, 2) : '—'} />
            <MiniStat label="VOLUME" value={currentQuote ? fmt(currentQuote.volume, 0) : '—'} />
          </div>
          {provider === 'yahoo_poll' && feed?.running && <p className="mt-3 text-[10px] text-amber-200/80">Yahoo 폴링 시세입니다. 시장 시간과 공급자 정책에 따라 지연될 수 있습니다.</p>}
        </article>

        <article className="panel xl:col-span-4">
          <PanelTitle eyebrow="WATCHLIST" title="구독 종목" trailing={<span className="rounded-full border border-white/10 px-2 py-1 font-mono text-[9px] text-slate-400">{sortedQuotes.length} SYMBOLS</span>} />
          <div className="mt-5 overflow-hidden rounded-xl border border-white/[0.06]">
            <div className="grid grid-cols-[1fr_90px_74px] bg-white/[0.025] px-3 py-2 text-[9px] tracking-[0.12em] text-slate-500"><span>SYMBOL</span><span className="text-right">PRICE</span><span className="text-right">1 BAR %</span></div>
            <div className="max-h-[280px] divide-y divide-white/[0.045] overflow-auto">
              {sortedQuotes.length ? sortedQuotes.map(({ symbol, bar }) => <button key={symbol} onClick={() => setSelected(symbol)} className={`grid w-full grid-cols-[1fr_90px_74px] items-center px-3 py-3 text-left transition hover:bg-white/[0.035] ${symbol === selected ? 'bg-cyan-300/[0.055]' : ''}`}>
                <span><b className="block text-xs font-semibold">{symbol}</b><small className="mt-1 block text-[9px] text-slate-500">{formatTime(bar.timestamp)}</small></span>
                <span className="text-right font-mono text-[11px] tabular-nums">{fmt(bar.close, 2)}</span>
                <span className={`text-right font-mono text-[10px] tabular-nums ${(bar.momentum_1 || 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>{fmt((bar.momentum_1 || 0) * 100, 2)}%</span>
              </button>) : <div className="px-3 py-10 text-center text-xs text-slate-500">시세를 구독하면 종목이 표시됩니다.</div>}
            </div>
          </div>
          <div className="mt-4 flex items-start gap-2 rounded-lg border border-white/[0.06] bg-white/[0.02] p-3 text-[10px] leading-5 text-slate-400"><Sparkles size={13} className="mt-0.5 shrink-0 text-cyan-200" /><span>1 bar 변화율은 시세 확인용입니다. 정책 action이나 매수 규칙으로 사용하지 않습니다.</span></div>
        </article>

        <article className="panel xl:col-span-5">
          <PanelTitle eyebrow="FROZEN ANALYSIS MODELS" title="Champion Expert" icon={<Database size={15} className="text-cyan-200" />} trailing={<StatusPill tone={champion?.available ? 'good' : 'idle'}>{champion?.available ? 'CHECKPOINT OK' : (champion?.status || 'LOADING').toUpperCase()}</StatusPill>} />
          <div className="mt-3 flex items-center justify-between gap-3 rounded-lg border border-white/[0.06] bg-[#0a111d] px-3 py-2.5">
            <div className="min-w-0"><div className="truncate font-mono text-[10px] text-slate-300">{champion?.path || '모델/champion.pt'}</div><div className="mt-1 text-[9px] text-slate-500">{champion?.size_bytes ? `${(champion.size_bytes / 1024 ** 3).toFixed(2)} GB · mmap` : '파일 확인 전'}</div></div>
            <div className="shrink-0 text-right"><div className="font-mono text-xs text-slate-200">{champion?.expert_count ?? '—'} experts</div><div className="mt-1 text-[9px] text-slate-500">{fmt(champion?.state_tensor_count || 0, 0)} tensors</div></div>
          </div>
          <div className="mt-3 flex items-start gap-2 rounded-lg border border-amber-300/15 bg-amber-300/[0.045] p-3 text-[10px] leading-5 text-amber-100/90">
            <CircleAlert size={14} className="mt-0.5 shrink-0" />
            <span>{champion?.inference_ready ? 'Expert 추론 연결됨' : (champion?.note || '체크포인트 manifest는 확인되며, native forward adapter 연결 전까지 예측으로 사용하지 않습니다.')}</span>
          </div>
          <div className="mt-3 max-h-[235px] space-y-1.5 overflow-auto pr-1">
            {(champion?.experts || []).map((expert) => <div key={expert.id} className="flex items-center justify-between gap-3 rounded-lg px-2.5 py-2 hover:bg-white/[0.025]">
              <div className="min-w-0"><div className="truncate text-[11px] font-medium text-slate-200">{expert.name}</div><div className="mt-0.5 font-mono text-[9px] text-slate-500">{expert.id} · {expert.backend} · input {expert.input_features ?? '—'}</div></div>
              <span className={`shrink-0 rounded-full border px-2 py-1 text-[8px] font-medium ${expert.asset_status === 'present' ? 'border-emerald-300/20 text-emerald-200' : 'border-amber-300/20 text-amber-200'}`}>{expert.asset_status.toUpperCase()}</span>
            </div>)}
            {!champion?.experts?.length && <div className="py-7 text-center text-xs text-slate-500">champion.pt 구조를 확인하는 중입니다.</div>}
          </div>
        </article>

        <article className="panel xl:col-span-7">
          <PanelTitle eyebrow="TORCHRL · POLICY / LEARNER" title="강화학습 정책" icon={<BrainCircuit size={16} className="text-violet-200" />} trailing={<StatusPill tone={training?.running ? 'violet' : 'idle'}>{training?.status?.toUpperCase() || 'IDLE'}</StatusPill>} />
          <p className="mt-2 max-w-2xl text-[11px] leading-5 text-slate-400">공유 actor와 EnvBase를 TorchRL Collector가 사용합니다. PPO는 learner baseline이며 정책 체크포인트는 champion.pt와 별도로 저장합니다.</p>
          <label className="mt-3 flex items-center justify-between rounded-lg border border-white/[0.06] bg-[#0a111d] px-3 py-2 text-[10px] text-slate-400">학습 알고리즘
            <select value={learner} disabled={training?.running} onChange={(event) => setLearner(event.target.value)} className="rounded-md border border-white/10 bg-[#111b2a] px-2 py-1 font-mono text-slate-200 outline-none">
              {(training?.available_algorithms?.length ? training.available_algorithms : ['PPO']).map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          <div className="mt-5 grid grid-cols-2 gap-2 md:grid-cols-4">
            <StatTile label="LEARNER" value={training?.algorithm || 'PPO'} />
            <StatTile label="ROLLOUT" value="32 bars / batch" />
            <StatTile label="DEVICE" value={training?.device || '—'} />
            <StatTile label="POLICY SAVED TO" value="runtime/policy_torchrl.pt" mono />
          </div>
          <div className="mt-4 rounded-xl border border-violet-300/10 bg-violet-300/[0.035] p-4">
            <div className="flex items-center justify-between text-[10px]"><span className="text-slate-400">Collector transitions</span><span className="font-mono text-violet-100">{fmt(training?.steps || 0, 0)}</span></div>
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/[0.06]"><div className="h-full rounded-full bg-violet-300 transition-all" style={{ width: `${Math.min(100, (training?.steps || 0) % 32 / 32 * 100)}%` }} /></div>
            <div className="mt-3 flex items-center justify-between text-[10px] text-slate-500"><span>최근 정책 업데이트 <b className="ml-1 font-mono text-slate-200">{fmt(training?.updates || 0, 0)}</b></span><span>loss <b className="ml-1 font-mono text-slate-300">{training?.last_loss == null ? '—' : fmt(training.last_loss, 5)}</b></span></div>
          </div>
          {training?.error && <div className="mt-3 rounded-lg border border-rose-300/20 bg-rose-300/[0.05] p-3 text-[10px] text-rose-200">{training.error}</div>}
          <div className="mt-4 flex flex-wrap gap-2">
            {training?.running
              ? <ActionButton tone="violet" disabled={busyAction} onClick={() => void perform('train-stop', api.stopTraining, 'Collector 중지 요청을 보냈습니다.')} icon={<Pause size={14} />}>학습 중지</ActionButton>
              : <ActionButton tone="violet" disabled={busyAction || !feed?.running} onClick={() => void perform('train', () => api.startTraining(learner), `TorchRL ${learner} learner를 시작했습니다.`)} icon={<Play size={14} />}>{learner} 실시간 학습 시작</ActionButton>}
            {paper?.enabled
              ? <ActionButton tone="neutral" disabled={busyAction} onClick={() => void perform('paper-stop', api.stopPaper, 'Paper 거래를 중지했습니다.')} icon={<Pause size={14} />}>Paper 중지</ActionButton>
              : <ActionButton tone="neutral" disabled={busyAction || !feed?.running || (training?.updates || 0) < 1} onClick={() => void perform('paper', api.startPaper, '같은 TorchRL actor의 결정론적 Paper 추론을 시작했습니다.')} icon={<Wallet size={14} />}>Paper 시작</ActionButton>}
          </div>
          {paper?.error && <div className="mt-3 text-[10px] text-rose-200">{paper.error}</div>}
          {state?.inference?.error && <div className="mt-3 text-[10px] text-rose-200">실시간 추론 오류: {state.inference.error}</div>}
        </article>

        <article className="panel xl:col-span-7">
          <PanelTitle eyebrow="FINRL-TRADING · BACKTEST" title="과거 구간 검증" icon={<BarChart3 size={16} className="text-cyan-200" />} trailing={<span className="rounded-full border border-cyan-300/15 px-2 py-1 text-[8px] tracking-[0.1em] text-cyan-100">CAUSAL WEIGHTS</span>} />
          <form onSubmit={(event) => void runBacktest(event)} className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <label className="field col-span-2">종목<input value={symbolsInput} onChange={(event) => setSymbolsInput(event.target.value)} /></label>
            <label className="field">시작일<input type="date" value={backtestRange.start} onChange={(event) => setBacktestRange((old) => ({ ...old, start: event.target.value }))} required /></label>
            <label className="field">종료일<input type="date" value={backtestRange.end} onChange={(event) => setBacktestRange((old) => ({ ...old, end: event.target.value }))} required /></label>
            <label className="field col-span-2 sm:col-span-1">초기 자금<input type="number" min={100000} step={100000} value={capital} onChange={(event) => setCapital(Number(event.target.value))} required /></label>
            <div className="col-span-2 flex items-end sm:col-span-3"><button type="submit" disabled={busyAction} className="flex h-10 w-full items-center justify-center gap-2 rounded-lg bg-cyan-200 px-4 text-[11px] font-semibold text-slate-950 transition hover:bg-cyan-100 disabled:opacity-50 sm:w-auto">{busy === 'backtest' ? <LoaderCircle size={14} className="animate-spin" /> : <Play size={14} />}{busy === 'backtest' ? '데이터 수집 및 검증 중…' : 'FinRL-X 백테스트 실행'}</button></div>
          </form>
          {backtest ? <BacktestSummary result={backtest} /> : <div className="mt-4 rounded-lg border border-dashed border-white/10 px-4 py-5 text-center text-[10px] text-slate-500">호환되는 TorchRL 정책 체크포인트를 같은 deterministic actor로 FinRL-Trading 백테스트에 연결합니다.</div>}
        </article>

        <article className="panel xl:col-span-5">
          <PanelTitle eyebrow="TARGET WEIGHTS · SINGLE CONTRACT" title="정책 배분" icon={<Gauge size={16} className="text-emerald-200" />} trailing={<StatusPill tone={state?.target_weights ? 'good' : 'idle'}>{state?.target_weights ? 'UPDATED' : 'WAITING'}</StatusPill>} />
          <div className="mt-2 text-[10px] text-slate-500">{state?.target_weights?.as_of ? `기준 시각 ${formatTime(state.target_weights.as_of)}` : '정책 결정 대기'}</div>
          <div className="mt-4 space-y-2">
            {Object.entries(state?.target_weights?.weights || {}).map(([symbol, weight]) => <WeightBar key={symbol} symbol={symbol} weight={weight} />)}
            {state?.target_weights && <div className="flex items-center justify-between border-t border-white/[0.05] px-1 pt-2 text-[10px]"><span className="text-slate-500">현금 잔여비중</span><span className="font-mono text-slate-300">{fmt(state.target_weights.cash_weight * 100, 1)}%</span></div>}
            {!state?.target_weights && <div className="rounded-lg border border-dashed border-white/10 px-3 py-8 text-center text-[10px] text-slate-500">Paper 추론을 시작하면 다음 실행 가능 bar에 목표 비중을 적용합니다.</div>}
          </div>
          <div className="mt-4 flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2.5 text-[10px]"><span className="text-slate-500">정책 입력</span><span className="font-mono text-slate-300">{state?.target_weights?.observation_source || '시장 특징 + 사용 가능한 전문가 신호'}</span></div>
        </article>

        <article className="panel xl:col-span-12">
          <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
            <PanelTitle eyebrow="PAPER LEDGER" title="가상 계좌와 체결" icon={<Wallet size={16} className="text-emerald-200" />} trailing={null} />
            <div className="flex flex-wrap items-center gap-2"><StatusPill tone={paper?.enabled && state?.inference?.running ? 'good' : 'idle'}>{paper?.enabled ? (state?.inference?.running ? 'PAPER ACTIVE' : 'INFERENCE STOPPED') : 'PAPER PAUSED'}</StatusPill><StatusPill tone={paperExecution?.mode === 'alpaca_paper' && paperExecution.available ? 'good' : 'idle'}>{paperExecution?.mode === 'alpaca_paper' ? (paperExecution.available ? 'FINRL ALPACA PAPER' : 'ALPACA PAPER UNAVAILABLE') : 'LOCAL PAPER'}</StatusPill><span className="text-[9px] text-slate-500">fee 0.10%</span></div>
            {paperExecution?.last_error && <div className="mt-2 text-[10px] text-rose-200">{paperExecution.last_error}</div>}
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <StatTile label="EQUITY" value={paper ? fmt(paper.equity, 0) : '—'} />
            <StatTile label="CASH" value={paper ? fmt(paper.cash, 0) : '—'} />
            <StatTile label="P&L" value={paper ? `${paper.return_pct >= 0 ? '+' : ''}${fmt(paper.return_pct, 2)}%` : '—'} accent={paper && paper.return_pct >= 0} />
            <StatTile label="OPEN POSITIONS" value={String(Object.keys(paper?.positions || {}).length)} />
          </div>
          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            <div><div className="mb-2 text-[9px] tracking-[0.14em] text-slate-500">POSITIONS</div><div className="overflow-hidden rounded-lg border border-white/[0.06]">
              {Object.entries(paper?.positions || {}).length ? Object.entries(paper?.positions || {}).map(([symbol, value]) => <div key={symbol} className="flex items-center justify-between border-b border-white/[0.04] px-3 py-2.5 last:border-0"><span className="text-xs font-medium">{symbol}</span><span className="font-mono text-[11px] text-slate-300">{fmt(value, 0)}</span></div>) : <div className="px-3 py-6 text-center text-[10px] text-slate-500">보유 종목 없음</div>}
            </div></div>
            <div><div className="mb-2 text-[9px] tracking-[0.14em] text-slate-500">RECENT FILLS</div><div className="max-h-[190px] overflow-auto rounded-lg border border-white/[0.06]">
              {(paper?.fills || []).slice().reverse().slice(0, 8).map((fill, index) => <div key={`${fill.timestamp}-${fill.symbol}-${index}`} className="grid grid-cols-[1fr_auto_auto] gap-3 border-b border-white/[0.04] px-3 py-2.5 last:border-0"><span><b className="text-[10px]">{fill.symbol}</b><small className="ml-2 text-[9px] text-slate-500">{formatTime(fill.timestamp)}</small></span><span className={`text-[9px] font-semibold ${fill.side === 'BUY' ? 'text-emerald-300' : 'text-rose-300'}`}>{fill.side}</span><span className="text-right font-mono text-[10px]">{fmt(fill.notional, 0)}</span></div>)}
              {!paper?.fills?.length && <div className="px-3 py-6 text-center text-[10px] text-slate-500">체결 내역 없음</div>}
            </div></div>
          </div>
        </article>
      </section>

      <footer className="mt-7 flex flex-col justify-between gap-2 border-t border-white/[0.06] pt-4 text-[9px] text-slate-600 sm:flex-row"><span>LOCAL ONLY · 127.0.0.1 · FinRL-X API v1</span><span className="inline-flex items-center gap-1"><LockKeyhole size={11} /> LIVE ORDER GATE: {state?.live_gate.status?.toUpperCase() || 'CLOSED'} · {state?.live_gate.reason || 'broker adapter not configured'}</span></footer>
    </main>
  </div>
}

function MetricCard({ icon, label, value, detail, tone }: { icon: ReactNode; label: string; value: string; detail: string; tone: string }) {
  const colors = tone === 'good' ? 'text-emerald-200' : tone === 'violet' ? 'text-violet-200' : tone === 'alert' ? 'text-amber-200' : 'text-slate-200'
  return <article className="rounded-xl border border-white/[0.07] bg-[#0d1522] p-4 sm:p-5"><div className="flex items-center justify-between"><span className="text-[9px] tracking-[0.13em] text-slate-500">{label}</span><span className="text-slate-500">{icon}</span></div><div className={`mt-3 truncate text-[15px] font-semibold tracking-wide ${colors}`}>{value}</div><div className="mt-1 truncate font-mono text-[9px] text-slate-500">{detail}</div></article>
}

function PanelTitle({ eyebrow, title, icon, trailing }: { eyebrow: string; title: string; icon?: ReactNode; trailing?: ReactNode | null }) {
  return <div className="flex items-start justify-between gap-3"><div><div className="eyebrow">{eyebrow}</div><h2 className="mt-1.5 flex items-center gap-2 text-[15px] font-semibold tracking-[-0.02em]">{icon}{title}</h2></div>{trailing}</div>
}

function StatusPill({ children, tone }: { children: ReactNode; tone: string }) {
  const colors = tone === 'good' ? 'border-emerald-300/20 bg-emerald-300/[0.06] text-emerald-200' : tone === 'violet' ? 'border-violet-300/20 bg-violet-300/[0.06] text-violet-200' : 'border-white/10 bg-white/[0.025] text-slate-400'
  return <span className={`inline-flex shrink-0 items-center rounded-full border px-2 py-1 text-[8px] font-semibold tracking-[0.1em] ${colors}`}>{children}</span>
}

function ActionButton({ children, onClick, disabled, icon, tone }: { children: ReactNode; onClick: () => void; disabled?: boolean; icon?: ReactNode; tone: 'primary' | 'violet' | 'neutral' }) {
  const colors = tone === 'primary' ? 'bg-cyan-200 text-slate-950 hover:bg-cyan-100' : tone === 'violet' ? 'border border-violet-300/15 bg-violet-300/[0.09] text-violet-100 hover:bg-violet-300/[0.15]' : 'border border-white/10 bg-white/[0.035] text-slate-200 hover:bg-white/[0.08]'
  return <button disabled={disabled} onClick={onClick} className={`inline-flex h-9 items-center justify-center gap-2 rounded-lg px-3 text-[10px] font-semibold transition disabled:cursor-not-allowed disabled:opacity-45 ${colors}`}>{icon}{children}</button>
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return <div className="rounded-lg bg-white/[0.025] px-3 py-2"><div className="text-[8px] tracking-[0.12em] text-slate-500">{label}</div><div className="mt-1 font-mono text-[10px] text-slate-200">{value}</div></div>
}

function StatTile({ label, value, mono, accent }: { label: string; value: string; mono?: boolean; accent?: boolean }) {
  return <div className="rounded-lg border border-white/[0.06] bg-[#0a111d] px-3 py-2.5"><div className="text-[8px] tracking-[0.12em] text-slate-500">{label}</div><div className={`mt-1.5 truncate text-[11px] font-semibold ${mono ? 'font-mono' : ''} ${accent ? 'text-emerald-200' : 'text-slate-200'}`}>{value}</div></div>
}

function MarketChart({ bars }: { bars: Bar[] }) {
  const data = bars.slice(-160)
  if (data.length < 2) return <div className="grid h-full place-items-center text-[10px] text-slate-600">1분 bar 데이터가 쌓이면 차트를 그립니다.</div>
  const width = 1000, height = 280, padX = 12, padY = 20
  const prices = data.map((bar) => bar.close)
  const min = Math.min(...prices), max = Math.max(...prices)
  const span = max - min || Math.max(max * 0.001, 0.01)
  const point = (value: number, index: number) => `${padX + index * (width - 2 * padX) / (prices.length - 1)},${height - padY - (value - min) * (height - 2 * padY) / span}`
  const line = prices.map(point).join(' ')
  const firstX = padX, lastX = width - padX, baseline = height - padY
  const path = `M ${line} L ${lastX},${baseline} L ${firstX},${baseline} Z`
  const last = prices.at(-1) || 0
  const up = last >= prices[0]
  return <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full" preserveAspectRatio="none" role="img" aria-label="최근 주가 차트">
    <defs><linearGradient id="market-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor={up ? '#69dfb1' : '#fb7185'} stopOpacity=".19" /><stop offset="100%" stopColor={up ? '#69dfb1' : '#fb7185'} stopOpacity="0" /></linearGradient></defs>
    {[0.2, 0.4, 0.6, 0.8].map((ratio) => <line key={ratio} x1={padX} x2={width - padX} y1={height * ratio} y2={height * ratio} stroke="rgba(148,163,184,.09)" strokeDasharray="3 6" />)}
    <path d={path} fill="url(#market-fill)" />
    <polyline points={line} fill="none" stroke={up ? '#69dfb1' : '#fb7185'} strokeWidth="2.2" vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round" />
    <circle cx={lastX} cy={Number(point(last, prices.length - 1).split(',')[1])} r="4" fill={up ? '#69dfb1' : '#fb7185'} />
    <text x={padX + 2} y={14} fill="#728198" fontSize="10">{fmt(max, 2)}</text><text x={padX + 2} y={height - 4} fill="#728198" fontSize="10">{fmt(min, 2)}</text>
    <text x={padX} y={height - 1} fill="#55647a" fontSize="8">{formatTime(data[0].timestamp)}</text><text x={width - padX} y={height - 1} textAnchor="end" fill="#55647a" fontSize="8">{formatTime(data.at(-1)!.timestamp)}</text>
  </svg>
}

function WeightBar({ symbol, weight }: { symbol: string; weight: number }) {
  const width = `${Math.min(100, Math.max(0, weight) * 100)}%`
  return <div className="grid grid-cols-[58px_1fr_54px] items-center gap-2"><span className="font-mono text-[9px] text-slate-300">{symbol}</span><div className="h-1.5 overflow-hidden rounded-full bg-white/[0.06]"><div className="h-full rounded-full bg-cyan-200 transition-all" style={{ width }} /></div><span className="text-right font-mono text-[9px] text-slate-400">{fmt(weight * 100, 1)}%</span></div>
}

function BacktestSummary({ result }: { result: BacktestResult }) {
  const metrics = result.metrics || {}
  const returnValue = metrics.total_return
  const metric = (value: number | null | undefined, percent = false) => value == null ? '—' : percent ? `${fmt(value * 100, 2)}%` : fmt(value, 2)
  return <div className="mt-4 rounded-xl border border-cyan-300/10 bg-cyan-300/[0.025] p-3.5">
    <div className="flex flex-wrap items-center justify-between gap-2"><div className="text-[11px] font-semibold text-slate-200">{result.strategy}</div><div className="font-mono text-[9px] text-cyan-100">{result.source}</div></div>
    <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">{[
      ['TOTAL RETURN', metric(returnValue, true)], ['ANNUAL RETURN', metric(metrics.annual_return, true)],
      ['SHARPE', metric(metrics.sharpe_ratio)], ['MAX DRAWDOWN', metric(metrics.max_drawdown, true)],
    ].map(([label, value]) => <div key={label} className="rounded-lg border border-white/[0.05] bg-[#0a111d] px-2.5 py-2"><div className="text-[8px] tracking-[0.1em] text-slate-500">{label}</div><div className={`mt-1.5 font-mono text-[11px] ${label === 'TOTAL RETURN' && returnValue != null ? (returnValue >= 0 ? 'text-emerald-200' : 'text-rose-200') : 'text-slate-200'}`}>{value}</div></div>)}
    <div className="mt-2 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-white/[0.05] bg-[#0a111d] px-2.5 py-2 text-[9px] text-slate-400"><span>현금 잔여비중 {fmt((result.cash_weight || 0) * 100, 1)}%</span><span>신호 bar 종가 → 다음 이용 가능 종가 실행</span></div>
    </div>
    <div className="mt-2 text-[9px] text-slate-500">{result.dates.length} bars · 비용 0.10% · 목표 비중은 다음 bar에 적용</div>
  </div>
}

function defaultDateRange() {
  const end = new Date()
  const start = new Date(); start.setFullYear(start.getFullYear() - 3)
  return { start: start.toISOString().slice(0, 10), end: end.toISOString().slice(0, 10) }
}

function fmt(value: number, decimals = 2) {
  return Number.isFinite(Number(value)) ? Number(value).toLocaleString('ko-KR', { maximumFractionDigits: decimals }) : '—'
}

function formatTime(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}
