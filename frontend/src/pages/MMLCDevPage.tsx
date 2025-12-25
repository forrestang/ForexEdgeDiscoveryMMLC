import { useState, useCallback, useEffect } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import { api } from '@/lib/api'
import { DEFAULT_WORKING_DIRECTORY } from '@/lib/constants'
import { useMMLCDevSettings } from '@/hooks/usePersistedSettings'
import { Loader2, Play, RotateCcw, ArrowLeft } from 'lucide-react'
import type { SessionType, TimeframeType, CandleData, WaveData } from '@/types'
import type { Data, Layout, Shape } from 'plotly.js'

interface MMLCDevPageProps {
  onBack: () => void
}

const SESSION_OPTIONS: SessionType[] = ['london', 'ny', 'asia', 'full_day']
const TIMEFRAME_OPTIONS: TimeframeType[] = ['M1', 'M5', 'M10', 'M15', 'M30', 'H1', 'H4']

export function MMLCDevPage({ onBack }: MMLCDevPageProps) {
  const [workingDirectory] = useState(DEFAULT_WORKING_DIRECTORY)
  const { settings: savedSettings, updateSettings: saveSettings } = useMMLCDevSettings()

  // Session settings - initialized from persisted settings
  const [pair, setPair] = useState<string>(savedSettings.pair)
  const [date, setDate] = useState<string>(savedSettings.date)
  const [session, setSession] = useState<SessionType>(savedSettings.session as SessionType)
  const [timeframe, setTimeframe] = useState<TimeframeType>(savedSettings.timeframe as TimeframeType)

  // Loaded data
  const [candles, setCandles] = useState<CandleData[]>([])
  const [waves, setWaves] = useState<WaveData[]>([])
  const [totalBars, setTotalBars] = useState(0)

  // Bar range - initialized from persisted settings
  const [startBar, setStartBar] = useState(savedSettings.startBar)
  const [endBar, setEndBar] = useState(savedSettings.endBar)

  // Display mode
  const [mode, setMode] = useState<'complete' | 'spline'>('complete')

  // Persist settings when they change
  useEffect(() => {
    saveSettings({ pair, date, session, timeframe, startBar, endBar })
  }, [pair, date, session, timeframe, startBar, endBar, saveSettings])

  // Fetch available pairs
  const { data: pairsData } = useQuery({
    queryKey: ['availablePairs', workingDirectory],
    queryFn: () => api.pipeline.getAvailablePairs(workingDirectory),
  })

  // Fetch available dates for selected pair
  const { data: datesData } = useQuery({
    queryKey: ['dates', pair, workingDirectory],
    queryFn: () => api.instruments.getDates(pair, workingDirectory),
    enabled: !!pair,
  })

  // Load session mutation
  const loadSession = useMutation({
    mutationFn: () => api.mmlcDev.loadSession({
      pair,
      date,
      session,
      timeframe,
      workingDirectory,
    }),
    onSuccess: (data) => {
      setCandles(data.candles)
      setTotalBars(data.total_bars)
      setStartBar(0)
      setEndBar(data.total_bars - 1)
      setWaves([]) // Clear waves when loading new session
    },
  })

  // Run dev engine mutation
  const runEngine = useMutation({
    mutationFn: () => api.mmlcDev.run({
      pair,
      date,
      session,
      timeframe,
      startBar,
      endBar,
      mode,
      workingDirectory,
    }),
    onSuccess: (data) => {
      setWaves(data.waves)
    },
  })

  const handleLoad = useCallback(() => {
    if (pair && date) {
      loadSession.mutate()
    }
  }, [pair, date, loadSession])

  const handleRun = useCallback(() => {
    runEngine.mutate()
  }, [runEngine])

  const handleReset = useCallback(() => {
    setWaves([])
  }, [])

  // Auto-run when bar range or mode changes (only if session is loaded)
  useEffect(() => {
    if (candles.length > 0 && !runEngine.isPending) {
      runEngine.mutate()
    }
  }, [startBar, endBar, mode])

  // Prepare candlestick trace
  const candlestickTrace: Data = candles.length > 0 ? {
    type: 'candlestick',
    x: candles.map((c) => c.timestamp),
    open: candles.map((c) => c.open),
    high: candles.map((c) => c.high),
    low: candles.map((c) => c.low),
    close: candles.map((c) => c.close),
    increasing: { line: { color: '#26a69a' }, fillcolor: '#26a69a' },
    decreasing: { line: { color: '#ef5350' }, fillcolor: '#ef5350' },
    name: 'Price',
    showlegend: false,
  } : { type: 'scatter', x: [], y: [], mode: 'lines' }

  // Prepare waveform traces
  const waveformTraces: Data[] = waves.map((wave) => ({
    type: 'scatter',
    mode: 'lines',
    x: [wave.start_time, wave.end_time],
    y: [wave.start_price, wave.end_price],
    line: {
      color: wave.color,
      width: wave.is_spline ? 1.5 : Math.max(4 - wave.level * 0.5, 1.5),
      dash: wave.is_spline ? 'dot' : 'solid',
    },
    name: wave.is_spline ? 'Spline' : `L${wave.level}`,
    hoverinfo: 'text',
    hovertext: wave.is_spline ? `Spline (${wave.direction})` : `Level ${wave.level} (${wave.direction})`,
    showlegend: false,
  }))

  // Bar range indicators
  const rangeShapes: Partial<Shape>[] = []
  if (candles.length > 0 && startBar >= 0 && startBar < candles.length) {
    rangeShapes.push({
      type: 'line',
      x0: candles[startBar].timestamp,
      x1: candles[startBar].timestamp,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: { color: 'rgba(0, 255, 0, 0.5)', width: 2, dash: 'dot' },
    })
  }
  if (candles.length > 0 && endBar >= 0 && endBar < candles.length) {
    rangeShapes.push({
      type: 'line',
      x0: candles[endBar].timestamp,
      x1: candles[endBar].timestamp,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: { color: 'rgba(255, 0, 0, 0.5)', width: 2, dash: 'dot' },
    })
  }

  const layout: Partial<Layout> = {
    title: {
      text: candles.length > 0 ? `MMLC Dev - ${pair} - ${date} (${session}) - ${timeframe}` : 'MMLC Development Sandbox',
      font: { color: '#e5e7eb', size: 14 },
    },
    dragmode: 'zoom',
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'hsl(222.2, 84%, 4.9%)',
    xaxis: {
      type: 'date',
      rangeslider: { visible: false },
      gridcolor: 'hsl(217.2, 32.6%, 17.5%)',
      linecolor: 'hsl(217.2, 32.6%, 17.5%)',
      tickfont: { color: '#9ca3af' },
    },
    yaxis: {
      autorange: true,
      gridcolor: 'hsl(217.2, 32.6%, 17.5%)',
      linecolor: 'hsl(217.2, 32.6%, 17.5%)',
      tickfont: { color: '#9ca3af' },
      side: 'right',
    },
    margin: { l: 50, r: 60, t: 50, b: 50 },
    showlegend: false,
    hovermode: 'x unified',
    shapes: rangeShapes,
  }

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'] as const,
    responsive: true,
  }

  const allTraces: Data[] = [candlestickTrace, ...waveformTraces]

  const pairs = pairsData?.pairs || []
  const dates = datesData?.dates || []

  return (
    <div className="h-screen flex flex-col bg-background text-foreground">
      {/* Header */}
      <div className="flex items-center gap-4 px-4 py-2 border-b border-border">
        <button
          onClick={onBack}
          className="flex items-center gap-1 px-2 py-1 rounded text-sm hover:bg-muted"
        >
          <ArrowLeft className="h-4 w-4" />
          Back
        </button>
        <h1 className="text-lg font-semibold">MMLC Development Sandbox</h1>
      </div>

      <div className="flex-1 flex min-h-0">
        {/* Left sidebar - Controls */}
        <div className="w-64 border-r border-border p-4 flex flex-col gap-4 overflow-y-auto">
          {/* Session Loader */}
          <div className="space-y-2">
            <h3 className="font-medium text-sm">Session</h3>

            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">Pair</label>
              <select
                value={pair}
                onChange={(e) => {
                  setPair(e.target.value)
                  setDate('')
                }}
                className="w-full px-2 py-1 rounded bg-muted border border-border text-sm"
              >
                <option value="">Select pair...</option>
                {pairs.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>

            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">Date</label>
              <select
                value={date}
                onChange={(e) => setDate(e.target.value)}
                className="w-full px-2 py-1 rounded bg-muted border border-border text-sm"
                disabled={!pair}
              >
                <option value="">Select date...</option>
                {dates.map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </div>

            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">Session</label>
              <select
                value={session}
                onChange={(e) => setSession(e.target.value as SessionType)}
                className="w-full px-2 py-1 rounded bg-muted border border-border text-sm"
              >
                {SESSION_OPTIONS.map((s) => (
                  <option key={s} value={s}>{s.replace('_', ' ')}</option>
                ))}
              </select>
            </div>

            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">Timeframe</label>
              <select
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value as TimeframeType)}
                className="w-full px-2 py-1 rounded bg-muted border border-border text-sm"
              >
                {TIMEFRAME_OPTIONS.map((tf) => (
                  <option key={tf} value={tf}>{tf}</option>
                ))}
              </select>
            </div>

            <button
              onClick={handleLoad}
              disabled={!pair || !date || loadSession.isPending}
              className="w-full px-3 py-1.5 rounded bg-primary text-primary-foreground text-sm font-medium disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {loadSession.isPending && <Loader2 className="h-3 w-3 animate-spin" />}
              Load Session
            </button>
          </div>

          {/* Bar Range Controls */}
          {totalBars > 0 && (
            <div className="space-y-2 border-t border-border pt-4">
              <h3 className="font-medium text-sm">Bar Range</h3>

              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Start</label>
                  <input
                    type="number"
                    min={0}
                    max={totalBars - 1}
                    value={startBar}
                    onChange={(e) => setStartBar(Math.max(0, Math.min(totalBars - 1, parseInt(e.target.value) || 0)))}
                    className="w-full px-2 py-1 rounded bg-muted border border-border text-sm"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">End</label>
                  <input
                    type="number"
                    min={0}
                    max={totalBars - 1}
                    value={endBar}
                    onChange={(e) => setEndBar(Math.max(0, Math.min(totalBars - 1, parseInt(e.target.value) || 0)))}
                    className="w-full px-2 py-1 rounded bg-muted border border-border text-sm"
                  />
                </div>
              </div>

              <p className="text-xs text-muted-foreground">
                Total bars: {totalBars}
              </p>

              {/* Display Mode Buttons */}
              <div className="flex gap-1">
                <button
                  onClick={() => setMode('complete')}
                  className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                    mode === 'complete'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted text-muted-foreground hover:bg-muted/80'
                  }`}
                >
                  Complete
                </button>
                <button
                  onClick={() => setMode('spline')}
                  className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                    mode === 'spline'
                      ? 'bg-cyan-600 text-white'
                      : 'bg-muted text-muted-foreground hover:bg-muted/80'
                  }`}
                >
                  Spline
                </button>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={handleRun}
                  disabled={runEngine.isPending}
                  className="flex-1 px-3 py-1.5 rounded bg-green-600 text-white text-sm font-medium disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {runEngine.isPending ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : (
                    <Play className="h-3 w-3" />
                  )}
                  Run
                </button>
                <button
                  onClick={handleReset}
                  className="px-3 py-1.5 rounded bg-muted text-muted-foreground text-sm font-medium flex items-center justify-center gap-2 hover:bg-muted/80"
                >
                  <RotateCcw className="h-3 w-3" />
                  Reset
                </button>
              </div>
            </div>
          )}

          {/* Wave State */}
          {waves.length > 0 && (
            <div className="space-y-2 border-t border-border pt-4">
              <h3 className="font-medium text-sm">Wave State</h3>
              <div className="space-y-1 text-xs">
                {waves.map((wave) => (
                  <div
                    key={wave.id}
                    className="flex items-center gap-2 px-2 py-1 rounded bg-muted/50"
                  >
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: wave.color }}
                    />
                    <span>L{wave.level}</span>
                    <span className={wave.direction === 'up' ? 'text-green-400' : 'text-red-400'}>
                      {wave.direction === 'up' ? '↑' : '↓'}
                    </span>
                    <span className="text-muted-foreground">
                      {wave.start_price.toFixed(5)} → {wave.end_price.toFixed(5)}
                    </span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                Total waves: {waves.length}
              </p>
            </div>
          )}

          {/* Status messages */}
          {loadSession.error && (
            <p className="text-xs text-red-400">
              Load error: {loadSession.error.message}
            </p>
          )}
          {runEngine.error && (
            <p className="text-xs text-red-400">
              Run error: {runEngine.error.message}
            </p>
          )}
        </div>

        {/* Chart area */}
        <div className="flex-1 min-w-0">
          {candles.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-muted-foreground">
              <p className="text-lg">Load a session to begin</p>
              <p className="text-sm">Select pair, date, session, and timeframe, then click Load</p>
            </div>
          ) : (
            <Plot
              data={allTraces}
              layout={layout}
              config={config}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          )}
        </div>
      </div>
    </div>
  )
}
