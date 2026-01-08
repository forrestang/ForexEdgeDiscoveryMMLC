import { useState, useCallback, useEffect, useRef } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import { api } from '@/lib/api'
import { DEFAULT_WORKING_DIRECTORY } from '@/lib/constants'
import { useMMLCDevSettings } from '@/hooks/usePersistedSettings'
import { Loader2, Play, RotateCcw, Bug, BarChart3 } from 'lucide-react'
import type { SessionType, TimeframeType, CandleData, WaveData, StitchAnnotation, SwingLabel, DebugState } from '@/types'
import type Plotly from 'plotly.js'
import type { Data, Layout, Shape, Annotations } from 'plotly.js'

const SESSION_OPTIONS: SessionType[] = ['london', 'ny', 'asia', 'full_day']
const TIMEFRAME_OPTIONS: TimeframeType[] = ['M1', 'M5', 'M10', 'M15', 'M30', 'H1', 'H4']

export function MMLCDevPage() {
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
  const [annotations, setAnnotations] = useState<StitchAnnotation[]>([])
  const [swingLabels, setSwingLabels] = useState<SwingLabel[]>([])
  const [totalBars, setTotalBars] = useState(0)

  // Bar range - initialized from persisted settings
  const [startBar, setStartBar] = useState(savedSettings.startBar)
  const [endBar, setEndBar] = useState(savedSettings.endBar)

  // Display mode
  const [mode, setMode] = useState<'complete' | 'spline' | 'stitch'>('complete')
  const [showWaveform, setShowWaveform] = useState(true)

  // Debug panel
  const [debugState, setDebugState] = useState<DebugState | null>(null)
  const [debugWindowRef, setDebugWindowRef] = useState<Window | null>(null)
  const [statsWindowRef, setStatsWindowRef] = useState<Window | null>(null)

  // Clicked bar data box
  const [clickedBarInfo, setClickedBarInfo] = useState<{
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    barIndex: number;
  } | null>(null)

  // Hover position for axis labels
  const [hoverInfo, setHoverInfo] = useState<{
    price: number;
    time: string;
    yPixel: number;
    xPixel: number;
    barIndex: number;
  } | null>(null)
  const [showStatusBox, setShowStatusBox] = useState(false)
  const chartContainerRef = useRef<HTMLDivElement>(null)

  // Persist settings when they change
  useEffect(() => {
    saveSettings({ pair, date, session, timeframe, startBar, endBar })
  }, [pair, date, session, timeframe, startBar, endBar, saveSettings])

  // Save debug state to localStorage when it changes (for popup window sync)
  useEffect(() => {
    if (debugState) {
      localStorage.setItem('mmlc-debug-state', JSON.stringify(debugState))
    }
  }, [debugState])

  // Open debug panel in popup window
  const openDebugPopup = useCallback(() => {
    // Check if popup already exists and is open
    if (debugWindowRef && !debugWindowRef.closed) {
      debugWindowRef.focus()
      return
    }
    // Open new popup window
    const popup = window.open(
      '/?page=debug-panel',
      'mmlc-debug-panel',
      'width=500,height=800,left=100,top=100,resizable=yes,scrollbars=yes'
    )
    if (popup) {
      setDebugWindowRef(popup)
    }
  }, [debugWindowRef])

  // Open stats panel in popup window
  const openStatsPopup = useCallback(() => {
    // Check if popup already exists and is open
    if (statsWindowRef && !statsWindowRef.closed) {
      statsWindowRef.focus()
      return
    }
    // Open new popup window
    const popup = window.open(
      '/?page=stats-panel',
      'mmlc-stats-panel',
      'width=600,height=800,left=650,top=100,resizable=yes,scrollbars=yes'
    )
    if (popup) {
      setStatsWindowRef(popup)
    }
  }, [statsWindowRef])

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
      // Auto-run waveform after loading
      setTimeout(() => runEngine.mutate(), 100)
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
      setAnnotations(data.annotations || [])
      setSwingLabels(data.swing_labels || [])
      setDebugState(data.debug_state || null)
      // Sync mmlc_out to localStorage for debug panel
      console.log('[MMLC] API response mmlc_out:', data.mmlc_out?.length, 'snapshots')
      console.log('[MMLC] First snapshot swings:', data.mmlc_out?.[0]?.swings)
      if (data.mmlc_out) {
        localStorage.setItem('mmlc-out', JSON.stringify(data.mmlc_out))
        console.log('[MMLC] Stored mmlc_out to localStorage')
      }
      // Sync lstm_out to localStorage for debug panel
      if (data.lstm_out) {
        localStorage.setItem('lstm-out', JSON.stringify(data.lstm_out))
        console.log('[MMLC] Stored lstm_out to localStorage:', data.lstm_out?.length, 'payloads')
      }
      // Sync session metadata for stats panel
      localStorage.setItem('mmlc-session-meta', JSON.stringify({
        pair,
        date,
        session,
        timeframe,
        mode,
        totalBars: candles.length,
        availableDates: datesData?.dates || [],
      }))
    },
  })

  const handleRun = useCallback(() => {
    runEngine.mutate()
  }, [runEngine])

  const handleReset = useCallback(() => {
    setWaves([])
  }, [])

  // Handle chart click to show bar data
  const handleChartClick = useCallback((event: Readonly<Plotly.PlotMouseEvent>) => {
    if (event.points && event.points.length > 0) {
      const point = event.points[0]
      // Find the candle at this timestamp
      const timestamp = point.x as string
      const candleIndex = candles.findIndex(c => c.timestamp === timestamp)
      if (candleIndex >= 0) {
        const candle = candles[candleIndex]
        setClickedBarInfo({
          timestamp: candle.timestamp,
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
          barIndex: candleIndex,
        })
      }
    }
  }, [candles])

  // Set up mouse move listener on the plot for axis labels
  useEffect(() => {
    const container = chartContainerRef.current
    if (!container || candles.length === 0) return

    const handleMouseMove = (e: MouseEvent) => {
      // Find the plotly graph div inside the container
      const plotlyDiv = container.querySelector('.js-plotly-plot') as HTMLElement & {
        _fullLayout?: {
          xaxis: { range: [string, string]; _length: number; _offset: number };
          yaxis: { range: [number, number]; _length: number; _offset: number };
          margin: { l: number; r: number; t: number; b: number };
        };
      }

      if (!plotlyDiv?._fullLayout) return

      const layout = plotlyDiv._fullLayout
      const rect = container.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top

      // Check if mouse is within plot area
      const plotLeft = layout.margin.l
      const plotRight = rect.width - layout.margin.r
      const plotTop = layout.margin.t
      const plotBottom = rect.height - layout.margin.b

      if (x < plotLeft || x > plotRight || y < plotTop || y > plotBottom) {
        setHoverInfo(null)
        return
      }

      // Convert pixel to data coordinates
      const xRatio = (x - plotLeft) / (plotRight - plotLeft)
      const yRatio = 1 - (y - plotTop) / (plotBottom - plotTop)

      const xRange = layout.xaxis.range
      const yRange = layout.yaxis.range

      const xMin = new Date(xRange[0]).getTime()
      const xMax = new Date(xRange[1]).getTime()
      const time = new Date(xMin + xRatio * (xMax - xMin)).toISOString()

      const price = yRange[0] + yRatio * (yRange[1] - yRange[0])

      // Find the closest bar to the cursor
      const hoverTime = new Date(time).getTime()
      let closestBarIndex = -1
      let closestDistance = Infinity
      for (let i = 0; i < candles.length; i++) {
        const barTime = new Date(candles[i].timestamp).getTime()
        const distance = Math.abs(barTime - hoverTime)
        if (distance < closestDistance) {
          closestDistance = distance
          closestBarIndex = i
        }
      }

      setHoverInfo({
        price,
        time,
        xPixel: x,
        yPixel: y,
        barIndex: closestBarIndex,
      })
    }

    const handleMouseLeave = () => {
      setHoverInfo(null)
    }

    container.addEventListener('mousemove', handleMouseMove)
    container.addEventListener('mouseleave', handleMouseLeave)

    return () => {
      container.removeEventListener('mousemove', handleMouseMove)
      container.removeEventListener('mouseleave', handleMouseLeave)
    }
  }, [candles.length])

  // Auto-load session when date/session/timeframe changes
  useEffect(() => {
    if (pair && date && !loadSession.isPending) {
      loadSession.mutate()
    }
  }, [date, session, timeframe])

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
    hoverinfo: 'none',  // Hide hover popup but keep spike labels
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
    hoverinfo: 'none',  // Hide hover popup but keep spike labels
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

  // Swing labels disabled - keeping code for potential future use
  const plotlyAnnotations: Partial<Annotations>[] = []

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
      // Spike to axis with label
      showspikes: true,
      spikemode: 'toaxis',
      spikesnap: 'cursor',
      spikethickness: 1,
      spikecolor: 'rgba(255, 255, 255, 0.5)',
      spikedash: 'dot',
    },
    yaxis: {
      autorange: true,
      gridcolor: 'hsl(217.2, 32.6%, 17.5%)',
      linecolor: 'hsl(217.2, 32.6%, 17.5%)',
      tickfont: { color: '#9ca3af' },
      tickformat: '.5f',  // Always show 5 decimal places
      side: 'right',
      // Spike to axis with label
      showspikes: true,
      spikemode: 'toaxis',
      spikesnap: 'cursor',
      spikethickness: 1,
      spikecolor: 'rgba(255, 255, 255, 0.5)',
      spikedash: 'dot',
    },
    margin: { l: 50, r: 80, t: 50, b: 50 },  // Increased right margin for 5 decimals
    showlegend: false,
    hovermode: 'x' as const,  // Show spike labels on x-axis hover
    spikedistance: -1,  // Always show spikes regardless of distance
    hoverdistance: 50,  // Distance threshold for hover
    shapes: rangeShapes,
    annotations: plotlyAnnotations,
  }

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'] as const,
    responsive: true,
  }

  const allTraces: Data[] = showWaveform ? [candlestickTrace, ...waveformTraces] : [candlestickTrace]

  const pairs = pairsData?.pairs || []
  const dates = datesData?.dates || []

  return (
    <div className="h-full flex flex-col bg-background text-foreground">
      {/* Header */}
      <div className="sandbox-header">
        <span className="sandbox-title">Development Environment</span>
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
              <label className="text-xs text-muted-foreground">
                Date {date && `(${dates.indexOf(date) + 1}/${dates.length})`}
              </label>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => {
                    const idx = dates.indexOf(date)
                    if (idx > 0) setDate(dates[idx - 1])
                  }}
                  disabled={!pair || dates.indexOf(date) <= 0}
                  className="px-2 py-1 rounded bg-muted hover:bg-muted/80 text-sm font-bold disabled:opacity-30"
                >
                  −
                </button>
                <button
                  onClick={() => {
                    const idx = dates.indexOf(date)
                    if (idx >= 0 && idx < dates.length - 1) setDate(dates[idx + 1])
                  }}
                  disabled={!pair || dates.indexOf(date) >= dates.length - 1}
                  className="px-2 py-1 rounded bg-muted hover:bg-muted/80 text-sm font-bold disabled:opacity-30"
                >
                  +
                </button>
                <select
                  value={date}
                  onChange={(e) => setDate(e.target.value)}
                  className="flex-1 px-2 py-1 rounded bg-muted border border-border text-sm"
                  disabled={!pair}
                >
                  <option value="">Select date...</option>
                  {dates.map((d) => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
              </div>
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

            {loadSession.isPending && (
              <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" />
                Loading...
              </div>
            )}
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

              {/* Bar Position Slider */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setEndBar(Math.max(startBar, endBar - 1))}
                  className="px-2 py-1 rounded bg-muted hover:bg-muted/80 text-sm font-bold"
                >
                  −
                </button>
                <button
                  onClick={() => setEndBar(Math.min(totalBars - 1, endBar + 1))}
                  className="px-2 py-1 rounded bg-muted hover:bg-muted/80 text-sm font-bold"
                >
                  +
                </button>
                <input
                  type="range"
                  min={startBar}
                  max={totalBars - 1}
                  value={endBar}
                  onChange={(e) => setEndBar(parseInt(e.target.value))}
                  className="flex-1 h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                />
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
                <button
                  onClick={() => setMode('stitch')}
                  className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                    mode === 'stitch'
                      ? 'bg-purple-600 text-white'
                      : 'bg-muted text-muted-foreground hover:bg-muted/80'
                  }`}
                >
                  Stitch
                </button>
              </div>

              {/* Waveform Toggle */}
              <button
                onClick={() => setShowWaveform(!showWaveform)}
                className={`w-full px-2 py-1 rounded text-xs font-medium transition-colors ${
                  showWaveform
                    ? 'bg-yellow-600 text-white'
                    : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }`}
              >
                Waveform: {showWaveform ? 'ON' : 'OFF'}
              </button>

              {/* Status Box Toggle */}
              <button
                onClick={() => setShowStatusBox(!showStatusBox)}
                className={`w-full px-2 py-1 rounded text-xs font-medium transition-colors ${
                  showStatusBox
                    ? 'bg-cyan-600 text-white'
                    : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }`}
              >
                Status: {showStatusBox ? 'ON' : 'OFF'}
              </button>

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

              {/* Debug & Stats Buttons - Opens popup windows */}
              <div className="flex gap-2">
                <button
                  onClick={openDebugPopup}
                  disabled={!debugState}
                  className="flex-1 px-3 py-1.5 rounded bg-orange-600 text-white text-sm font-medium disabled:opacity-50 flex items-center justify-center gap-2 hover:bg-orange-700"
                >
                  <Bug className="h-3 w-3" />
                  Debug
                </button>
                <button
                  onClick={openStatsPopup}
                  className="flex-1 px-3 py-1.5 rounded bg-blue-600 text-white text-sm font-medium flex items-center justify-center gap-2 hover:bg-blue-700"
                >
                  <BarChart3 className="h-3 w-3" />
                  Stats
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
            <div className="relative h-full" ref={chartContainerRef}>
              <Plot
                data={allTraces}
                layout={layout}
                config={config}
                useResizeHandler
                style={{ width: '100%', height: '100%' }}
                onClick={handleChartClick}
              />
              {/* Horizontal crosshair line */}
              {hoverInfo && (
                <div
                  className="absolute pointer-events-none z-10"
                  style={{
                    left: 50,  // margin.l
                    right: 80, // margin.r
                    top: hoverInfo.yPixel,
                    height: 1,
                    backgroundColor: 'rgba(59, 130, 246, 0.5)',
                  }}
                />
              )}
              {/* Vertical crosshair line */}
              {hoverInfo && (
                <div
                  className="absolute pointer-events-none z-10"
                  style={{
                    left: hoverInfo.xPixel,
                    top: 50,   // margin.t
                    bottom: 50, // margin.b
                    width: 1,
                    backgroundColor: 'rgba(59, 130, 246, 0.5)',
                  }}
                />
              )}
              {/* Price label on Y-axis (right side) */}
              {hoverInfo && (
                <div
                  className="absolute bg-blue-600 text-white text-xs font-mono px-2 py-0.5 rounded-sm pointer-events-none z-20"
                  style={{
                    right: 4,
                    top: hoverInfo.yPixel,
                    transform: 'translateY(-50%)',
                  }}
                >
                  {hoverInfo.price.toFixed(5)}
                </div>
              )}
              {/* Time label on X-axis (bottom) */}
              {hoverInfo && (
                <div
                  className="absolute bg-blue-600 text-white text-xs font-mono px-2 py-0.5 rounded-sm pointer-events-none z-20"
                  style={{
                    left: hoverInfo.xPixel,
                    bottom: 8,
                    transform: 'translateX(-50%)',
                  }}
                >
                  {new Date(hoverInfo.time).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                </div>
              )}
              {/* Click data box - shows when a bar is clicked */}
              {clickedBarInfo && (
                <div className="absolute top-2 left-2 bg-black/80 text-white text-xs font-mono p-2 rounded border border-gray-600 z-10">
                  <div className="flex justify-between gap-4 mb-1">
                    <span className="text-gray-400">Bar {clickedBarInfo.barIndex}</span>
                    <button
                      onClick={() => setClickedBarInfo(null)}
                      className="text-gray-400 hover:text-white"
                    >
                      ×
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
                    <span className="text-gray-400">O:</span>
                    <span>{clickedBarInfo.open.toFixed(5)}</span>
                    <span className="text-gray-400">H:</span>
                    <span className="text-green-400">{clickedBarInfo.high.toFixed(5)}</span>
                    <span className="text-gray-400">L:</span>
                    <span className="text-red-400">{clickedBarInfo.low.toFixed(5)}</span>
                    <span className="text-gray-400">C:</span>
                    <span>{clickedBarInfo.close.toFixed(5)}</span>
                  </div>
                </div>
              )}
              {/* Status box - fixed position upper left, toggle controlled */}
              {showStatusBox && debugState && (
                <div className="absolute top-2 left-2 bg-black/95 text-white text-xs font-mono p-2 rounded border border-gray-600 z-30 min-w-[180px]">
                  <div className="text-gray-400 mb-1 border-b border-gray-600 pb-1">
                    Bar {debugState.end_bar}
                  </div>
                  {debugState.current_candle && (
                    <div className="grid grid-cols-4 gap-x-2 gap-y-0.5 mb-2">
                      <span className="text-gray-400">O</span>
                      <span className="text-gray-400">H</span>
                      <span className="text-gray-400">L</span>
                      <span className="text-gray-400">C</span>
                      <span>{debugState.current_candle.open.toFixed(5)}</span>
                      <span className="text-green-400">{debugState.current_candle.high.toFixed(5)}</span>
                      <span className="text-red-400">{debugState.current_candle.low.toFixed(5)}</span>
                      <span>{debugState.current_candle.close.toFixed(5)}</span>
                    </div>
                  )}
                  <div className="border-t border-gray-600 pt-1">
                    <div className="grid grid-cols-3 gap-x-2 gap-y-0.5">
                      <span className="text-gray-400"></span>
                      <span className="text-gray-400 text-center">High</span>
                      <span className="text-gray-400 text-center">Low</span>
                      {debugState.levels.map((level) => (
                        <div key={level.level} className="contents">
                          <span className="text-cyan-400">L{level.level}</span>
                          <span className="text-green-400 text-right">{level.high.toFixed(5)}</span>
                          <span className="text-red-400 text-right">{level.low.toFixed(5)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
