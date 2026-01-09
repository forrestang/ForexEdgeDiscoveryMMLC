import { useMemo, useState, useEffect, useRef } from 'react'
import Plot from 'react-plotly.js'
import { useParquetFiles, useParquetData, useParquetDates } from '@/hooks/useTransformer'
import { usePersistedState } from '@/hooks/usePersistedSettings'
import { Select } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import {
  ChevronLeft,
  ChevronRight,
  Loader2,
  FileSpreadsheet,
  ArrowUp,
  ArrowDown,
  Minus,
  Plus,
} from 'lucide-react'
import type { Data, Layout } from 'plotly.js'

interface ParquetViewerProps {
  workingDirectory: string
}

// Session toggle buttons with abbreviated labels
const SESSION_BUTTONS = [
  { value: 'asia', label: 'ASIA' },
  { value: 'lon', label: 'LON' },
  { value: 'ny', label: 'NY' },
  { value: 'day', label: 'FULL' },
  { value: 'asia_lon', label: 'ASIA+LON' },
  { value: 'lon_ny', label: 'LON+NY' },
]

// Session time boundaries in UTC hours
const SESSION_TIME_RANGES: Record<string, { startHour: number; endHour: number }> = {
  asia: { startHour: 0, endHour: 9 },
  lon: { startHour: 8, endHour: 17 },
  ny: { startHour: 13, endHour: 22 },
  day: { startHour: 0, endHour: 22 },
  asia_lon: { startHour: 0, endHour: 17 },
  lon_ny: { startHour: 8, endHour: 22 },
}

const VISIBLE_BARS = 100 // Number of bars to show at once

// Filter candles and states to only include bars within the session time window
interface CandleData {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
}

interface MmlcState {
  level: number | null
  event: string | null
  dir: string | null
  out_max_up: number | null
  out_max_down: number | null
}

function filterBySessionTime(
  candles: CandleData[],
  states: MmlcState[],
  session: string
): { filteredCandles: CandleData[]; filteredStates: MmlcState[] } {
  const timeRange = SESSION_TIME_RANGES[session]
  if (!timeRange) {
    return { filteredCandles: candles, filteredStates: states }
  }

  const filteredCandles: CandleData[] = []
  const filteredStates: MmlcState[] = []

  candles.forEach((candle, idx) => {
    // Extract hour and minute directly from timestamp string
    // Parquet timestamps are in UTC (converted from EST during parquet creation)
    // Timestamp format: "2022-01-26T13:00:00" or "2022-01-26 13:00:00"
    const timeMatch = candle.timestamp.match(/[T ](\d{2}):(\d{2})/)
    if (!timeMatch) return

    const hour = parseInt(timeMatch[1], 10)
    const minute = parseInt(timeMatch[2], 10)
    const timeValue = hour * 60 + minute

    const startValue = timeRange.startHour * 60
    const endValue = timeRange.endHour * 60

    if (timeValue >= startValue && timeValue < endValue) {
      filteredCandles.push(candle)
      if (states[idx]) {
        filteredStates.push(states[idx])
      }
    }
  })

  return { filteredCandles, filteredStates }
}

export function ParquetViewer({ workingDirectory }: ParquetViewerProps) {
  // Persisted state for LSTM ParquetViewer
  const [selectedFile, setSelectedFile] = usePersistedState<string | null>('lstmParquetViewerFile', null)
  const [selectedDate, setSelectedDate] = usePersistedState<string | null>('lstmParquetViewerDate', null)
  const [session, setSession] = usePersistedState('lstmParquetViewerSession', 'lon')
  const [currentBar, setCurrentBar] = usePersistedState('lstmParquetViewerBar', 0)

  // Ref for chart container (for mouse position tracking)
  const chartContainerRef = useRef<HTMLDivElement>(null)

  // Hover state for crosshair axis labels (pixel positions + data values)
  const [hoverInfo, setHoverInfo] = useState<{
    price: number
    time: string
    xPixel: number
    yPixel: number
  } | null>(null)

  // Fetch available parquet files
  const { data: filesData, isLoading: filesLoading } = useParquetFiles(workingDirectory)

  // Fetch available dates for selected file
  const { data: datesData, isLoading: datesLoading } = useParquetDates(selectedFile, workingDirectory)

  // Fetch ALL parquet data for the selected date (no backend pagination - we filter in frontend)
  const { data: parquetData, isLoading: dataLoading } = useParquetData(
    selectedFile,
    workingDirectory,
    session,
    0,      // Start from beginning
    10000,  // Fetch all bars (large limit)
    selectedDate
  )

  const files = filesData?.files || []
  const fileOptions = files.map((f) => ({
    value: f.path,
    label: `${f.name} (${f.rows.toLocaleString()} rows)`,
  }))

  // Filter data by session time
  const { filteredCandles, filteredStates, totalRows } = useMemo(() => {
    if (!parquetData?.candles?.length) {
      return { filteredCandles: [], filteredStates: [], totalRows: 0 }
    }
    const { filteredCandles, filteredStates } = filterBySessionTime(
      parquetData.candles as CandleData[],
      (parquetData.states || []) as MmlcState[],
      session
    )
    return { filteredCandles, filteredStates, totalRows: filteredCandles.length }
  }, [parquetData, session])

  // Compute visible window of bars centered around currentBar
  const visibleStartIdx = Math.max(0, currentBar - Math.floor(VISIBLE_BARS / 2))
  const visibleEndIdx = Math.min(totalRows, visibleStartIdx + VISIBLE_BARS)
  const visibleCandles = filteredCandles.slice(visibleStartIdx, visibleEndIdx)

  // Get the state and candle at the current bar position
  const currentState = useMemo(() => {
    if (!filteredStates.length) return null
    if (currentBar >= 0 && currentBar < filteredStates.length) {
      return filteredStates[currentBar]
    }
    return null
  }, [filteredStates, currentBar])

  const currentCandle = useMemo(() => {
    if (!filteredCandles.length) return null
    if (currentBar >= 0 && currentBar < filteredCandles.length) {
      return filteredCandles[currentBar]
    }
    return null
  }, [filteredCandles, currentBar])

  // Get dates list for navigation
  const datesList = useMemo(() => datesData?.dates || [], [datesData])

  // Handle file selection change
  const handleFileChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedFile(e.target.value || null)
    setSelectedDate(null) // Reset date when file changes
    setCurrentBar(0)
  }

  // Handle date selection change
  const handleDateChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedDate(e.target.value || null)
    setCurrentBar(0) // Reset to first bar when date changes
  }

  // Date navigation - prev/next buttons
  const handlePrevDate = () => {
    if (!datesList.length || !selectedDate) return
    const currentIdx = datesList.indexOf(selectedDate)
    if (currentIdx > 0) {
      setSelectedDate(datesList[currentIdx - 1])
      setCurrentBar(0)
    }
  }

  const handleNextDate = () => {
    if (!datesList.length || !selectedDate) return
    const currentIdx = datesList.indexOf(selectedDate)
    if (currentIdx < datesList.length - 1) {
      setSelectedDate(datesList[currentIdx + 1])
      setCurrentBar(0)
    }
  }

  // Build date options for dropdown
  const dateOptions = useMemo(() => {
    return [
      { value: '', label: 'All dates' },
      ...datesList.map((d) => ({ value: d, label: d })),
    ]
  }, [datesList])

  // Candlestick chart traces (using filtered & windowed candles)
  // Use raw timestamps directly like sandbox does
  const chartData: Data[] = useMemo(() => {
    if (!visibleCandles.length) return []

    const candlestickTrace: Data = {
      type: 'candlestick',
      x: visibleCandles.map((c) => c.timestamp),
      open: visibleCandles.map((c) => c.open),
      high: visibleCandles.map((c) => c.high),
      low: visibleCandles.map((c) => c.low),
      close: visibleCandles.map((c) => c.close),
      increasing: { line: { color: '#26a69a' }, fillcolor: '#26a69a' },
      decreasing: { line: { color: '#ef5350' }, fillcolor: '#ef5350' },
      name: 'Price',
      showlegend: false,
      hoverinfo: 'none',
    }

    return [candlestickTrace]
  }, [visibleCandles])

  // Set up mouse move listener on the plot for axis labels
  useEffect(() => {
    const container = chartContainerRef.current
    if (!container || !visibleCandles.length) return

    const handleMouseMove = (e: MouseEvent) => {
      const plotlyDiv = container.querySelector('.js-plotly-plot') as HTMLElement & {
        _fullLayout?: {
          xaxis: { range: [string, string]; _length: number; _offset: number }
          yaxis: { range: [number, number]; _length: number; _offset: number }
          margin: { l: number; r: number; t: number; b: number }
        }
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

      setHoverInfo({
        price,
        time,
        xPixel: x,
        yPixel: y,
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
  }, [visibleCandles.length])

  // Chart layout with current bar marker - matches sandbox style
  const chartLayout: Partial<Layout> = useMemo(() => {
    const shapes: Layout['shapes'] = []

    // Add vertical line at current bar
    if (currentCandle) {
      shapes.push({
        type: 'line',
        x0: currentCandle.timestamp,
        x1: currentCandle.timestamp,
        y0: 0,
        y1: 1,
        yref: 'paper',
        line: { color: '#f59e0b', width: 2, dash: 'dot' },
      })
    }

    return {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'hsl(222.2, 84%, 4.9%)',
      margin: { l: 50, r: 80, t: 20, b: 50 },
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
        tickformat: '.5f',
        side: 'right',
      },
      shapes,
      dragmode: 'zoom',
      showlegend: false,
      hovermode: false,
    }
  }, [currentCandle])

  return (
    <div className="flex flex-col h-full bg-card/50 rounded-lg border border-border/50">
      {/* Header Controls */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50 gap-6">
        {/* File dropdown section */}
        <div className="flex-1 flex items-center gap-2">
          <FileSpreadsheet className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <Select
            value={selectedFile || ''}
            onChange={handleFileChange}
            options={[{ value: '', label: 'Select parquet...' }, ...fileOptions]}
            className="text-xs h-8 bg-secondary/30 border-border/50 w-full"
            disabled={filesLoading}
          />
        </div>

        {/* Date dropdown with +/- navigation buttons */}
        <div className="flex-1 flex items-center justify-center gap-1">
          {selectedFile ? (
            <>
              <Button
                size="sm"
                variant="outline"
                className="h-8 w-8 p-0"
                onClick={handlePrevDate}
                disabled={datesLoading || !selectedDate || datesList.indexOf(selectedDate) === 0}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Select
                value={selectedDate || ''}
                onChange={handleDateChange}
                options={dateOptions}
                className="text-xs h-8 w-32 bg-secondary/30 border-border/50"
                disabled={datesLoading}
              />
              <Button
                size="sm"
                variant="outline"
                className="h-8 w-8 p-0"
                onClick={handleNextDate}
                disabled={datesLoading || !selectedDate || datesList.indexOf(selectedDate) === datesList.length - 1}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </>
          ) : (
            <span className="text-xs text-muted-foreground">—</span>
          )}
        </div>

        {/* Playhead slider with +/- buttons */}
        <div className="flex-1 flex items-center justify-center gap-2">
          {selectedFile && totalRows > 0 ? (
            <>
              <input
                type="range"
                min={0}
                max={Math.max(0, totalRows - 1)}
                value={currentBar}
                onChange={(e) => setCurrentBar(parseInt(e.target.value))}
                className="w-28 h-1.5 bg-secondary/50 rounded-lg appearance-none cursor-pointer accent-primary"
              />
              <Button
                size="sm"
                variant="outline"
                className="h-6 w-6 p-0"
                onClick={() => setCurrentBar(Math.max(0, currentBar - 1))}
                disabled={currentBar === 0}
              >
                <Minus className="h-3 w-3" />
              </Button>
              <Button
                size="sm"
                variant="outline"
                className="h-6 w-6 p-0"
                onClick={() => setCurrentBar(Math.min(totalRows - 1, currentBar + 1))}
                disabled={currentBar >= totalRows - 1}
              >
                <Plus className="h-3 w-3" />
              </Button>
            </>
          ) : (
            <span className="text-xs text-muted-foreground">—</span>
          )}
        </div>

        {/* Session toggle buttons */}
        <div className="flex-1 flex items-center justify-end gap-1.5">
          {SESSION_BUTTONS.map((s) => (
            <Button
              key={s.value}
              size="sm"
              variant={session === s.value ? 'default' : 'outline'}
              className={`h-7 px-2 text-xs font-mono ${
                session === s.value
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary/30 hover:bg-secondary/50'
              }`}
              onClick={() => {
                setSession(s.value)
                setCurrentBar(0)
              }}
            >
              {s.label}
            </Button>
          ))}
        </div>
      </div>

      {/* Chart Area */}
      <div className="flex-1 min-h-0 relative">
        {dataLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/50 z-10">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        )}

        {!selectedFile && (
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <FileSpreadsheet className="h-12 w-12 mx-auto mb-2 opacity-30" />
              <p className="text-sm">Select a parquet file to view</p>
            </div>
          </div>
        )}

        {selectedFile && chartData.length > 0 && (
          <div ref={chartContainerRef} className="w-full h-full relative">
            <Plot
              data={chartData}
              layout={chartLayout}
              config={{
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'] as const,
                scrollZoom: true,
                responsive: true,
              }}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler
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
                  backgroundColor: 'rgba(255, 255, 255, 0.5)',
                }}
              />
            )}
            {/* Vertical crosshair line */}
            {hoverInfo && (
              <div
                className="absolute pointer-events-none z-10"
                style={{
                  left: hoverInfo.xPixel,
                  top: 20,   // margin.t
                  bottom: 50, // margin.b
                  width: 1,
                  backgroundColor: 'rgba(255, 255, 255, 0.5)',
                }}
              />
            )}
            {/* Price label on Y-axis (right side) */}
            {hoverInfo && (
              <div
                className="absolute bg-gray-600 text-white text-xs font-mono px-2 py-0.5 rounded-sm pointer-events-none z-20"
                style={{
                  right: 4,
                  top: hoverInfo.yPixel,
                  transform: 'translateY(-50%)',
                }}
              >
                {hoverInfo.price.toFixed(5)}
              </div>
            )}
            {/* Time label on X-axis (bottom) - 24hr format to match axis */}
            {hoverInfo && (
              <div
                className="absolute bg-gray-600 text-white text-xs font-mono px-2 py-0.5 rounded-sm pointer-events-none z-20"
                style={{
                  left: hoverInfo.xPixel,
                  bottom: 8,
                  transform: 'translateX(-50%)',
                }}
              >
                {new Date(hoverInfo.time).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false })}
              </div>
            )}
          </div>
        )}

        {/* MMLC State Overlay - Top Right Corner */}
        {selectedFile && currentCandle && (
          <div className="absolute top-3 right-3 bg-background/85 backdrop-blur-sm rounded-lg border border-border/60 p-3 z-20 shadow-lg">
            {/* Time */}
            <div className="text-sm text-muted-foreground mb-2 font-mono">
              {currentCandle.timestamp}
            </div>

            {/* Main State Row */}
            <div className="flex items-center gap-4 text-base">
              {/* Level */}
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground text-sm">L</span>
                <span
                  className={`font-mono font-bold text-xl ${
                    currentState?.level === 1
                      ? 'text-teal-400'
                      : currentState?.level === 2
                        ? 'text-amber-400'
                        : currentState?.level === 3
                          ? 'text-orange-400'
                          : currentState?.level === 4
                            ? 'text-red-400'
                            : 'text-muted-foreground'
                  }`}
                >
                  {currentState?.level ?? '—'}
                </span>
              </div>

              {/* Event */}
              <div className="flex items-center gap-2">
                <span
                  className={`font-mono font-semibold ${
                    currentState?.event === 'SPAWN'
                      ? 'text-cyan-400'
                      : currentState?.event === 'EXTENSION'
                        ? 'text-purple-400'
                        : currentState?.event === 'REVERSAL'
                          ? 'text-pink-400'
                          : 'text-muted-foreground'
                  }`}
                >
                  {currentState?.event || '—'}
                </span>
              </div>

              {/* Direction */}
              <div className="flex items-center">
                {currentState?.dir === 'UP' ? (
                  <ArrowUp className="h-6 w-6 text-teal-400" />
                ) : currentState?.dir === 'DOWN' ? (
                  <ArrowDown className="h-6 w-6 text-red-400" />
                ) : (
                  <Minus className="h-5 w-5 text-muted-foreground" />
                )}
              </div>
            </div>

            {/* Outcomes Row */}
            <div className="flex items-center gap-4 mt-2 text-sm">
              <div className="flex items-center gap-1.5">
                <ArrowUp className="h-4 w-4 text-muted-foreground" />
                <span
                  className={`font-mono ${
                    (currentState?.out_max_up ?? 0) > 0 ? 'text-teal-400' : 'text-muted-foreground'
                  }`}
                >
                  {currentState?.out_max_up?.toFixed(5) || '—'}
                </span>
              </div>

              <div className="flex items-center gap-1.5">
                <ArrowDown className="h-4 w-4 text-muted-foreground" />
                <span
                  className={`font-mono ${
                    (currentState?.out_max_down ?? 0) < 0 ? 'text-red-400' : 'text-muted-foreground'
                  }`}
                >
                  {currentState?.out_max_down?.toFixed(5) || '—'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
