import { useEffect, useCallback } from 'react'
import Plot from 'react-plotly.js'
import { useChartData } from '@/hooks/useChartData'
import { Loader2, BarChart3 } from 'lucide-react'
import type { ChartSettings } from '@/App'
import type { Data, Layout, Shape, PlotMouseEvent } from 'plotly.js'

interface CandlestickChartProps {
  chartSettings: ChartSettings
  workingDirectory: string
  selectedBarIndex: number | null
  onBarSelect: (index: number | null) => void
  onTotalBarsChange: (count: number | null) => void
}

export function CandlestickChart({
  chartSettings,
  workingDirectory,
  selectedBarIndex,
  onBarSelect,
  onTotalBarsChange,
}: CandlestickChartProps) {
  const { data, isLoading, error } = useChartData({
    pair: chartSettings.pair,
    date: chartSettings.date,
    session: chartSettings.session,
    timeframe: chartSettings.timeframe,
    workingDirectory,
  })

  // Update total bars count when data changes
  useEffect(() => {
    if (data?.candles) {
      onTotalBarsChange(data.candles.length)
      // Default to last bar if no selection
      if (selectedBarIndex === null && data.candles.length > 0) {
        onBarSelect(data.candles.length - 1)
      }
    } else {
      onTotalBarsChange(null)
    }
  }, [data?.candles?.length])

  // Handle click on chart to select bar
  const handleClick = useCallback((event: PlotMouseEvent) => {
    if (!data?.candles || !event.points || event.points.length === 0) return

    // Get the clicked point's x value (timestamp)
    const clickedPoint = event.points[0]
    const clickedX = clickedPoint.x

    // Find the closest bar index
    const barIndex = data.candles.findIndex((c) => c.timestamp === clickedX)
    if (barIndex !== -1) {
      onBarSelect(barIndex)
    }
  }, [data?.candles, onBarSelect])

  // Show placeholder when no selection
  if (!chartSettings.pair || !chartSettings.date) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-muted-foreground">
        <BarChart3 className="h-16 w-16 mb-4 opacity-30" />
        <p className="text-lg">Select an instrument and date</p>
        <p className="text-sm">Use the Explore tab to configure chart settings</p>
      </div>
    )
  }

  // Show loading state
  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  // Show error state
  if (error) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-red-400">
        <p className="text-lg">Error loading chart</p>
        <p className="text-sm">{error.message}</p>
      </div>
    )
  }

  // Show empty state
  if (!data || data.candles.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-muted-foreground">
        <p className="text-lg">No data available</p>
        <p className="text-sm">No candles found for the selected date and session</p>
      </div>
    )
  }

  // Determine the effective selected index (default to last bar)
  const effectiveSelectedIndex = selectedBarIndex ?? data.candles.length - 1

  // Split candles into "past" (including selected) and "future" for opacity treatment
  const pastCandles = data.candles.slice(0, effectiveSelectedIndex + 1)
  const futureCandles = data.candles.slice(effectiveSelectedIndex + 1)

  // Prepare candlestick trace for past bars (full opacity)
  const pastCandlestickTrace: Data = {
    type: 'candlestick',
    x: pastCandles.map((c) => c.timestamp),
    open: pastCandles.map((c) => c.open),
    high: pastCandles.map((c) => c.high),
    low: pastCandles.map((c) => c.low),
    close: pastCandles.map((c) => c.close),
    increasing: { line: { color: '#26a69a' }, fillcolor: '#26a69a' },
    decreasing: { line: { color: '#ef5350' }, fillcolor: '#ef5350' },
    name: 'Price',
    showlegend: false,
  }

  // Prepare candlestick trace for future bars (reduced opacity)
  const futureCandlestickTrace: Data = futureCandles.length > 0 ? {
    type: 'candlestick',
    x: futureCandles.map((c) => c.timestamp),
    open: futureCandles.map((c) => c.open),
    high: futureCandles.map((c) => c.high),
    low: futureCandles.map((c) => c.low),
    close: futureCandles.map((c) => c.close),
    increasing: { line: { color: 'rgba(38, 166, 154, 0.3)' }, fillcolor: 'rgba(38, 166, 154, 0.3)' },
    decreasing: { line: { color: 'rgba(239, 83, 80, 0.3)' }, fillcolor: 'rgba(239, 83, 80, 0.3)' },
    name: 'Future',
    showlegend: false,
    hoverinfo: 'skip',
  } : null

  // Prepare waveform traces (one scatter trace per wave)
  // Apply opacity based on whether wave ends before or after selected bar
  const selectedTimestamp = data.candles[effectiveSelectedIndex]?.timestamp
  const waveformTraces: Data[] = data.waveform.map((wave) => {
    const isInFuture = selectedTimestamp && wave.start_time > selectedTimestamp
    const opacity = isInFuture ? 0.3 : 1
    const baseColor = wave.color

    return {
      type: 'scatter',
      mode: 'lines',
      x: [wave.start_time, wave.end_time],
      y: [wave.start_price, wave.end_price],
      line: {
        color: baseColor,
        width: Math.max(4 - wave.level * 0.5, 1.5),
      },
      opacity,
      name: `L${wave.level}`,
      hoverinfo: 'text',
      hovertext: `Level ${wave.level} (${wave.direction})`,
      showlegend: false,
    }
  })

  // Create selection indicator shape (vertical line at selected bar)
  const selectionShapes: Partial<Shape>[] = selectedTimestamp ? [{
    type: 'line',
    x0: selectedTimestamp,
    x1: selectedTimestamp,
    y0: 0,
    y1: 1,
    yref: 'paper',
    line: {
      color: 'rgba(255, 255, 255, 0.5)',
      width: 2,
      dash: 'dot',
    },
  }] : []

  // Prepare layout
  const layout: Partial<Layout> = {
    title: {
      text: `${data.pair} - ${data.date} (${data.session.replace('_', ' ').toUpperCase()}) - ${data.timeframe}`,
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
    shapes: selectionShapes,
  }

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'] as const,
    responsive: true,
  }

  // Combine traces, filtering out null future trace
  const allTraces: Data[] = [
    pastCandlestickTrace,
    ...(futureCandlestickTrace ? [futureCandlestickTrace] : []),
    ...waveformTraces,
  ]

  return (
    <div className="flex-1 w-full min-h-0">
      <Plot
        data={allTraces}
        layout={layout}
        config={config}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
        onClick={handleClick}
      />
    </div>
  )
}
