import Plot from 'react-plotly.js'
import { Loader2, TrendingUp } from 'lucide-react'
import type { EdgeGraphDataPoint, BarEdgeData } from '@/types'
import type { Data, Layout, Shape, PlotMouseEvent } from 'plotly.js'

interface EdgeGraphProps {
  graphData: EdgeGraphDataPoint[] | null
  edgeTable: BarEdgeData[] | null
  selectedBarIndex: number | null
  onBarSelect: (index: number) => void
  isLoading: boolean
  error?: string | null
}

export function EdgeGraph({
  graphData,
  edgeTable,
  selectedBarIndex,
  onBarSelect,
  isLoading,
  error,
}: EdgeGraphProps) {
  // Loading state
  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground bg-background">
        <Loader2 className="h-5 w-5 animate-spin mr-2" />
        <span className="text-sm">Mining edge scores...</span>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground bg-background">
        <span className="text-sm text-yellow-500">{error}</span>
      </div>
    )
  }

  // Empty state
  if (!graphData || graphData.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground bg-background">
        <TrendingUp className="h-5 w-5 mr-2 opacity-50" />
        <span className="text-sm">Load a chart to see edge scores</span>
      </div>
    )
  }

  // Handle click on chart to select bar
  const handleClick = (event: PlotMouseEvent) => {
    if (!event.points || event.points.length === 0) return
    const barIndex = event.points[0].x as number
    onBarSelect(barIndex)
  }

  // Prepare data for Plotly
  const barIndices = graphData.map((d) => d.bar_index)
  const sessionScores = graphData.map((d) => d.session_score)
  const nextBarScores = graphData.map((d) => d.next_bar_score)

  // Session edge score line (blue)
  const sessionTrace: Data = {
    type: 'scatter',
    mode: 'lines',
    name: 'Session Edge',
    x: barIndices,
    y: sessionScores,
    line: { color: '#3b82f6', width: 2 },
    hovertemplate: 'Bar %{x}<br>Session Edge: %{y:.4f}<extra></extra>',
  }

  // Next bar edge score line (green)
  const nextBarTrace: Data = {
    type: 'scatter',
    mode: 'lines',
    name: 'Next Bar Edge',
    x: barIndices,
    y: nextBarScores,
    line: { color: '#22c55e', width: 2 },
    hovertemplate: 'Bar %{x}<br>Next Bar Edge: %{y:.4f}<extra></extra>',
  }

  // Zero line
  const zeroTrace: Data = {
    type: 'scatter',
    mode: 'lines',
    name: 'Zero',
    x: [0, barIndices.length - 1],
    y: [0, 0],
    line: { color: 'rgba(255,255,255,0.3)', width: 1, dash: 'dash' },
    showlegend: false,
    hoverinfo: 'skip',
  }

  // Selection indicator shape
  const selectionShapes: Partial<Shape>[] =
    selectedBarIndex !== null
      ? [
          {
            type: 'line',
            x0: selectedBarIndex,
            x1: selectedBarIndex,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: {
              color: 'rgba(255, 255, 255, 0.7)',
              width: 2,
              dash: 'dot',
            },
          },
        ]
      : []

  // Layout
  const layout: Partial<Layout> = {
    title: {
      text: 'Edge Scores by Bar',
      font: { color: '#9ca3af', size: 12 },
      x: 0.02,
      xanchor: 'left',
    },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'hsl(222.2, 84%, 4.9%)',
    xaxis: {
      title: { text: 'Bar Index', font: { color: '#6b7280', size: 10 } },
      gridcolor: 'hsl(217.2, 32.6%, 17.5%)',
      linecolor: 'hsl(217.2, 32.6%, 17.5%)',
      tickfont: { color: '#6b7280', size: 9 },
      zeroline: false,
    },
    yaxis: {
      title: { text: 'Edge Score', font: { color: '#6b7280', size: 10 } },
      gridcolor: 'hsl(217.2, 32.6%, 17.5%)',
      linecolor: 'hsl(217.2, 32.6%, 17.5%)',
      tickfont: { color: '#6b7280', size: 9 },
      zeroline: false,
      side: 'right',
    },
    margin: { l: 40, r: 50, t: 30, b: 40 },
    showlegend: true,
    legend: {
      orientation: 'h',
      x: 0.5,
      xanchor: 'center',
      y: 1.1,
      font: { color: '#9ca3af', size: 10 },
    },
    hovermode: 'x unified',
    shapes: selectionShapes,
  }

  const config = {
    displayModeBar: false,
    responsive: true,
  }

  // Get selected bar details for tooltip
  const selectedBarData = edgeTable?.find((e) => e.bar_index === selectedBarIndex)

  return (
    <div className="h-full flex flex-col bg-background border-t border-border">
      {/* Selected bar info */}
      {selectedBarData && (
        <div className="flex items-center gap-4 px-3 py-1 text-xs border-b border-border bg-card/50">
          <span className="text-muted-foreground">Bar {selectedBarIndex}:</span>
          <span className="flex items-center gap-1">
            <span className="text-muted-foreground">Session:</span>
            <span className={selectedBarData.session_edge_score >= 0 ? 'text-green-400' : 'text-red-400'}>
              {selectedBarData.session_edge_score.toFixed(4)}
            </span>
            <span className={`text-xs ${selectedBarData.session_bias === 'long' ? 'text-green-400' : 'text-red-400'}`}>
              ({selectedBarData.session_bias})
            </span>
          </span>
          <span className="flex items-center gap-1">
            <span className="text-muted-foreground">Next Bar:</span>
            <span className={selectedBarData.next_bar_edge_score >= 0 ? 'text-green-400' : 'text-red-400'}>
              {selectedBarData.next_bar_edge_score.toFixed(4)}
            </span>
          </span>
          <span className="flex items-center gap-1">
            <span className="text-muted-foreground">Matches:</span>
            <span>{selectedBarData.num_matches}</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="text-muted-foreground">R:R:</span>
            <span className={selectedBarData.session_risk_reward >= 1.5 ? 'text-green-400' : selectedBarData.session_risk_reward < 0.8 ? 'text-red-400' : 'text-yellow-400'}>
              {selectedBarData.session_risk_reward.toFixed(2)}
            </span>
          </span>
        </div>
      )}
      {/* Chart */}
      <div className="flex-1 min-h-0">
        <Plot
          data={[zeroTrace, sessionTrace, nextBarTrace]}
          layout={layout}
          config={config}
          useResizeHandler
          style={{ width: '100%', height: '100%' }}
          onClick={handleClick}
        />
      </div>
    </div>
  )
}
