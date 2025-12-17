import { useState, useEffect, useRef, useCallback } from 'react'
import { createPortal } from 'react-dom'
import Plot from 'react-plotly.js'
import { ChevronDown } from 'lucide-react'
import { useChartData } from '@/hooks/useChartData'
import type { MatchDetail } from '@/types'
import type { Data, Layout } from 'plotly.js'

interface MatchedSessionModalProps {
  isOpen: boolean
  onClose: () => void
  matches: MatchDetail[]
  workingDirectory: string
}

// Parse session_id to extract chart parameters
// Format: "EURUSD_2024-01-15_london_M5"
function parseSessionId(sessionId: string): {
  pair: string
  date: string
  session: string
  timeframe: string
} | null {
  const parts = sessionId.split('_')
  if (parts.length < 4) return null

  return {
    pair: parts[0],
    date: parts[1],
    session: parts[2],
    timeframe: parts[3],
  }
}

// CSS styles for the popup window
const popupStyles = `
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; overflow: hidden; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    padding: 16px;
  }
  .container { height: calc(100vh - 32px); display: flex; flex-direction: column; }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 16px;
    border-bottom: 1px solid #334155;
  }
  .header h2 { font-size: 18px; font-weight: 600; }
  .selector { padding: 16px 0; border-bottom: 1px solid #334155; }
  .selector-row { display: flex; align-items: center; gap: 16px; }
  .selector-label { font-size: 14px; color: #94a3b8; }
  .dropdown-container { position: relative; flex: 1; max-width: 500px; }
  .dropdown-button {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    color: #e2e8f0;
    font-size: 14px;
    cursor: pointer;
  }
  .dropdown-button:hover { background: #334155; }
  .dropdown-menu {
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    margin-top: 4px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    max-height: 300px;
    overflow-y: auto;
    z-index: 100;
  }
  .dropdown-item {
    width: 100%;
    padding: 8px 12px;
    text-align: left;
    background: none;
    border: none;
    color: #e2e8f0;
    font-size: 14px;
    cursor: pointer;
  }
  .dropdown-item:hover { background: #334155; }
  .dropdown-item.selected { background: #334155; }
  .dropdown-item-header { display: flex; justify-content: space-between; }
  .dropdown-item-meta { font-size: 12px; color: #64748b; margin-top: 4px; }
  .metrics { display: flex; gap: 12px; margin-top: 12px; flex-wrap: wrap; }
  .metric {
    background: #1e293b;
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 14px;
  }
  .metric-label { color: #94a3b8; }
  .metric-value.green { color: #4ade80; }
  .metric-value.red { color: #f87171; }
  .chart-area { flex: 1; padding-top: 16px; min-height: 400px; display: flex; flex-direction: column; width: 100%; overflow: hidden; }
  .chart-area > div { flex: 1; width: 100%; height: 100%; }
  .js-plotly-plot, .plot-container, .svg-container { width: 100% !important; height: 100% !important; }
  .loading, .error, .empty {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #94a3b8;
  }
  .error { color: #f87171; }
  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid #334155;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
`

export function MatchedSessionModal({
  isOpen,
  onClose,
  matches,
  workingDirectory,
}: MatchedSessionModalProps) {
  const [selectedMatchIndex, setSelectedMatchIndex] = useState(0)
  const [isDropdownOpen, setIsDropdownOpen] = useState(false)
  const [popupContainer, setPopupContainer] = useState<HTMLDivElement | null>(null)
  const [chartDimensions, setChartDimensions] = useState({ width: 1150, height: 500 })
  const popupRef = useRef<Window | null>(null)

  // Get the currently selected match
  const selectedMatch = matches[selectedMatchIndex]
  const parsedSession = selectedMatch ? parseSessionId(selectedMatch.session_id) : null

  // Load chart data for the selected match
  const { data: chartData, isLoading, error } = useChartData({
    pair: parsedSession?.pair || '',
    date: parsedSession?.date || '',
    session: parsedSession?.session || 'full_day',
    timeframe: parsedSession?.timeframe || 'M5',
    workingDirectory,
    enabled: isOpen && !!parsedSession,
  })

  // Calculate chart dimensions from popup window
  const updateChartDimensions = useCallback(() => {
    if (popupRef.current && !popupRef.current.closed) {
      const width = popupRef.current.innerWidth - 50 // Account for padding
      const height = popupRef.current.innerHeight - 280 // Account for header, selector, metrics
      setChartDimensions({ width: Math.max(400, width), height: Math.max(300, height) })
    }
  }, [])

  // Open popup window
  const openPopup = useCallback(() => {
    const width = 1200
    const height = 800
    const left = window.screenX + (window.outerWidth - width) / 2
    const top = window.screenY + (window.outerHeight - height) / 2

    const popup = window.open(
      '',
      'MatchedSessionViewer',
      `width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=yes`
    )

    if (popup) {
      popup.document.title = 'Matched Session Viewer'

      // Add styles
      const style = popup.document.createElement('style')
      style.textContent = popupStyles
      popup.document.head.appendChild(style)

      // Create container for React portal
      const container = popup.document.createElement('div')
      container.id = 'popup-root'
      container.className = 'container'
      popup.document.body.appendChild(container)

      popupRef.current = popup
      setPopupContainer(container)

      // Set initial dimensions after a short delay to let the popup render
      setTimeout(() => {
        updateChartDimensions()
      }, 100)

      // Handle resize
      popup.addEventListener('resize', updateChartDimensions)

      // Handle popup close
      popup.addEventListener('beforeunload', () => {
        popupRef.current = null
        setPopupContainer(null)
        onClose()
      })
    }
  }, [onClose, updateChartDimensions])

  // Close popup window
  const closePopup = useCallback(() => {
    if (popupRef.current && !popupRef.current.closed) {
      // Remove event listeners before closing
      popupRef.current.removeEventListener('resize', updateChartDimensions)
      popupRef.current.close()
    }
    popupRef.current = null
    setPopupContainer(null)
  }, [updateChartDimensions])

  // Handle isOpen changes
  useEffect(() => {
    if (isOpen && matches.length > 0) {
      if (!popupRef.current || popupRef.current.closed) {
        openPopup()
      }
      setSelectedMatchIndex(0)
    } else {
      closePopup()
    }
  }, [isOpen, matches.length, openPopup, closePopup])

  // Cleanup on unmount
  useEffect(() => {
    return () => closePopup()
  }, [closePopup])

  // Cleanup Plotly chart to prevent hover layer errors
  const cleanupPlotly = useCallback(() => {
    if (popupRef.current && !popupRef.current.closed) {
      try {
        const plotDiv = popupRef.current.document.getElementById('popup-plotly-chart')
        // Access Plotly from the window object (react-plotly.js exposes it)
        const PlotlyLib = (popupRef.current as Window & { Plotly?: { purge: (el: HTMLElement) => void } }).Plotly
          || (window as Window & { Plotly?: { purge: (el: HTMLElement) => void } }).Plotly
        if (plotDiv && PlotlyLib) {
          PlotlyLib.purge(plotDiv)
        }
      } catch (e) {
        // Ignore cleanup errors
      }
    }
  }, [])

  // Cleanup chart when popup closes or component unmounts
  useEffect(() => {
    return () => {
      cleanupPlotly()
    }
  }, [cleanupPlotly])

  // Cleanup chart before it re-renders with new data
  useEffect(() => {
    return () => {
      cleanupPlotly()
    }
  }, [selectedMatchIndex, cleanupPlotly])

  // Build chart data and layout
  const buildChartData = (): { data: Data[]; layout: Partial<Layout> } => {
    if (!chartData?.candles || chartData.candles.length === 0) {
      return { data: [], layout: {} }
    }

    const candles = chartData.candles
    const barIndex = selectedMatch?.bar_index ?? candles.length - 1
    const selectedTimestamp = candles[barIndex]?.timestamp

    // Calculate price range for vertical line
    const allHighs = candles.map(c => c.high)
    const allLows = candles.map(c => c.low)
    const minPrice = Math.min(...allLows)
    const maxPrice = Math.max(...allHighs)
    const priceRange = maxPrice - minPrice
    const yMin = minPrice - priceRange * 0.02
    const yMax = maxPrice + priceRange * 0.02

    // Candlestick trace - use timestamps directly like main chart
    const candlestickTrace: Data = {
      type: 'candlestick',
      x: candles.map(c => c.timestamp),
      open: candles.map(c => c.open),
      high: candles.map(c => c.high),
      low: candles.map(c => c.low),
      close: candles.map(c => c.close),
      increasing: { line: { color: '#22c55e' }, fillcolor: '#22c55e' },
      decreasing: { line: { color: '#ef4444' }, fillcolor: '#ef4444' },
      name: 'Price',
      showlegend: false,
    }

    // Waveform traces - use timestamps directly
    const waveTraces: Data[] = (chartData.waveform || []).map(wave => ({
      type: 'scatter',
      mode: 'lines',
      x: [wave.start_time, wave.end_time],
      y: [wave.start_price, wave.end_price],
      line: { color: wave.color, width: Math.max(1, 4 - wave.level) },
      name: `L${wave.level}`,
      hoverinfo: 'none',
      showlegend: false,
    }))

    // Vertical line at matched bar position as a scatter trace
    const verticalLineTrace: Data | null = selectedTimestamp ? {
      type: 'scatter',
      mode: 'lines',
      x: [selectedTimestamp, selectedTimestamp],
      y: [yMin, yMax],
      line: { color: '#f59e0b', width: 2, dash: 'dash' },
      name: 'Match Position',
      hoverinfo: 'skip',
      showlegend: false,
    } : null

    const layout: Partial<Layout> = {
      paper_bgcolor: 'transparent',
      plot_bgcolor: '#0f172a',
      font: { color: '#94a3b8', size: 11 },
      margin: { l: 10, r: 70, t: 10, b: 50 },
      width: chartDimensions.width,
      height: chartDimensions.height,
      xaxis: {
        type: 'date',
        gridcolor: '#1e293b',
        showgrid: true,
        rangeslider: { visible: false },
        tickfont: { color: '#9ca3af' },
      },
      yaxis: {
        side: 'right',
        gridcolor: '#1e293b',
        showgrid: true,
        range: [yMin, yMax],
        tickfont: { color: '#9ca3af' },
      },
      showlegend: false,
      dragmode: 'pan',
    }

    const traces: Data[] = [candlestickTrace, ...waveTraces]
    if (verticalLineTrace) {
      traces.push(verticalLineTrace)
    }

    return { data: traces, layout }
  }

  // Don't render anything in the main window
  if (!popupContainer) return null

  const { data: plotData, layout } = buildChartData()

  const popupContent = (
    <>
      {/* Header */}
      <div className="header">
        <h2>Matched Session Viewer</h2>
      </div>

      {/* Dropdown selector */}
      <div className="selector">
        <div className="selector-row">
          <span className="selector-label">Select Match:</span>
          <div className="dropdown-container">
            <button
              className="dropdown-button"
              onClick={(e) => {
                e.stopPropagation()
                setIsDropdownOpen(!isDropdownOpen)
              }}
            >
              <span>
                {selectedMatch
                  ? `${selectedMatch.session_id} (bar ${selectedMatch.bar_index}, dist: ${selectedMatch.distance.toFixed(3)})`
                  : 'Select a match...'}
              </span>
              <ChevronDown size={16} />
            </button>

            {isDropdownOpen && (
              <div className="dropdown-menu">
                {matches.map((match, index) => (
                  <button
                    key={`${match.session_id}-${match.bar_index}`}
                    className={`dropdown-item ${index === selectedMatchIndex ? 'selected' : ''}`}
                    onClick={(e) => {
                      e.stopPropagation()
                      setSelectedMatchIndex(index)
                      setIsDropdownOpen(false)
                    }}
                  >
                    <div className="dropdown-item-header">
                      <span>{match.session_id}</span>
                      <span style={{ color: '#64748b' }}>
                        bar {match.bar_index} | dist: {match.distance.toFixed(3)}
                      </span>
                    </div>
                    <div className="dropdown-item-meta">
                      Next: {match.next_bar_move > 0 ? '+' : ''}{match.next_bar_move.toFixed(2)} |
                      Drift: {match.session_drift > 0 ? '+' : ''}{match.session_drift.toFixed(2)} |
                      MAE: {match.mae.toFixed(2)} | MFE: +{match.mfe.toFixed(2)}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Match metrics summary */}
        {selectedMatch && (
          <div className="metrics">
            <div className="metric">
              <span className="metric-label">Next Bar: </span>
              <span className={`metric-value ${selectedMatch.next_bar_move > 0 ? 'green' : 'red'}`}>
                {selectedMatch.next_bar_move > 0 ? '+' : ''}{selectedMatch.next_bar_move.toFixed(2)}
              </span>
            </div>
            <div className="metric">
              <span className="metric-label">Session Drift: </span>
              <span className={`metric-value ${selectedMatch.session_drift > 0 ? 'green' : 'red'}`}>
                {selectedMatch.session_drift > 0 ? '+' : ''}{selectedMatch.session_drift.toFixed(2)}
              </span>
            </div>
            <div className="metric">
              <span className="metric-label">MAE: </span>
              <span className="metric-value red">{selectedMatch.mae.toFixed(2)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">MFE: </span>
              <span className="metric-value green">+{selectedMatch.mfe.toFixed(2)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Progress: </span>
              <span className="metric-value">{(selectedMatch.session_progress * 100).toFixed(0)}%</span>
            </div>
          </div>
        )}
      </div>

      {/* Chart area */}
      <div className="chart-area">
        {isLoading ? (
          <div className="loading">
            <div className="spinner" />
          </div>
        ) : error ? (
          <div className="error">
            <p>Error loading chart: {error.message}</p>
          </div>
        ) : !chartData || chartData.candles.length === 0 ? (
          <div className="empty">
            <p>No chart data available for this session</p>
          </div>
        ) : (
          <Plot
            key={`popup-chart-${selectedMatch?.session_id}-${selectedMatch?.bar_index}`}
            divId="popup-plotly-chart"
            data={plotData}
            layout={{
              ...layout,
              hovermode: false, // Disable hover to prevent _hoverlayer errors
            }}
            config={{
              displayModeBar: true,
              displaylogo: false,
              modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'lasso2d', 'select2d'],
              scrollZoom: true,
            }}
          />
        )}
      </div>
    </>
  )

  return createPortal(popupContent, popupContainer)
}
