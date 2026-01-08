import { useState, useEffect } from 'react'
import type { DebugState, MMLCBarData, LSTMBarPayload } from '@/types'

export function DebugPanelPage() {
  const [debugState, setDebugState] = useState<DebugState | null>(null)
  const [mmlcOut, setMmlcOut] = useState<MMLCBarData[]>([])
  const [lstmOut, setLstmOut] = useState<LSTMBarPayload[]>([])

  useEffect(() => {
    // Load initial state from localStorage
    const stored = localStorage.getItem('mmlc-debug-state')
    if (stored) {
      try {
        setDebugState(JSON.parse(stored))
      } catch (e) {
        console.error('Failed to parse debug state:', e)
      }
    }

    // Load mmlc_out from localStorage
    const storedMmlcOut = localStorage.getItem('mmlc-out')
    console.log('[DebugPanel] Loading mmlc_out from localStorage:', storedMmlcOut ? 'found' : 'not found')
    if (storedMmlcOut) {
      try {
        const parsed = JSON.parse(storedMmlcOut)
        console.log('[DebugPanel] Parsed mmlc_out:', parsed.length, 'snapshots')
        console.log('[DebugPanel] First snapshot swings:', parsed[0]?.swings)
        setMmlcOut(parsed)
      } catch (e) {
        console.error('Failed to parse mmlc_out:', e)
      }
    }

    // Load lstm_out from localStorage
    const storedLstmOut = localStorage.getItem('lstm-out')
    if (storedLstmOut) {
      try {
        const parsed = JSON.parse(storedLstmOut)
        console.log('[DebugPanel] Parsed lstm_out:', parsed.length, 'payloads')
        setLstmOut(parsed)
      } catch (e) {
        console.error('Failed to parse lstm_out:', e)
      }
    }

    // Listen for changes from main window
    const handleStorage = (e: StorageEvent) => {
      if (e.key === 'mmlc-debug-state' && e.newValue) {
        try {
          setDebugState(JSON.parse(e.newValue))
        } catch (err) {
          console.error('Failed to parse debug state:', err)
        }
      }
      if (e.key === 'mmlc-out' && e.newValue) {
        try {
          setMmlcOut(JSON.parse(e.newValue))
        } catch (err) {
          console.error('Failed to parse mmlc_out:', err)
        }
      }
      if (e.key === 'lstm-out' && e.newValue) {
        try {
          setLstmOut(JSON.parse(e.newValue))
        } catch (err) {
          console.error('Failed to parse lstm_out:', err)
        }
      }
    }

    window.addEventListener('storage', handleStorage)

    // Also poll for changes (backup in case storage events don't fire in same origin)
    const interval = setInterval(() => {
      const stored = localStorage.getItem('mmlc-debug-state')
      if (stored) {
        try {
          const parsed = JSON.parse(stored)
          setDebugState(prev => {
            if (JSON.stringify(prev) !== stored) {
              return parsed
            }
            return prev
          })
        } catch (e) {
          // ignore
        }
      }
      const storedMmlcOut = localStorage.getItem('mmlc-out')
      if (storedMmlcOut) {
        try {
          const parsed = JSON.parse(storedMmlcOut)
          setMmlcOut(prev => {
            if (JSON.stringify(prev) !== storedMmlcOut) {
              return parsed
            }
            return prev
          })
        } catch (e) {
          // ignore
        }
      }
      const storedLstmOut = localStorage.getItem('lstm-out')
      if (storedLstmOut) {
        try {
          const parsed = JSON.parse(storedLstmOut)
          setLstmOut(prev => {
            if (JSON.stringify(prev) !== storedLstmOut) {
              return parsed
            }
            return prev
          })
        } catch (e) {
          // ignore
        }
      }
    }, 100)

    return () => {
      window.removeEventListener('storage', handleStorage)
      clearInterval(interval)
    }
  }, [])

  if (!debugState) {
    return (
      <div className="min-h-screen bg-background text-foreground p-4 font-mono">
        <h1 className="text-xl font-bold mb-4">MMLC Debug Panel</h1>
        <p className="text-muted-foreground">Waiting for debug data...</p>
        <p className="text-sm text-muted-foreground mt-2">Run the engine in the main window to see debug state.</p>
      </div>
    )
  }

  const levelColors: Record<number, string> = {
    1: 'text-yellow-500',
    2: 'text-orange-500',
    3: 'text-pink-500',
    4: 'text-purple-500',
    5: 'text-blue-500',
    6: 'text-cyan-500',
  }

  return (
    <div className="h-screen bg-background text-foreground p-4 font-mono text-sm overflow-y-auto">
      <h1 className="text-xl font-bold mb-4">MMLC Debug Panel</h1>

      {/* Current Bar Info */}
      <div className="mb-4 p-3 bg-muted rounded">
        <h2 className="font-semibold text-blue-400 text-lg mb-2">
          Bar {debugState.end_bar} ({debugState.mode}) - {debugState.num_waves_returned} waves
        </h2>
        {debugState.current_candle && (
          <div className="grid grid-cols-4 gap-4 mb-2">
            <div>
              <span className="text-cyan-400">candle.open</span>
              <span className="text-muted-foreground"> = </span>
              <span>{debugState.current_candle.open.toFixed(5)}</span>
            </div>
            <div>
              <span className="text-cyan-400">candle.high</span>
              <span className="text-muted-foreground"> = </span>
              <span className="text-green-400">{debugState.current_candle.high.toFixed(5)}</span>
            </div>
            <div>
              <span className="text-cyan-400">candle.low</span>
              <span className="text-muted-foreground"> = </span>
              <span className="text-red-400">{debugState.current_candle.low.toFixed(5)}</span>
            </div>
            <div>
              <span className="text-cyan-400">candle.close</span>
              <span className="text-muted-foreground"> = </span>
              <span>{debugState.current_candle.close.toFixed(5)}</span>
            </div>
          </div>
        )}
        <div>
          <span className="text-cyan-400">_prev_L1_Direction</span>
          <span className="text-muted-foreground"> = </span>
          <span className="text-white">{debugState.prev_L1_Direction}</span>
        </div>
        <div className="mt-2 grid grid-cols-2 gap-4">
          <div>
            <span className="text-cyan-400 font-bold">L1_leg</span>
            <span className="text-muted-foreground"> = </span>
            <span className="text-green-400 text-lg font-bold">{debugState.L1_leg}</span>
          </div>
          <div>
            <span className="text-cyan-400 font-bold">L1_count</span>
            <span className="text-muted-foreground"> = </span>
            <span className="text-yellow-300 text-lg font-bold">{debugState.L1_count}</span>
          </div>
        </div>
      </div>

      {/* MMLC Output - Bar-by-Bar Payloads */}
      {(() => {
        // Find the payload for the current end_bar
        const currentPayload = lstmOut.find(p => p.sequence_id === debugState.end_bar + 1)
        const eventColors: Record<string, string> = {
          EXTENSION: 'bg-green-900 border-green-500 text-green-400',
          SPAWN: 'bg-orange-900 border-orange-500 text-orange-400',
          REVERSAL: 'bg-red-900 border-red-500 text-red-400',
        }
        return (
          <div className="mb-4 p-3 bg-muted rounded border-2 border-blue-600">
            <h2 className="text-lg font-bold text-blue-400 mb-3">
              MMLC Output - Bar {debugState.end_bar}
            </h2>

            {currentPayload ? (
              <>
                {/* Current Bar Payload */}
                <div className="mb-3 p-3 bg-background rounded">
                  <div className="grid grid-cols-3 gap-4">
                    {/* Left: Vector */}
                    <div>
                      <div className="text-sm font-semibold text-purple-400 mb-2">vector</div>
                      <div className="space-y-1 text-xs">
                        <div>
                          <span className="text-cyan-400">price_raw</span>
                          <span className="text-muted-foreground"> = </span>
                          <span className="text-white">{currentPayload.vector.price_raw.toFixed(5)}</span>
                        </div>
                        <div>
                          <span className="text-cyan-400">price_delta</span>
                          <span className="text-muted-foreground"> = </span>
                          <span className={currentPayload.vector.price_delta >= 0 ? 'text-green-400' : 'text-red-400'}>
                            {currentPayload.vector.price_delta >= 0 ? '+' : ''}{currentPayload.vector.price_delta.toFixed(5)}
                          </span>
                        </div>
                        <div>
                          <span className="text-cyan-400">time_delta</span>
                          <span className="text-muted-foreground"> = </span>
                          <span className="text-white">{currentPayload.vector.time_delta}</span>
                        </div>
                        <div>
                          <span className="text-cyan-400">timestamp</span>
                          <span className="text-muted-foreground"> = </span>
                          <span className="text-white">{currentPayload.timestamp}</span>
                        </div>
                        <div>
                          <span className="text-cyan-400">total_bars</span>
                          <span className="text-muted-foreground"> = </span>
                          <span className="text-white">{currentPayload.total_session_bars}</span>
                        </div>
                      </div>
                    </div>
                    {/* Middle: State */}
                    <div>
                      <div className="text-sm font-semibold text-purple-400 mb-2">state</div>
                      <div className="space-y-1 text-xs">
                        <div>
                          <span className="text-cyan-400">level</span>
                          <span className="text-muted-foreground"> = </span>
                          <span className="text-yellow-400 font-bold">L{currentPayload.state.level}</span>
                        </div>
                        <div>
                          <span className="text-cyan-400">direction</span>
                          <span className="text-muted-foreground"> = </span>
                          <span className={currentPayload.state.direction === 'UP' ? 'text-green-400' : 'text-red-400'}>
                            {currentPayload.state.direction}
                          </span>
                        </div>
                        <div>
                          <span className="text-cyan-400">event</span>
                          <span className="text-muted-foreground"> = </span>
                          <span className={`px-2 py-0.5 rounded border ${eventColors[currentPayload.state.event] || 'text-white'}`}>
                            {currentPayload.state.event}
                          </span>
                        </div>
                      </div>
                    </div>
                    {/* Right: Outcome */}
                    <div>
                      <div className="text-sm font-semibold text-amber-400 mb-2">outcome</div>
                      {currentPayload.outcome ? (
                        <div className="space-y-1 text-xs">
                          <div>
                            <span className="text-cyan-400">next_bar_delta</span>
                            <span className="text-muted-foreground"> = </span>
                            <span className={currentPayload.outcome.next_bar_delta >= 0 ? 'text-green-400' : 'text-red-400'}>
                              {currentPayload.outcome.next_bar_delta >= 0 ? '+' : ''}{currentPayload.outcome.next_bar_delta.toFixed(5)}
                            </span>
                          </div>
                          <div>
                            <span className="text-cyan-400">session_close_delta</span>
                            <span className="text-muted-foreground"> = </span>
                            <span className={currentPayload.outcome.session_close_delta >= 0 ? 'text-green-400' : 'text-red-400'}>
                              {currentPayload.outcome.session_close_delta >= 0 ? '+' : ''}{currentPayload.outcome.session_close_delta.toFixed(5)}
                            </span>
                          </div>
                          <div>
                            <span className="text-cyan-400">session_max_up</span>
                            <span className="text-muted-foreground text-[10px]"> (MFE)</span>
                            <span className="text-muted-foreground"> = </span>
                            <span className="text-green-400">+{currentPayload.outcome.session_max_up.toFixed(5)}</span>
                          </div>
                          <div>
                            <span className="text-cyan-400">session_max_down</span>
                            <span className="text-muted-foreground text-[10px]"> (MAE)</span>
                            <span className="text-muted-foreground"> = </span>
                            <span className="text-red-400">{currentPayload.outcome.session_max_down.toFixed(5)}</span>
                          </div>
                        </div>
                      ) : (
                        <div className="text-muted-foreground text-xs">Not available</div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Recent Payloads Table */}
                <div className="p-2 bg-background rounded">
                  <div className="text-sm font-semibold text-green-400 mb-2">
                    Recent Bar Payloads ({Math.min(lstmOut.length, 10)} of {lstmOut.length})
                  </div>
                  <div className="max-h-48 overflow-y-auto">
                    <table className="w-full text-xs">
                      <thead className="sticky top-0 bg-background">
                        <tr className="text-muted-foreground border-b border-gray-700">
                          <th className="text-left py-1 px-1">seq</th>
                          <th className="text-left py-1 px-1">price</th>
                          <th className="text-right py-1 px-1">delta</th>
                          <th className="text-center py-1 px-1">lvl</th>
                          <th className="text-center py-1 px-1">dir</th>
                          <th className="text-center py-1 px-1">event</th>
                          <th className="text-right py-1 px-1 text-amber-400">next</th>
                          <th className="text-right py-1 px-1 text-green-400">MFE</th>
                          <th className="text-right py-1 px-1 text-red-400">MAE</th>
                        </tr>
                      </thead>
                      <tbody>
                        {lstmOut.slice(-10).reverse().map((payload) => (
                          <tr
                            key={payload.sequence_id}
                            className={`border-b border-gray-800 ${payload.sequence_id === currentPayload.sequence_id ? 'bg-blue-900/30' : ''}`}
                          >
                            <td className="py-1 px-1 text-yellow-300">{payload.sequence_id}</td>
                            <td className="py-1 px-1 text-white">{payload.vector.price_raw.toFixed(5)}</td>
                            <td className={`py-1 px-1 text-right ${payload.vector.price_delta >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {payload.vector.price_delta >= 0 ? '+' : ''}{payload.vector.price_delta.toFixed(5)}
                            </td>
                            <td className="py-1 px-1 text-center text-yellow-400">L{payload.state.level}</td>
                            <td className={`py-1 px-1 text-center ${payload.state.direction === 'UP' ? 'text-green-400' : 'text-red-400'}`}>
                              {payload.state.direction === 'UP' ? '↑' : '↓'}
                            </td>
                            <td className="py-1 px-1 text-center">
                              <span className={`px-1 py-0.5 rounded text-[10px] ${
                                payload.state.event === 'EXTENSION' ? 'bg-green-900/50 text-green-400' :
                                payload.state.event === 'SPAWN' ? 'bg-orange-900/50 text-orange-400' :
                                'bg-red-900/50 text-red-400'
                              }`}>
                                {payload.state.event.slice(0, 3)}
                              </span>
                            </td>
                            <td className={`py-1 px-1 text-right ${payload.outcome?.next_bar_delta !== undefined ? (payload.outcome.next_bar_delta >= 0 ? 'text-green-400' : 'text-red-400') : 'text-muted-foreground'}`}>
                              {payload.outcome?.next_bar_delta !== undefined
                                ? `${payload.outcome.next_bar_delta >= 0 ? '+' : ''}${payload.outcome.next_bar_delta.toFixed(5)}`
                                : '-'}
                            </td>
                            <td className="py-1 px-1 text-right text-green-400">
                              {payload.outcome?.session_max_up !== undefined
                                ? `+${payload.outcome.session_max_up.toFixed(5)}`
                                : '-'}
                            </td>
                            <td className="py-1 px-1 text-right text-red-400">
                              {payload.outcome?.session_max_down !== undefined
                                ? payload.outcome.session_max_down.toFixed(5)
                                : '-'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            ) : (
              <div className="text-muted-foreground text-sm">
                <p>No MMLC data available. Run the engine to generate payloads.</p>
              </div>
            )}

            {/* Summary Stats */}
            <div className="mt-3 pt-2 border-t border-gray-700 text-xs text-muted-foreground">
              Total payloads: {lstmOut.length}
              {lstmOut.length > 0 && (
                <span> (seq {lstmOut[0].sequence_id} to {lstmOut[lstmOut.length - 1].sequence_id})</span>
              )}
            </div>
          </div>
        )
      })()}

      {/* Stitch Swings - Final Waveform Points (ALWAYS SHOW) */}
      <div className="mb-4 p-3 bg-muted rounded border-2 border-green-600">
        <div className="mb-2">
          <span className="text-cyan-400 text-lg font-bold">_stitch_swings</span>
          <span className="text-muted-foreground"> ({debugState.stitch_swings?.length || 0} points)</span>
          {debugState.mode !== 'stitch' && (
            <span className="text-yellow-500 text-xs ml-2">(only populated in stitch mode)</span>
          )}
        </div>
        {debugState.stitch_swings && debugState.stitch_swings.length > 0 ? (
          <>
            <div className="mb-2">
              <span className="text-muted-foreground">bars: </span>
              <span className="text-yellow-300">[{debugState.stitch_swings.map(s => s.bar).join(', ')}]</span>
            </div>
            <div className="mb-2">
              <span className="text-muted-foreground">prices: </span>
              <span className="text-yellow-300">[{debugState.stitch_swings.map(s => s.price.toFixed(5)).join(', ')}]</span>
            </div>
            <div className="mb-2">
              <span className="text-muted-foreground">directions: </span>
              <span className="text-yellow-300">[{debugState.stitch_swings.map(s => s.direction > 0 ? '+1' : (s.direction < 0 ? '-1' : '0')).join(', ')}]</span>
            </div>
            <div className="mb-2">
              <span className="text-muted-foreground">levels: </span>
              <span className="text-purple-400">[{debugState.stitch_swings_level?.join(', ') || ''}]</span>
              <span className="text-muted-foreground text-xs ml-2">(0=OPEN, 1=L1, 2=L2, 3=L3...)</span>
            </div>
            <div className="grid grid-cols-4 gap-2 text-xs mt-2">
              {debugState.stitch_swings.map((swing, i) => (
                <div key={i} className="bg-background p-1 rounded">
                  [{i}] bar {swing.bar} @ {swing.price.toFixed(5)}
                  <span className={swing.direction > 0 ? 'text-green-400' : (swing.direction < 0 ? 'text-red-400' : 'text-gray-400')}>
                    {' '}{swing.direction > 0 ? 'UP' : (swing.direction < 0 ? 'DOWN' : 'OPEN')}
                  </span>
                  <span className="text-purple-400 ml-1">L{debugState.stitch_swings_level?.[i] ?? '?'}</span>
                </div>
              ))}
            </div>
          </>
        ) : (
          <p className="text-muted-foreground text-sm">[] (empty - awaiting population)</p>
        )}
      </div>

      {/* L1 Swing Arrays - Used by Complete/Spline modes */}
      {(() => {
        const l1Level = debugState.levels?.find(l => l.level === 1)
        const swingPoints = l1Level?.swing_points || []
        return (
          <div className="mb-4 p-3 bg-muted rounded border-2 border-yellow-600">
            <div className="mb-2">
              <span className="text-yellow-400 text-lg font-bold">L1_swing_x / L1_swing_y</span>
              <span className="text-muted-foreground"> ({swingPoints.length} points)</span>
              <span className="text-yellow-500 text-xs ml-2">(used by complete/spline mode)</span>
            </div>
            {swingPoints.length > 0 ? (
              <>
                <div className="mb-2">
                  <span className="text-muted-foreground">L1_swing_x: </span>
                  <span className="text-yellow-300">[{swingPoints.map(pt => pt.bar).join(', ')}]</span>
                </div>
                <div className="mb-2">
                  <span className="text-muted-foreground">L1_swing_y: </span>
                  <span className="text-yellow-300">[{swingPoints.map(pt => pt.price.toFixed(5)).join(', ')}]</span>
                </div>
                <div className="grid grid-cols-4 gap-2 text-xs mt-2">
                  {swingPoints.map((pt, i) => (
                    <div key={i} className="bg-background p-1 rounded">
                      [{i}] bar {pt.bar} @ {pt.price.toFixed(5)}
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-muted-foreground text-sm">[] (empty)</p>
            )}
          </div>
        )
      })()}

      {/* All Levels */}
      <div className="space-y-4">
        {(debugState.levels || []).map((level) => {
          const color = levelColors[level.level] || 'text-gray-400'
          const prefix = `L${level.level}`

          return (
            <div key={level.level} className="p-3 bg-muted rounded">
              <h3 className={`font-semibold text-lg ${color} mb-2`}>
                L{level.level} - {level.direction}
              </h3>

              {/* Direction */}
              <div className="mb-2">
                <span className="text-cyan-400">{prefix}_Direction</span>
                <span className="text-muted-foreground"> = </span>
                <span className={level.direction === 'UP' ? 'text-green-400' : level.direction === 'DOWN' ? 'text-red-400' : 'text-gray-400'}>
                  {level.direction}
                </span>
              </div>

              {/* High/Low with variable names */}
              <div className="grid grid-cols-2 gap-4 mb-2">
                <div>
                  <span className="text-cyan-400">{prefix}_High</span>
                  <span className="text-muted-foreground"> = </span>
                  <span className="text-green-400">{level.high.toFixed(5)}</span>
                </div>
                <div>
                  <span className="text-cyan-400">{prefix}_High_bar</span>
                  <span className="text-muted-foreground"> = </span>
                  <span>{level.high_bar}</span>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4 mb-2">
                <div>
                  <span className="text-cyan-400">{prefix}_Low</span>
                  <span className="text-muted-foreground"> = </span>
                  <span className="text-red-400">{level.low.toFixed(5)}</span>
                </div>
                <div>
                  <span className="text-cyan-400">{prefix}_Low_bar</span>
                  <span className="text-muted-foreground"> = </span>
                  <span>{level.low_bar}</span>
                </div>
              </div>

              {/* Origin (L2+ only) */}
              {level.level > 1 && (
                <div className="grid grid-cols-2 gap-4 mb-2">
                  <div>
                    <span className="text-cyan-400">{prefix}_origin_price</span>
                    <span className="text-muted-foreground"> = </span>
                    <span>{level.origin_price.toFixed(5)}</span>
                  </div>
                  <div>
                    <span className="text-cyan-400">{prefix}_origin_bar</span>
                    <span className="text-muted-foreground"> = </span>
                    <span>{level.origin_bar}</span>
                  </div>
                </div>
              )}

              {/* Per-bar indexed counters */}
              <div className="grid grid-cols-2 gap-4 mb-2 mt-2 p-2 bg-background rounded">
                <div>
                  <span className="text-cyan-400 font-bold">{prefix}_leg</span>
                  <span className="text-muted-foreground"> = </span>
                  <span className="text-green-400 font-bold">{level.current_leg}</span>
                </div>
                <div>
                  <span className="text-cyan-400 font-bold">{prefix}_count</span>
                  <span className="text-muted-foreground"> = </span>
                  <span className="text-yellow-300 font-bold">{level.current_count}</span>
                </div>
              </div>

              {/* Per-bar history arrays */}
              {level.leg_history && level.leg_history.length > 0 && (
                <div className="mt-2 space-y-1 text-xs">
                  <div>
                    <span className="text-cyan-400">{prefix}_leg_history</span>
                    <span className="text-muted-foreground"> = </span>
                    <span className="text-green-300">[{level.leg_history.join(', ')}]</span>
                  </div>
                  <div>
                    <span className="text-cyan-400">{prefix}_count_history</span>
                    <span className="text-muted-foreground"> = </span>
                    <span className="text-yellow-200">[{level.count_history.join(', ')}]</span>
                  </div>
                  <div>
                    <span className="text-cyan-400">{prefix}_direction_history</span>
                    <span className="text-muted-foreground"> = </span>
                    <span className="text-purple-300">[{level.direction_history.join(', ')}]</span>
                  </div>
                </div>
              )}

              {/* Swing Points Arrays (L1 only) */}
              {level.swing_points && level.swing_points.length > 0 && (
                <div className="mt-3 space-y-2">
                  <div>
                    <span className="text-cyan-400">{prefix}_swing_x</span>
                    <span className="text-muted-foreground"> = </span>
                    <span className="text-yellow-300">[{level.swing_points.map(pt => pt.bar).join(', ')}]</span>
                  </div>
                  <div>
                    <span className="text-cyan-400">{prefix}_swing_y</span>
                    <span className="text-muted-foreground"> = </span>
                    <span className="text-yellow-300">[{level.swing_points.map(pt => pt.price.toFixed(5)).join(', ')}]</span>
                  </div>
                  <div className="text-muted-foreground text-xs mt-1">
                    Swing Points ({level.swing_points.length}):
                  </div>
                  <div className="grid grid-cols-4 gap-2 text-xs">
                    {level.swing_points.map((pt, i) => (
                      <div key={i} className="bg-background p-1 rounded">
                        [{i}] bar {pt.bar} @ {pt.price.toFixed(5)}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Completed Waves Array */}
              {level.completed_waves.length > 0 && (
                <div className="mt-3">
                  <div className="mb-1">
                    <span className="text-cyan-400">{prefix}_completed_waves</span>
                    <span className="text-muted-foreground"> ({level.completed_waves.length} items):</span>
                  </div>
                  <div className="grid grid-cols-4 gap-2 text-xs">
                    {level.completed_waves.map((w, i) => (
                      <div key={i} className="bg-background p-1 rounded">
                        [{i}] ({w.start_bar}, {w.start_price.toFixed(5)}, {w.end_bar}, {w.end_price.toFixed(5)})
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Spline Segments Array */}
              {level.spline_segments.length > 0 && (
                <div className="mt-3">
                  <div className="mb-1">
                    <span className="text-cyan-400">{prefix}_spline_segments</span>
                    <span className="text-muted-foreground"> ({level.spline_segments.length} items):</span>
                  </div>
                  <div className="grid grid-cols-4 gap-2 text-xs">
                    {level.spline_segments.map((s, i) => (
                      <div key={i} className="bg-background p-1 rounded">
                        [{i}] ({s.start_bar}, {s.start_price.toFixed(5)}, {s.end_bar}, {s.end_price.toFixed(5)})
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Stitch Permanent Legs */}
      {debugState.stitch_permanent_legs.length > 0 && (
        <div className="mt-4 p-3 bg-muted rounded">
          <div className="mb-2">
            <span className="text-cyan-400">_stitch_permanent_legs</span>
            <span className="text-muted-foreground"> ({debugState.stitch_permanent_legs.length} legs):</span>
          </div>
          <div className="space-y-1 text-xs font-mono">
            {debugState.stitch_permanent_legs.map((leg, i) => (
              <div key={i}>
                <span className="text-muted-foreground">[{i}]</span>{' '}
                <span className="text-yellow-400">bar {leg.start_bar}</span>{' '}
                <span className="text-white">({leg.start_price.toFixed(5)})</span>{' '}
                <span className={leg.direction === 'UP' ? 'text-green-400' : 'text-red-400'}>
                  {leg.direction === 'UP' ? '↑' : '↓'}
                </span>{' '}
                <span className="text-yellow-400">bar {leg.end_bar}</span>{' '}
                <span className="text-white">({leg.end_price.toFixed(5)})</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
