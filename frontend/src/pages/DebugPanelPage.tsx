import { useState, useEffect } from 'react'
import type { DebugState } from '@/types'

export function DebugPanelPage() {
  const [debugState, setDebugState] = useState<DebugState | null>(null)

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

    // Listen for changes from main window
    const handleStorage = (e: StorageEvent) => {
      if (e.key === 'mmlc-debug-state' && e.newValue) {
        try {
          setDebugState(JSON.parse(e.newValue))
        } catch (err) {
          console.error('Failed to parse debug state:', err)
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
              <span className="text-yellow-300">[{debugState.stitch_swings.map(s => s.direction > 0 ? '+1' : '-1').join(', ')}]</span>
            </div>
            <div className="grid grid-cols-4 gap-2 text-xs mt-2">
              {debugState.stitch_swings.map((swing, i) => (
                <div key={i} className="bg-background p-1 rounded">
                  [{i}] bar {swing.bar} @ {swing.price.toFixed(5)}
                  <span className={swing.direction > 0 ? 'text-green-400' : 'text-red-400'}>
                    {' '}{swing.direction > 0 ? 'UP' : 'DOWN'}
                  </span>
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
