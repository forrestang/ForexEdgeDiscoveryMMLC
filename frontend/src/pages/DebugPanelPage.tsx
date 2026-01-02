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
    <div className="min-h-screen bg-background text-foreground p-4 font-mono text-sm">
      <h1 className="text-xl font-bold mb-4">MMLC Debug Panel</h1>

      {/* Current Bar Info */}
      <div className="mb-4 p-3 bg-muted rounded">
        <h2 className="font-semibold text-blue-400 text-lg mb-2">
          Bar {debugState.end_bar} ({debugState.mode}) - {debugState.num_waves_returned} waves
        </h2>
        {debugState.current_candle && (
          <div className="grid grid-cols-4 gap-4 mb-2">
            <div>
              <span className="text-muted-foreground">Open: </span>
              <span>{debugState.current_candle.open.toFixed(5)}</span>
            </div>
            <div>
              <span className="text-muted-foreground">High: </span>
              <span className="text-green-400">{debugState.current_candle.high.toFixed(5)}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Low: </span>
              <span className="text-red-400">{debugState.current_candle.low.toFixed(5)}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Close: </span>
              <span>{debugState.current_candle.close.toFixed(5)}</span>
            </div>
          </div>
        )}
        <div className="text-muted-foreground">
          Prev L1 Direction: <span className="text-white">{debugState.prev_L1_Direction}</span>
        </div>
      </div>

      {/* All Levels */}
      <div className="space-y-4">
        {(debugState.levels || []).map((level) => {
          const color = levelColors[level.level] || 'text-gray-400'

          return (
            <div key={level.level} className="p-3 bg-muted rounded">
              <h3 className={`font-semibold text-lg ${color} mb-2`}>
                L{level.level} - {level.direction}
              </h3>

              <div className="grid grid-cols-3 gap-4 mb-2">
                <div>
                  <span className="text-muted-foreground">High: </span>
                  <span className="text-green-400">{level.high.toFixed(5)}</span>
                  <span className="text-muted-foreground"> @bar </span>
                  <span>{level.high_bar}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Low: </span>
                  <span className="text-red-400">{level.low.toFixed(5)}</span>
                  <span className="text-muted-foreground"> @bar </span>
                  <span>{level.low_bar}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Origin: </span>
                  <span>{level.origin_price.toFixed(5)}</span>
                  <span className="text-muted-foreground"> @bar </span>
                  <span>{level.origin_bar}</span>
                </div>
              </div>

              {/* Swing Points (L1 only) */}
              {level.swing_points && level.swing_points.length > 0 && (
                <div className="mt-2">
                  <div className="text-muted-foreground font-semibold mb-1">
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

              {/* Completed Waves */}
              {level.completed_waves.length > 0 && (
                <div className="mt-2">
                  <div className="text-muted-foreground font-semibold mb-1">
                    Completed Waves ({level.completed_waves.length}):
                  </div>
                  <div className="grid grid-cols-4 gap-2 text-xs">
                    {level.completed_waves.map((w, i) => (
                      <div key={i} className="bg-background p-1 rounded">
                        [{i}] {w.start_bar} @ {w.start_price.toFixed(5)} → {w.end_bar} @ {w.end_price.toFixed(5)}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Spline Segments */}
              {level.spline_segments.length > 0 && (
                <div className="mt-2">
                  <div className="text-muted-foreground font-semibold mb-1">
                    Spline Segments ({level.spline_segments.length}):
                  </div>
                  <div className="grid grid-cols-4 gap-2 text-xs">
                    {level.spline_segments.map((s, i) => (
                      <div key={i} className="bg-background p-1 rounded">
                        [{i}] {s.start_bar} → {s.end_bar}
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
          <h3 className="font-semibold text-purple-500 text-lg mb-2">
            Stitch Permanent Legs ({debugState.stitch_permanent_legs.length})
          </h3>
          <div className="grid grid-cols-4 gap-2 text-xs">
            {debugState.stitch_permanent_legs.map((leg, i) => (
              <div key={i} className="bg-background p-1 rounded">
                [{i}] {leg.start_bar} → {leg.end_bar}
                <span className={leg.direction === 'UP' ? 'text-green-400' : 'text-red-400'}>
                  {' '}{leg.direction}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
