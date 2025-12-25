import { useState, useRef, useEffect } from 'react'
import { ChevronDown, ChevronRight, Bug, ArrowUp, ArrowDown } from 'lucide-react'
import { usePersistedState } from '@/hooks/usePersistedSettings'
import type { StackSnapshotData } from '@/types'

interface MMLCDebugPanelProps {
  snapshot: StackSnapshotData | null | undefined
  selectedBarIndex: number | null
  isLoading?: boolean
}

const WAVE_COLORS = [
  '#FFD700',  // Yellow (L1)
  '#00FFFF',  // Cyan (L2)
  '#FF0000',  // Red (L3)
  '#800080',  // Purple (L4)
  '#90EE90',  // Light Green (L5)
]

export function MMLCDebugPanel({
  snapshot,
  selectedBarIndex,
  isLoading = false,
}: MMLCDebugPanelProps) {
  const [isCollapsed, setIsCollapsed] = usePersistedState('mmlcDebugCollapsed', false)
  const [panelHeight, setPanelHeight] = usePersistedState('mmlcDebugHeight', 180)
  const [isResizing, setIsResizing] = useState(false)
  const panelRef = useRef<HTMLDivElement>(null)

  // Handle resize drag
  useEffect(() => {
    if (!isResizing) return

    const handleMouseMove = (e: MouseEvent) => {
      if (panelRef.current) {
        const rect = panelRef.current.getBoundingClientRect()
        const newHeight = rect.bottom - e.clientY
        setPanelHeight(Math.max(100, Math.min(400, newHeight)))
      }
    }

    const handleMouseUp = () => {
      setIsResizing(false)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isResizing, setPanelHeight])

  const barLabel = selectedBarIndex !== null ? `Bar ${selectedBarIndex}` : 'No bar selected'

  return (
    <div
      ref={panelRef}
      className="border-t border-border bg-card flex flex-col"
      style={{ height: isCollapsed ? 'auto' : panelHeight }}
    >
      {/* Resize Handle */}
      {!isCollapsed && (
        <div
          className="h-1 cursor-ns-resize bg-border hover:bg-primary/50 transition-colors"
          onMouseDown={() => setIsResizing(true)}
        />
      )}

      {/* Header */}
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="flex items-center gap-2 px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors w-full text-left"
      >
        {isCollapsed ? (
          <ChevronRight className="h-3 w-3" />
        ) : (
          <ChevronDown className="h-3 w-3" />
        )}
        <Bug className="h-3 w-3" />
        <span>MMLC Debug</span>
        <span className="text-muted-foreground">({barLabel})</span>
        {isLoading && (
          <span className="ml-auto text-muted-foreground animate-pulse">Loading...</span>
        )}
      </button>

      {/* Content */}
      {!isCollapsed && (
        <div className="flex-1 overflow-auto px-3 pb-3 space-y-3">
          {!snapshot ? (
            <div className="text-xs text-muted-foreground italic">
              {selectedBarIndex === null
                ? 'Select a bar to view MMLC state'
                : 'No snapshot data available'}
            </div>
          ) : (
            <>
              {/* Cumulative Leg Counts */}
              <div>
                <div className="text-xs font-medium mb-1">Cumulative Leg Counts</div>
                <div className="flex gap-2 text-xs font-mono">
                  <span style={{ color: WAVE_COLORS[0] }}>L1: {snapshot.l1_count}</span>
                  <span style={{ color: WAVE_COLORS[1] }}>L2: {snapshot.l2_count}</span>
                  <span style={{ color: WAVE_COLORS[2] }}>L3: {snapshot.l3_count}</span>
                  <span style={{ color: WAVE_COLORS[3] }}>L4: {snapshot.l4_count}</span>
                  <span style={{ color: WAVE_COLORS[4] }}>L5: {snapshot.l5_count}</span>
                </div>
              </div>

              {/* Active Waves */}
              <div>
                <div className="text-xs font-medium mb-1">Active Waves ({snapshot.waves.length})</div>
                {snapshot.waves.length === 0 ? (
                  <div className="text-xs text-muted-foreground italic">No active waves</div>
                ) : (
                  <div className="space-y-1">
                    {snapshot.waves.map((wave, idx) => (
                      <div
                        key={idx}
                        className="flex items-center gap-2 text-xs font-mono bg-secondary/50 rounded px-2 py-1"
                      >
                        <span
                          className="font-medium"
                          style={{ color: WAVE_COLORS[(wave.level - 1) % WAVE_COLORS.length] }}
                        >
                          L{wave.level}
                        </span>
                        {wave.direction > 0 ? (
                          <ArrowUp className="h-3 w-3 text-green-500" />
                        ) : (
                          <ArrowDown className="h-3 w-3 text-red-500" />
                        )}
                        <span className="text-muted-foreground">
                          started bar {wave.start_bar_index}
                        </span>
                        <span className="text-muted-foreground">
                          ({wave.duration_bars} bar{wave.duration_bars !== 1 ? 's' : ''})
                        </span>
                        <span className="ml-auto">
                          {wave.amplitude > 0 ? '+' : ''}{wave.amplitude.toFixed(5)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Additional Info */}
              <div className="text-xs text-muted-foreground border-t border-border pt-2 mt-2">
                <div>Close: {snapshot.close_price.toFixed(5)}</div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
