import { useState, useEffect, useMemo, useCallback } from 'react'
import { Copy, Check, Loader2 } from 'lucide-react'
import { api } from '@/lib/api'
import { DEFAULT_WORKING_DIRECTORY } from '@/lib/constants'
import type { MMLCBarData, TimeframeType, SessionType } from '@/types'

const TIMEFRAMES: TimeframeType[] = ['M1', 'M5', 'M10', 'M15', 'M30', 'H1', 'H4']

interface BarStats {
  barIndex: number
  totalLegs: number
  l1Legs: number
  l2PlusLegs: number
  maxLevel: number
}

interface SessionStats {
  maxTotalLegs: number
  maxL1Legs: number
  maxL2PlusLegs: number
  avgTotalLegs: number
  avgL1Legs: number
  avgL2PlusLegs: number
  maxLevelSeen: number
  barStats: BarStats[]
}

interface SessionMeta {
  pair: string
  date: string
  session: string
  timeframe: string
  mode: 'complete' | 'spline' | 'stitch'
  totalBars: number
  availableDates: string[]
}

interface DateRange {
  start: string
  end: string
}

interface TimeframeData {
  mmlcOut: MMLCBarData[]
  stats: SessionStats
  loading: boolean
  error: string | null
}

function computeDateRanges(dates: string[]): DateRange[] {
  if (dates.length === 0) return []

  const sorted = [...dates].sort()
  const ranges: DateRange[] = []
  let rangeStart = sorted[0]
  let prevDate = sorted[0]

  for (let i = 1; i < sorted.length; i++) {
    const curr = sorted[i]
    const prev = new Date(prevDate)
    const current = new Date(curr)
    const diffDays = (current.getTime() - prev.getTime()) / (1000 * 60 * 60 * 24)

    if (diffDays > 4) {
      ranges.push({ start: rangeStart, end: prevDate })
      rangeStart = curr
    }
    prevDate = curr
  }

  ranges.push({ start: rangeStart, end: prevDate })
  return ranges
}

function computeStats(mmlcOut: MMLCBarData[]): SessionStats {
  if (mmlcOut.length === 0) {
    return {
      maxTotalLegs: 0,
      maxL1Legs: 0,
      maxL2PlusLegs: 0,
      avgTotalLegs: 0,
      avgL1Legs: 0,
      avgL2PlusLegs: 0,
      maxLevelSeen: 0,
      barStats: [],
    }
  }

  const barStats: BarStats[] = mmlcOut.map((snapshot) => {
    const l1Legs = snapshot.legs.filter((leg) => leg.level === 1).length
    const l2PlusLegs = snapshot.legs.filter((leg) => leg.level >= 2).length
    const maxLevel = snapshot.legs.length > 0
      ? Math.max(...snapshot.legs.map((leg) => leg.level))
      : 0

    return {
      barIndex: snapshot.bar_index,
      totalLegs: snapshot.legs.length,
      l1Legs,
      l2PlusLegs,
      maxLevel,
    }
  })

  const maxTotalLegs = Math.max(...barStats.map((s) => s.totalLegs))
  const maxL1Legs = Math.max(...barStats.map((s) => s.l1Legs))
  const maxL2PlusLegs = Math.max(...barStats.map((s) => s.l2PlusLegs))
  const maxLevelSeen = Math.max(...barStats.map((s) => s.maxLevel))

  const avgTotalLegs = barStats.reduce((sum, s) => sum + s.totalLegs, 0) / barStats.length
  const avgL1Legs = barStats.reduce((sum, s) => sum + s.l1Legs, 0) / barStats.length
  const avgL2PlusLegs = barStats.reduce((sum, s) => sum + s.l2PlusLegs, 0) / barStats.length

  return {
    maxTotalLegs,
    maxL1Legs,
    maxL2PlusLegs,
    avgTotalLegs,
    avgL1Legs,
    avgL2PlusLegs,
    maxLevelSeen,
    barStats,
  }
}

function formatSingleTimeframeStats(
  tf: string,
  stats: SessionStats,
  includePerBar: boolean = false
): string[] {
  const lines: string[] = []

  lines.push(`[${tf}] Bars: ${stats.barStats.length}, Max Level: L${stats.maxLevelSeen}`)
  lines.push(`  Max Legs: Total=${stats.maxTotalLegs}, L1=${stats.maxL1Legs}, L2+=${stats.maxL2PlusLegs}`)
  lines.push(`  Avg Legs: Total=${stats.avgTotalLegs.toFixed(2)}, L1=${stats.avgL1Legs.toFixed(2)}, L2+=${stats.avgL2PlusLegs.toFixed(2)}`)

  // Distribution
  const distribution: Record<number, number> = {}
  stats.barStats.forEach((bar) => {
    distribution[bar.totalLegs] = (distribution[bar.totalLegs] || 0) + 1
  })
  const entries = Object.entries(distribution)
    .map(([legs, count]) => ({ legs: parseInt(legs), count }))
    .sort((a, b) => a.legs - b.legs)

  const distStr = entries
    .map(({ legs, count }) => `${legs}:${count}`)
    .join(', ')
  lines.push(`  Distribution: {${distStr}}`)

  if (includePerBar && stats.barStats.length > 0) {
    lines.push(`  Per-Bar: Bar(Total/L1/L2+/MaxLvl)`)
    const perBarStr = stats.barStats
      .map((b) => `${b.barIndex}(${b.totalLegs}/${b.l1Legs}/${b.l2PlusLegs}/L${b.maxLevel})`)
      .join(', ')
    lines.push(`    ${perBarStr}`)
  }

  return lines
}

function formatAllStatsAsText(
  allTimeframeData: Record<string, TimeframeData>,
  meta: SessionMeta | null,
  dateRanges: DateRange[],
  activeTimeframe: string
): string {
  const lines: string[] = []

  lines.push('=== MMLC WAVEFORM STATS (ALL TIMEFRAMES) ===')
  lines.push('')

  if (meta) {
    lines.push(`Pair: ${meta.pair}`)
    lines.push(`Date: ${meta.date} (${meta.session})`)
    lines.push(`Mode: ${meta.mode?.toUpperCase() || 'UNKNOWN'}`)
    lines.push(`Total Dates Available: ${meta.availableDates.length}`)
    lines.push('')
  }

  if (dateRanges.length > 0) {
    lines.push('Date Ranges:')
    dateRanges.forEach((range) => {
      lines.push(`  ${range.start} to ${range.end}`)
    })
    lines.push('')
  }

  lines.push('--- STATS BY TIMEFRAME ---')
  lines.push('')

  for (const tf of TIMEFRAMES) {
    const data = allTimeframeData[tf]
    if (data && !data.loading && !data.error && data.stats.barStats.length > 0) {
      const tfLines = formatSingleTimeframeStats(tf, data.stats, tf === activeTimeframe)
      lines.push(...tfLines)
      lines.push('')
    } else if (data?.loading) {
      lines.push(`[${tf}] Loading...`)
      lines.push('')
    } else if (data?.error) {
      lines.push(`[${tf}] Error: ${data.error}`)
      lines.push('')
    } else {
      lines.push(`[${tf}] Not loaded`)
      lines.push('')
    }
  }

  // Detailed per-bar for active timeframe
  const activeData = allTimeframeData[activeTimeframe]
  if (activeData && activeData.stats.barStats.length > 0) {
    lines.push(`--- DETAILED PER-BAR (${activeTimeframe}) ---`)
    lines.push('Bar\tTotal\tL1\tL2+\tMaxLvl')
    activeData.stats.barStats.forEach((bar) => {
      lines.push(`${bar.barIndex}\t${bar.totalLegs}\t${bar.l1Legs}\t${bar.l2PlusLegs}\tL${bar.maxLevel || '-'}`)
    })
  }

  return lines.join('\n')
}

export function StatsPanelPage() {
  const [sessionMeta, setSessionMeta] = useState<SessionMeta | null>(null)
  const [activeTab, setActiveTab] = useState<TimeframeType>('M5')
  const [allTimeframeData, setAllTimeframeData] = useState<Record<string, TimeframeData>>({})
  const [copied, setCopied] = useState(false)
  const [loadingAll, setLoadingAll] = useState(false)

  // Load initial data from localStorage
  useEffect(() => {
    const loadData = () => {
      const storedMeta = localStorage.getItem('mmlc-session-meta')
      if (storedMeta) {
        try {
          const meta = JSON.parse(storedMeta)
          setSessionMeta(meta)
          setActiveTab(meta.timeframe as TimeframeType)
        } catch (e) {
          console.error('Failed to parse session meta:', e)
        }
      }

      const storedMmlcOut = localStorage.getItem('mmlc-out')
      const storedMeta2 = localStorage.getItem('mmlc-session-meta')
      if (storedMmlcOut && storedMeta2) {
        try {
          const mmlcOut = JSON.parse(storedMmlcOut)
          const meta = JSON.parse(storedMeta2)
          const stats = computeStats(mmlcOut)
          setAllTimeframeData((prev) => ({
            ...prev,
            [meta.timeframe]: { mmlcOut, stats, loading: false, error: null },
          }))
        } catch (e) {
          console.error('Failed to parse data:', e)
        }
      }
    }

    loadData()

    // Listen for changes
    const handleStorage = (e: StorageEvent) => {
      if (e.key === 'mmlc-session-meta' && e.newValue) {
        try {
          const meta = JSON.parse(e.newValue)
          setSessionMeta(meta)
        } catch (err) {
          console.error('Failed to parse session meta:', err)
        }
      }
      if (e.key === 'mmlc-out' && e.newValue) {
        try {
          const mmlcOut = JSON.parse(e.newValue)
          const storedMeta = localStorage.getItem('mmlc-session-meta')
          if (storedMeta) {
            const meta = JSON.parse(storedMeta)
            const stats = computeStats(mmlcOut)
            setAllTimeframeData((prev) => ({
              ...prev,
              [meta.timeframe]: { mmlcOut, stats, loading: false, error: null },
            }))
          }
        } catch (err) {
          console.error('Failed to parse mmlc_out:', err)
        }
      }
    }

    window.addEventListener('storage', handleStorage)

    const interval = setInterval(() => {
      const storedMmlcOut = localStorage.getItem('mmlc-out')
      const storedMeta = localStorage.getItem('mmlc-session-meta')
      if (storedMmlcOut && storedMeta) {
        try {
          const mmlcOut = JSON.parse(storedMmlcOut)
          const meta = JSON.parse(storedMeta)
          setSessionMeta((prev) => {
            if (JSON.stringify(prev) !== storedMeta) return meta
            return prev
          })
          setAllTimeframeData((prev) => {
            const currentData = prev[meta.timeframe]
            if (!currentData || JSON.stringify(currentData.mmlcOut) !== storedMmlcOut) {
              return {
                ...prev,
                [meta.timeframe]: {
                  mmlcOut,
                  stats: computeStats(mmlcOut),
                  loading: false,
                  error: null,
                },
              }
            }
            return prev
          })
        } catch {
          // ignore
        }
      }
    }, 100)

    return () => {
      window.removeEventListener('storage', handleStorage)
      clearInterval(interval)
    }
  }, [])

  const dateRanges = useMemo(
    () => computeDateRanges(sessionMeta?.availableDates || []),
    [sessionMeta?.availableDates]
  )

  // Fetch data for a specific timeframe
  const fetchTimeframeData = useCallback(
    async (tf: TimeframeType) => {
      if (!sessionMeta) return

      setAllTimeframeData((prev) => ({
        ...prev,
        [tf]: { mmlcOut: [], stats: computeStats([]), loading: true, error: null },
      }))

      try {
        const result = await api.mmlcDev.run({
          pair: sessionMeta.pair,
          date: sessionMeta.date,
          session: sessionMeta.session as SessionType,
          timeframe: tf,
          mode: sessionMeta.mode || 'stitch',
          workingDirectory: DEFAULT_WORKING_DIRECTORY,
        })

        const stats = computeStats(result.mmlc_out)
        setAllTimeframeData((prev) => ({
          ...prev,
          [tf]: { mmlcOut: result.mmlc_out, stats, loading: false, error: null },
        }))
      } catch (err) {
        setAllTimeframeData((prev) => ({
          ...prev,
          [tf]: {
            mmlcOut: [],
            stats: computeStats([]),
            loading: false,
            error: err instanceof Error ? err.message : 'Unknown error',
          },
        }))
      }
    },
    [sessionMeta]
  )

  // Load all timeframes
  const loadAllTimeframes = useCallback(async () => {
    if (!sessionMeta) return

    setLoadingAll(true)
    for (const tf of TIMEFRAMES) {
      if (!allTimeframeData[tf] || allTimeframeData[tf].error) {
        await fetchTimeframeData(tf)
      }
    }
    setLoadingAll(false)
  }, [sessionMeta, allTimeframeData, fetchTimeframeData])

  // Handle tab click
  const handleTabClick = useCallback(
    (tf: TimeframeType) => {
      setActiveTab(tf)
      if (!allTimeframeData[tf] || allTimeframeData[tf].error) {
        fetchTimeframeData(tf)
      }
    },
    [allTimeframeData, fetchTimeframeData]
  )

  const handleCopy = useCallback(() => {
    const text = formatAllStatsAsText(allTimeframeData, sessionMeta, dateRanges, activeTab)
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }, [allTimeframeData, sessionMeta, dateRanges, activeTab])

  const activeData = allTimeframeData[activeTab]
  const stats = activeData?.stats || computeStats([])

  if (!sessionMeta) {
    return (
      <div className="min-h-screen bg-background text-foreground p-4 font-mono">
        <h1 className="text-xl font-bold mb-4">MMLC Stats Panel</h1>
        <p className="text-muted-foreground">Waiting for data...</p>
        <p className="text-sm text-muted-foreground mt-2">
          Run the engine in the main window to see stats.
        </p>
      </div>
    )
  }

  return (
    <div className="h-screen bg-background text-foreground p-4 font-mono text-sm overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold">MMLC Stats Panel</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={loadAllTimeframes}
            disabled={loadingAll}
            className="flex items-center gap-2 px-3 py-1.5 rounded text-sm font-medium bg-purple-600 text-white hover:bg-purple-700 disabled:opacity-50"
          >
            {loadingAll ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading...
              </>
            ) : (
              'Load All'
            )}
          </button>
          <button
            onClick={handleCopy}
            className={`flex items-center gap-2 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              copied
                ? 'bg-green-600 text-white'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {copied ? (
              <>
                <Check className="h-4 w-4" />
                Copied!
              </>
            ) : (
              <>
                <Copy className="h-4 w-4" />
                Copy All
              </>
            )}
          </button>
        </div>
      </div>

      {/* Session Info & Date Ranges */}
      <div className="mb-4 p-3 bg-muted rounded border border-gray-700">
        <div className="flex items-center justify-between mb-2">
          <span className="text-purple-400 font-semibold">{sessionMeta.pair}</span>
          <span className="text-muted-foreground text-xs">
            {sessionMeta.availableDates.length} dates available
          </span>
        </div>
        <div className="text-xs text-muted-foreground mb-2">
          Date: {sessionMeta.date} ({sessionMeta.session}) | Mode: <span className="text-cyan-400 font-semibold">{sessionMeta.mode?.toUpperCase() || 'UNKNOWN'}</span>
        </div>
        {dateRanges.length > 0 && (
          <div className="mt-2 pt-2 border-t border-gray-700">
            <div className="text-xs text-cyan-400 mb-1">Date Ranges:</div>
            <div className="text-xs text-yellow-300 space-y-0.5">
              {dateRanges.map((range, i) => (
                <div key={i}>
                  {range.start} to {range.end}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Timeframe Tabs */}
      <div className="mb-4">
        <div className="flex gap-1 flex-wrap">
          {TIMEFRAMES.map((tf) => {
            const data = allTimeframeData[tf]
            const isActive = activeTab === tf
            const hasData = data && !data.loading && !data.error && data.stats.barStats.length > 0
            const isLoading = data?.loading

            return (
              <button
                key={tf}
                onClick={() => handleTabClick(tf)}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-colors relative ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : hasData
                    ? 'bg-green-800 text-green-200 hover:bg-green-700'
                    : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }`}
              >
                {isLoading && (
                  <Loader2 className="h-3 w-3 animate-spin absolute -top-1 -right-1" />
                )}
                {tf}
                {hasData && !isActive && (
                  <span className="ml-1 text-green-400">({data.stats.barStats.length})</span>
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* Loading/Error State */}
      {activeData?.loading && (
        <div className="flex items-center justify-center py-8 text-muted-foreground">
          <Loader2 className="h-6 w-6 animate-spin mr-2" />
          Loading {activeTab} data...
        </div>
      )}

      {activeData?.error && (
        <div className="p-4 bg-red-900/50 rounded border border-red-600 text-red-200 mb-4">
          Error loading {activeTab}: {activeData.error}
        </div>
      )}

      {/* Stats Display */}
      {activeData && !activeData.loading && !activeData.error && stats.barStats.length > 0 && (
        <>
          {/* Summary Stats */}
          <div className="mb-6 p-4 bg-muted rounded border-2 border-blue-600">
            <h2 className="text-lg font-bold text-blue-400 mb-4">
              {activeTab} Summary ({stats.barStats.length} bars)
            </h2>

            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="p-3 bg-background rounded">
                <div className="text-muted-foreground text-xs mb-1">Total Bars</div>
                <div className="text-2xl font-bold text-white">{stats.barStats.length}</div>
              </div>
              <div className="p-3 bg-background rounded">
                <div className="text-muted-foreground text-xs mb-1">Max Level Seen</div>
                <div className="text-2xl font-bold text-purple-400">L{stats.maxLevelSeen}</div>
              </div>
              <div className="p-3 bg-background rounded">
                <div className="text-muted-foreground text-xs mb-1">Session Bars</div>
                <div className="text-2xl font-bold text-white">
                  {activeData.mmlcOut[0]?.total_session_bars || '-'}
                </div>
              </div>
            </div>

            {/* Max Counts */}
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-green-400 mb-2">Maximum Legs (any single bar)</h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="p-3 bg-background rounded">
                  <div className="text-muted-foreground text-xs mb-1">Max Total</div>
                  <div className="text-xl font-bold text-yellow-400">{stats.maxTotalLegs}</div>
                </div>
                <div className="p-3 bg-background rounded">
                  <div className="text-muted-foreground text-xs mb-1">Max L1</div>
                  <div className="text-xl font-bold text-yellow-500">{stats.maxL1Legs}</div>
                </div>
                <div className="p-3 bg-background rounded">
                  <div className="text-muted-foreground text-xs mb-1">Max L2+</div>
                  <div className="text-xl font-bold text-orange-400">{stats.maxL2PlusLegs}</div>
                </div>
              </div>
            </div>

            {/* Average Counts */}
            <div>
              <h3 className="text-sm font-semibold text-cyan-400 mb-2">Average Legs (per bar)</h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="p-3 bg-background rounded">
                  <div className="text-muted-foreground text-xs mb-1">Avg Total</div>
                  <div className="text-xl font-bold text-yellow-400">
                    {stats.avgTotalLegs.toFixed(2)}
                  </div>
                </div>
                <div className="p-3 bg-background rounded">
                  <div className="text-muted-foreground text-xs mb-1">Avg L1</div>
                  <div className="text-xl font-bold text-yellow-500">
                    {stats.avgL1Legs.toFixed(2)}
                  </div>
                </div>
                <div className="p-3 bg-background rounded">
                  <div className="text-muted-foreground text-xs mb-1">Avg L2+</div>
                  <div className="text-xl font-bold text-orange-400">
                    {stats.avgL2PlusLegs.toFixed(2)}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Distribution Histogram */}
          <div className="mb-4 p-4 bg-muted rounded border border-gray-700">
            <h2 className="text-lg font-bold text-green-400 mb-4">Leg Count Distribution</h2>

            {(() => {
              const distribution: Record<number, number> = {}
              stats.barStats.forEach((bar) => {
                distribution[bar.totalLegs] = (distribution[bar.totalLegs] || 0) + 1
              })

              const maxCount = Math.max(...Object.values(distribution))
              const entries = Object.entries(distribution)
                .map(([legs, count]) => ({ legs: parseInt(legs), count }))
                .sort((a, b) => a.legs - b.legs)

              return (
                <div className="space-y-1">
                  {entries.map(({ legs, count }) => {
                    const pct = ((count / stats.barStats.length) * 100).toFixed(1)
                    const barWidth = (count / maxCount) * 100

                    return (
                      <div key={legs} className="flex items-center gap-2">
                        <div className="w-16 text-right text-muted-foreground">
                          {legs} legs:
                        </div>
                        <div className="flex-1 h-5 bg-background rounded overflow-hidden">
                          <div
                            className="h-full bg-green-600 flex items-center px-2"
                            style={{ width: `${barWidth}%` }}
                          >
                            <span className="text-xs text-white whitespace-nowrap">
                              {count} ({pct}%)
                            </span>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )
            })()}
          </div>

          {/* Per-Bar Table */}
          <div className="p-4 bg-muted rounded border border-gray-700">
            <h2 className="text-lg font-bold text-purple-400 mb-4">Per-Bar Breakdown</h2>

            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-2 px-2 text-muted-foreground">Bar</th>
                    <th className="text-right py-2 px-2 text-yellow-400">Total</th>
                    <th className="text-right py-2 px-2 text-yellow-500">L1</th>
                    <th className="text-right py-2 px-2 text-orange-400">L2+</th>
                    <th className="text-right py-2 px-2 text-purple-400">Max Lvl</th>
                    <th className="text-left py-2 px-4 text-muted-foreground">Distribution</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.barStats.map((bar) => {
                    const maxDisplay = Math.max(stats.maxTotalLegs, 1)
                    const l1Width = (bar.l1Legs / maxDisplay) * 100
                    const l2Width = (bar.l2PlusLegs / maxDisplay) * 100

                    return (
                      <tr
                        key={bar.barIndex}
                        className="border-b border-gray-800 hover:bg-background/50"
                      >
                        <td className="py-1 px-2 text-white">{bar.barIndex}</td>
                        <td className="py-1 px-2 text-right text-yellow-400 font-bold">
                          {bar.totalLegs}
                        </td>
                        <td className="py-1 px-2 text-right text-yellow-500">{bar.l1Legs}</td>
                        <td className="py-1 px-2 text-right text-orange-400">{bar.l2PlusLegs}</td>
                        <td className="py-1 px-2 text-right text-purple-400">
                          {bar.maxLevel > 0 ? `L${bar.maxLevel}` : '-'}
                        </td>
                        <td className="py-1 px-4">
                          <div className="flex h-3 bg-background rounded overflow-hidden">
                            <div
                              className="bg-yellow-500 h-full"
                              style={{ width: `${l1Width}%` }}
                              title={`L1: ${bar.l1Legs}`}
                            />
                            <div
                              className="bg-orange-500 h-full"
                              style={{ width: `${l2Width}%` }}
                              title={`L2+: ${bar.l2PlusLegs}`}
                            />
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* No data state */}
      {(!activeData || (!activeData.loading && stats.barStats.length === 0)) && !activeData?.error && (
        <div className="text-center py-8 text-muted-foreground">
          <p>No data for {activeTab}. Click the tab to load.</p>
        </div>
      )}
    </div>
  )
}
