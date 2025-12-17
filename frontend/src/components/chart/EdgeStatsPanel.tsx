import type { EdgeProbabilities } from '@/types'
import { TrendingUp, TrendingDown, Minus, Loader2, AlertCircle, Eye } from 'lucide-react'

interface EdgeStatsPanelProps {
  edge: EdgeProbabilities | null
  isLoading: boolean
  error?: string | null
  selectedBarIndex: number | null
  totalBars: number | null
  kNeighbors?: number
  onViewMatches?: () => void
}

export function EdgeStatsPanel({ edge, isLoading, error, selectedBarIndex, totalBars, kNeighbors = 500, onViewMatches }: EdgeStatsPanelProps) {
  const getBiasColor = (pct: number) => {
    if (pct > 0.55) return 'text-green-400'
    if (pct < 0.45) return 'text-red-400'
    return 'text-gray-400'
  }

  const getBiasIcon = (pct: number) => {
    if (pct > 0.55) return <TrendingUp className="h-4 w-4" />
    if (pct < 0.45) return <TrendingDown className="h-4 w-4" />
    return <Minus className="h-4 w-4" />
  }

  // Format bias to show dominant direction
  const formatBias = (upPct: number) => {
    if (upPct >= 0.5) {
      return `${(upPct * 100).toFixed(0)}% Up`
    } else {
      return `${((1 - upPct) * 100).toFixed(0)}% Down`
    }
  }

  const getRRColor = (rr: number) => {
    if (rr > 1.5) return 'text-green-400'
    if (rr < 0.8) return 'text-red-400'
    return 'text-yellow-400'
  }

  const getQualityColor = (distance: number) => {
    if (distance < 0.1) return 'text-green-400'
    if (distance < 0.2) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getQualityLabel = (distance: number) => {
    if (distance < 0.1) return 'High'
    if (distance < 0.2) return 'Medium'
    return 'Low'
  }

  // Loading state
  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground">
        <Loader2 className="h-5 w-5 animate-spin mr-2" />
        <span className="text-sm">Loading edge analysis...</span>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground">
        <AlertCircle className="h-5 w-5 mr-2 text-yellow-500" />
        <span className="text-sm">{error}</span>
      </div>
    )
  }

  // Empty state
  if (!edge) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground">
        <span className="text-sm">Load a chart with a generated session to see edge probabilities</span>
      </div>
    )
  }

  return (
    <div className="h-full bg-background border-t border-border p-3 overflow-auto">
      <div className="flex flex-wrap gap-4">
        {/* Position Context */}
        <div className="bg-card border border-border rounded-lg p-3 min-w-[160px]">
          <div className="text-xs font-medium text-muted-foreground mb-2">Position</div>
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">Selected Bar:</span>
              <span className="text-sm font-medium">
                {selectedBarIndex !== null && totalBars !== null
                  ? `${selectedBarIndex + 1} / ${totalBars}`
                  : '—'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">Match Avg:</span>
              <span className="text-sm font-medium">
                {(edge.avg_session_progress * 100).toFixed(0)}% into session
              </span>
            </div>
          </div>
        </div>

        {/* Match Quality */}
        <div className="bg-card border border-border rounded-lg p-3 min-w-[180px]">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-muted-foreground">Pattern Matches</span>
            {onViewMatches && edge.top_matches && edge.top_matches.length > 0 && (
              <button
                onClick={onViewMatches}
                className="flex items-center gap-1 px-2 py-0.5 text-xs bg-primary/20 hover:bg-primary/30 text-primary rounded transition-colors"
              >
                <Eye className="h-3 w-3" />
                View
              </button>
            )}
          </div>
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">K-Neighbors:</span>
              <span className="text-sm font-medium">{kNeighbors}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">Unique Sessions:</span>
              <span className="text-sm font-medium">{edge.num_matches}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">Quality:</span>
              <span className={`text-sm font-medium ${getQualityColor(edge.top_10_avg_distance)}`}>
                {getQualityLabel(edge.top_10_avg_distance)}
              </span>
            </div>
          </div>
        </div>

        {/* Next Bar */}
        <div className="bg-card border border-border rounded-lg p-3 min-w-[160px]">
          <div className="text-xs font-medium text-muted-foreground mb-2">Next Bar</div>
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">Bias:</span>
              <span className={`flex items-center gap-1 text-sm font-medium ${getBiasColor(edge.next_bar_up_pct)}`}>
                {getBiasIcon(edge.next_bar_up_pct)}
                {formatBias(edge.next_bar_up_pct)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">Avg Move:</span>
              <span className={`text-sm font-medium ${edge.next_bar_avg_move > 0 ? 'text-green-400' : edge.next_bar_avg_move < 0 ? 'text-red-400' : ''}`}>
                {edge.next_bar_avg_move > 0 ? '+' : ''}{edge.next_bar_avg_move.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        {/* Session */}
        <div className="bg-card border border-border rounded-lg p-3 min-w-[160px]">
          <div className="text-xs font-medium text-muted-foreground mb-2">Session</div>
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">Bias:</span>
              <span className={`flex items-center gap-1 text-sm font-medium ${getBiasColor(edge.session_up_pct)}`}>
                {getBiasIcon(edge.session_up_pct)}
                {formatBias(edge.session_up_pct)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">Drift:</span>
              <span className={`text-sm font-medium ${edge.session_avg_drift > 0 ? 'text-green-400' : edge.session_avg_drift < 0 ? 'text-red-400' : ''}`}>
                {edge.session_avg_drift > 0 ? '+' : ''}{edge.session_avg_drift.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        {/* Risk/Reward */}
        <div className="bg-card border border-border rounded-lg p-3 min-w-[180px]">
          <div className="text-xs font-medium text-muted-foreground mb-2">Risk/Reward</div>
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">MAE (Risk):</span>
              <span className="text-sm font-medium text-red-400">{edge.avg_mae.toFixed(2)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">MFE (Reward):</span>
              <span className="text-sm font-medium text-green-400">+{edge.avg_mfe.toFixed(2)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">R:R Ratio:</span>
              <span className={`text-sm font-medium ${getRRColor(edge.risk_reward_ratio)}`}>
                {edge.risk_reward_ratio.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
