import type { EdgeProbabilities } from '@/types'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface EdgeStatsOverlayProps {
  edge: EdgeProbabilities | null
  visible: boolean
}

export function EdgeStatsOverlay({ edge, visible }: EdgeStatsOverlayProps) {
  if (!visible || !edge) return null

  const getBiasColor = (pct: number) => {
    if (pct > 0.55) return 'text-green-400'
    if (pct < 0.45) return 'text-red-400'
    return 'text-gray-400'
  }

  const getBiasIcon = (pct: number) => {
    if (pct > 0.55) return <TrendingUp className="h-3 w-3" />
    if (pct < 0.45) return <TrendingDown className="h-3 w-3" />
    return <Minus className="h-3 w-3" />
  }

  const getRRColor = (rr: number) => {
    if (rr > 1.5) return 'text-green-400'
    if (rr < 0.8) return 'text-red-400'
    return 'text-yellow-400'
  }

  return (
    <div className="absolute top-2 right-2 z-10 pointer-events-none">
      <div className="bg-background/90 backdrop-blur-sm border border-border rounded-lg p-3 shadow-lg">
        <div className="text-xs font-medium mb-2 text-muted-foreground">
          Edge Analysis ({edge.num_matches} matches)
        </div>

        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
          {/* Next Bar */}
          <div className="col-span-2 text-muted-foreground font-medium mt-1 border-b border-border pb-1">
            Next Bar
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Bias:</span>
            <span className={`flex items-center gap-0.5 ${getBiasColor(edge.next_bar_up_pct)}`}>
              {getBiasIcon(edge.next_bar_up_pct)}
              {(edge.next_bar_up_pct * 100).toFixed(0)}%
            </span>
          </div>
          <div>
            <span className="text-muted-foreground">Avg: </span>
            <span className={edge.next_bar_avg_move > 0 ? 'text-green-400' : edge.next_bar_avg_move < 0 ? 'text-red-400' : ''}>
              {edge.next_bar_avg_move > 0 ? '+' : ''}{edge.next_bar_avg_move.toFixed(2)}
            </span>
          </div>

          {/* Session */}
          <div className="col-span-2 text-muted-foreground font-medium mt-2 border-b border-border pb-1">
            Session
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Bias:</span>
            <span className={`flex items-center gap-0.5 ${getBiasColor(edge.session_up_pct)}`}>
              {getBiasIcon(edge.session_up_pct)}
              {(edge.session_up_pct * 100).toFixed(0)}%
            </span>
          </div>
          <div>
            <span className="text-muted-foreground">Drift: </span>
            <span className={edge.session_avg_drift > 0 ? 'text-green-400' : edge.session_avg_drift < 0 ? 'text-red-400' : ''}>
              {edge.session_avg_drift > 0 ? '+' : ''}{edge.session_avg_drift.toFixed(2)}
            </span>
          </div>

          {/* Risk/Reward */}
          <div className="col-span-2 text-muted-foreground font-medium mt-2 border-b border-border pb-1">
            Risk/Reward
          </div>
          <div>
            <span className="text-muted-foreground">MAE: </span>
            <span className="text-red-400">{edge.avg_mae.toFixed(2)}</span>
          </div>
          <div>
            <span className="text-muted-foreground">MFE: </span>
            <span className="text-green-400">+{edge.avg_mfe.toFixed(2)}</span>
          </div>
          <div className="col-span-2">
            <span className="text-muted-foreground">R:R Ratio: </span>
            <span className={getRRColor(edge.risk_reward_ratio)}>
              {edge.risk_reward_ratio.toFixed(2)}
            </span>
          </div>
        </div>

        {/* Quality Indicator */}
        <div className="mt-2 pt-2 border-t border-border">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Signal Quality:</span>
            <span className={
              edge.top_10_avg_distance < 0.1 ? 'text-green-400' :
              edge.top_10_avg_distance < 0.2 ? 'text-yellow-400' :
              'text-red-400'
            }>
              {edge.top_10_avg_distance < 0.1 ? 'High' :
               edge.top_10_avg_distance < 0.2 ? 'Medium' : 'Low'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
