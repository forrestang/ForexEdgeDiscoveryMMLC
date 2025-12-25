import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { SessionType, TimeframeType } from '@/types'

interface UseChartDataParams {
  pair: string | null
  date: string | null
  session: SessionType | string
  timeframe: TimeframeType | string
  workingDirectory?: string
  barIndex?: number
  enabled?: boolean
}

export function useChartData(params: UseChartDataParams) {
  const { pair, date, session, timeframe, workingDirectory, barIndex, enabled = true } = params

  return useQuery({
    queryKey: ['chart', pair, date, session, timeframe, workingDirectory, barIndex],
    queryFn: () =>
      api.chart.getData(pair!, {
        timeframe: timeframe as TimeframeType,
        date: date!,
        session: session as SessionType,
        workingDirectory,
        barIndex,
      }),
    enabled: enabled && !!pair && !!date,
    gcTime: 0, // Don't cache during development
  })
}

/**
 * Hook to fetch MMLC snapshot data for a specific bar.
 * Used by the debug panel - always fetches snapshot data regardless of chart toggle.
 */
export function useSnapshotData(params: UseChartDataParams) {
  const { pair, date, session, timeframe, workingDirectory, barIndex, enabled = true } = params

  return useQuery({
    queryKey: ['snapshot', pair, date, session, timeframe, workingDirectory, barIndex],
    queryFn: () =>
      api.chart.getData(pair!, {
        timeframe: timeframe as TimeframeType,
        date: date!,
        session: session as SessionType,
        workingDirectory,
        barIndex,
      }),
    enabled: enabled && !!pair && !!date && barIndex !== undefined,
    // Only extract the snapshot from the response
    select: (data) => data.snapshot,
  })
}
