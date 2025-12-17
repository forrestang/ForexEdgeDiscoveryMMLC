import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { SessionType, TimeframeType } from '@/types'

interface UseChartDataParams {
  pair: string | null
  date: string | null
  session: SessionType | string
  timeframe: TimeframeType | string
  workingDirectory?: string
  enabled?: boolean
}

export function useChartData(params: UseChartDataParams) {
  const { pair, date, session, timeframe, workingDirectory, enabled = true } = params

  return useQuery({
    queryKey: ['chart', pair, date, session, timeframe, workingDirectory],
    queryFn: () =>
      api.chart.getData(pair!, {
        timeframe: timeframe as TimeframeType,
        date: date!,
        session: session as SessionType,
        workingDirectory,
      }),
    enabled: enabled && !!pair && !!date,
  })
}
