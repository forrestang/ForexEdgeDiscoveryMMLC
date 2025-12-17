import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useInstruments(workingDirectory?: string) {
  return useQuery({
    queryKey: ['instruments', workingDirectory],
    queryFn: () => api.instruments.list(workingDirectory),
  })
}

export function useInstrumentDates(pair: string | null, workingDirectory?: string) {
  return useQuery({
    queryKey: ['instrument-dates', pair, workingDirectory],
    queryFn: () => api.instruments.getDates(pair!, workingDirectory),
    enabled: !!pair,
  })
}
