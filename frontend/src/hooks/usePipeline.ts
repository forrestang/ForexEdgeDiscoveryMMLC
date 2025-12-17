import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { ProcessRequest, ProcessAllRequest } from '@/types'

export function usePipelineStatus() {
  return useQuery({
    queryKey: ['pipeline-status'],
    queryFn: () => api.pipeline.getStatus(),
    refetchInterval: (query) => {
      // Poll more frequently while processing
      return query.state.data?.is_processing ? 1000 : false
    },
  })
}

export function useProcessPipeline() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: ProcessRequest) => api.pipeline.process(request),
    onSuccess: () => {
      // Invalidate instruments list to show new data
      queryClient.invalidateQueries({ queryKey: ['instruments'] })
    },
  })
}

// --- Combined Pipeline Hooks ---

export function useAvailablePairs(workingDirectory?: string) {
  return useQuery({
    queryKey: ['available-pairs', workingDirectory],
    queryFn: () => api.pipeline.getAvailablePairs(workingDirectory),
    enabled: !!workingDirectory,
  })
}

export function useProcessAllStatus() {
  return useQuery({
    queryKey: ['process-all-status'],
    queryFn: () => api.pipeline.getProcessAllStatus(),
    refetchInterval: (query) => {
      const data = query.state.data
      // Poll while processing
      if (data?.is_processing) return 1000
      // Poll if not idle yet (transition state) - ensures we catch completion
      if (data?.stage && data.stage !== 'idle' && data.stage !== 'complete') return 1000
      // Stop polling when idle or complete
      return false
    },
  })
}

export function useProcessAll() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: ProcessAllRequest) => api.pipeline.processAll(request),
    onMutate: () => {
      // Start polling by invalidating status
      queryClient.invalidateQueries({ queryKey: ['process-all-status'] })
    },
    onSuccess: () => {
      // Invalidate instruments and sessions to show new data
      queryClient.invalidateQueries({ queryKey: ['instruments'] })
      queryClient.invalidateQueries({ queryKey: ['session-stats'] })
      queryClient.invalidateQueries({ queryKey: ['files'] })
    },
  })
}
