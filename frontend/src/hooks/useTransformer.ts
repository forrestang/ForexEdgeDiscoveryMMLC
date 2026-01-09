import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'

// ================================================================
// Configuration Hooks
// ================================================================

export function useTransformerConfigDefaults(
  session: string,
  timeframe: string,
  combineSessions?: string
) {
  return useQuery({
    queryKey: ['transformer-config-defaults', session, timeframe, combineSessions],
    queryFn: () =>
      api.transformer.getConfigDefaults({
        session,
        timeframe,
        combine_sessions: combineSessions,
      }),
    staleTime: 60000, // Config defaults rarely change
  })
}

// ================================================================
// Training Status Hooks
// ================================================================

export function useTransformerStatus(enabled = true) {
  return useQuery({
    queryKey: ['transformer-status'],
    queryFn: () => api.transformer.getStatus(),
    enabled,
    refetchInterval: (data) => {
      // Poll every 2s while training, otherwise every 10s
      if (data?.state?.data?.status === 'training') {
        return 2000
      }
      return 10000
    },
  })
}

export function useStartTransformerTraining() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (params: {
      target_session?: string
      combine_sessions?: string
      sequence_length?: number
      batch_size?: number
      d_model?: number
      n_layers?: number
      n_heads?: number
      dropout_rate?: number
      learning_rate?: number
      num_epochs?: number
      early_stopping_patience?: number
      model_name?: string
      save_to_models_folder?: boolean
      working_directory?: string
    }) => api.transformer.startTraining(params),
    onSuccess: () => {
      // Immediately refetch status to show training started
      queryClient.invalidateQueries({ queryKey: ['transformer-status'] })
    },
  })
}

export function useStopTransformerTraining() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => api.transformer.stopTraining(),
    onSuccess: () => {
      // Refetch status to reflect stopped state
      queryClient.invalidateQueries({ queryKey: ['transformer-status'] })
    },
  })
}

// ================================================================
// Model Management Hooks
// ================================================================

export function useTransformerModels(workingDirectory?: string) {
  return useQuery({
    queryKey: ['transformer-models', workingDirectory],
    queryFn: () => api.transformer.getModels(workingDirectory),
    enabled: !!workingDirectory,
  })
}

export function useDeleteTransformerModel() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (params: { modelName: string; workingDirectory?: string }) =>
      api.transformer.deleteModel(params.modelName, params.workingDirectory),
    onSuccess: (_, variables) => {
      // Invalidate models list to remove deleted model
      queryClient.invalidateQueries({
        queryKey: ['transformer-models', variables.workingDirectory],
      })
    },
  })
}

// ================================================================
// Parquet Viewer Hooks
// ================================================================

export function useParquetFiles(workingDirectory?: string) {
  return useQuery({
    queryKey: ['transformer-parquet-files', workingDirectory],
    queryFn: () => api.transformer.getParquetFiles(workingDirectory),
    enabled: !!workingDirectory,
  })
}

export function useParquetDates(filename: string | null, workingDirectory?: string) {
  return useQuery({
    queryKey: ['transformer-parquet-dates', filename, workingDirectory],
    queryFn: () =>
      api.transformer.getParquetDates({
        filename: filename!,
        working_directory: workingDirectory,
      }),
    enabled: !!filename && !!workingDirectory,
  })
}

export function useParquetData(
  filename: string | null,
  workingDirectory?: string,
  session = 'lon',
  startIdx = 0,
  limit = 500,
  date: string | null = null
) {
  return useQuery({
    queryKey: ['transformer-parquet-data', filename, workingDirectory, session, startIdx, limit, date],
    queryFn: () =>
      api.transformer.getParquetData({
        filename: filename!,
        working_directory: workingDirectory,
        session,
        start_idx: startIdx,
        limit,
        date: date || undefined,
      }),
    enabled: !!filename && !!workingDirectory,
  })
}

// ================================================================
// Queue Hooks
// ================================================================

export function useQueueStatus(enabled = true) {
  return useQuery({
    queryKey: ['transformer-queue-status'],
    queryFn: () => api.transformer.getQueueStatus(),
    enabled,
    refetchInterval: (data) => {
      // Poll every 2s while queue is running, otherwise every 10s
      if (data?.state?.data?.queue_running) {
        return 2000
      }
      return 10000
    },
  })
}

export function useAddToQueue() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (params: {
      card_id: string
      model_name: string
      session_option: string
      sequence_length?: number
      batch_size?: number
      d_model?: number
      n_layers?: number
      n_heads?: number
      dropout_rate?: number
      learning_rate?: number
      num_epochs?: number
      early_stopping_patience?: number
      save_to_models_folder?: boolean
      working_directory?: string
    }) => api.transformer.addToQueue(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['transformer-queue-status'] })
    },
  })
}

export function useRemoveFromQueue() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (cardId: string) => api.transformer.removeFromQueue(cardId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['transformer-queue-status'] })
    },
  })
}

export function useStartQueue() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (workingDirectory?: string) => api.transformer.startQueue(workingDirectory),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['transformer-queue-status'] })
    },
  })
}

export function useStopQueue() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => api.transformer.stopQueue(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['transformer-queue-status'] })
    },
  })
}

export function useClearQueue() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => api.transformer.clearQueue(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['transformer-queue-status'] })
    },
  })
}
