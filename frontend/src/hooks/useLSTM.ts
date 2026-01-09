import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useLSTMDataFiles(workingDirectory?: string) {
  return useQuery({
    queryKey: ['lstm-data-files', workingDirectory],
    queryFn: () => api.lstm.getDataFiles(workingDirectory),
    enabled: !!workingDirectory,
  })
}

export function useLSTMParquets(workingDirectory?: string) {
  return useQuery({
    queryKey: ['lstm-parquets', workingDirectory],
    queryFn: () => api.lstm.getParquets(workingDirectory),
    enabled: !!workingDirectory,
  })
}

export function useCreateLSTMParquet() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (params: {
      pair: string
      start_date: string
      end_date: string
      working_directory?: string
    }) => api.lstm.createParquet(params),
    onSuccess: (_, variables) => {
      // Invalidate parquets list to show new file
      queryClient.invalidateQueries({ queryKey: ['lstm-parquets', variables.working_directory] })
    },
  })
}

export function useDeleteLSTMParquet() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (params: { filename: string; workingDirectory?: string }) =>
      api.lstm.deleteParquet(params.filename, params.workingDirectory),
    onSuccess: (_, variables) => {
      // Invalidate parquets list to remove deleted file
      queryClient.invalidateQueries({ queryKey: ['lstm-parquets', variables.workingDirectory] })
    },
  })
}

export function useDeleteLSTMParquetsBatch() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (params: { filenames: string[]; workingDirectory?: string }) =>
      api.lstm.deleteParquetsBatch({
        filenames: params.filenames,
        working_directory: params.workingDirectory,
      }),
    onSuccess: (_, variables) => {
      // Invalidate parquets list to remove deleted files
      queryClient.invalidateQueries({ queryKey: ['lstm-parquets', variables.workingDirectory] })
      // Also invalidate raw parquets for bridge
      queryClient.invalidateQueries({ queryKey: ['bridge-raw-parquets', variables.workingDirectory] })
    },
  })
}

export function useCreateFromFiles() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (params: {
      files: string[]
      working_directory?: string
      adr_period?: number
      timeframes?: string[]
    }) => api.lstm.createFromFiles(params),
    onSuccess: (_, variables) => {
      // Invalidate parquets list to show new files
      queryClient.invalidateQueries({ queryKey: ['lstm-parquets', variables.working_directory] })
      // Also invalidate raw parquets for bridge
      queryClient.invalidateQueries({ queryKey: ['bridge-raw-parquets', variables.working_directory] })
    },
  })
}

// ================================================================
// Bridge Hooks
// ================================================================

export function useRawParquetsForBridge(workingDirectory?: string) {
  return useQuery({
    queryKey: ['bridge-raw-parquets', workingDirectory],
    queryFn: () => api.lstm.getRawParquetsForBridge(workingDirectory),
    enabled: !!workingDirectory,
  })
}

export function useBridgedParquets(workingDirectory?: string) {
  return useQuery({
    queryKey: ['bridge-bridged-parquets', workingDirectory],
    queryFn: () => api.lstm.getBridgedParquets(workingDirectory),
    enabled: !!workingDirectory,
  })
}

export function useBridgeFiles() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (params: {
      filenames: string[]
      working_directory?: string
    }) => api.lstm.bridgeFiles(params),
    onSuccess: (_, variables) => {
      // Invalidate bridged parquets list to show new files
      queryClient.invalidateQueries({ queryKey: ['bridge-bridged-parquets', variables.working_directory] })
    },
  })
}

export function useDeleteBridgedParquet() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (params: { filename: string; workingDirectory?: string }) =>
      api.lstm.deleteBridgedParquet(params.filename, params.workingDirectory),
    onSuccess: (_, variables) => {
      // Invalidate bridged parquets list
      queryClient.invalidateQueries({ queryKey: ['bridge-bridged-parquets', variables.workingDirectory] })
    },
  })
}

