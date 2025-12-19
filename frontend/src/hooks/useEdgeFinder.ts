import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useEdgeFinderHealth(workingDirectory?: string) {
  return useQuery({
    queryKey: ['edge-finder', 'health', workingDirectory],
    queryFn: () => api.edgeFinder.health(workingDirectory),
    refetchInterval: 5000, // Poll every 5 seconds
  })
}

export function useEdgeFinderSessionStats(workingDirectory?: string) {
  return useQuery({
    queryKey: ['edge-finder', 'sessions', 'stats', workingDirectory],
    queryFn: () => api.edgeFinder.getSessionStats(workingDirectory),
  })
}

export function useEdgeFinderModels(workingDirectory?: string) {
  return useQuery({
    queryKey: ['edge-finder', 'models', workingDirectory],
    queryFn: () => api.edgeFinder.listModels(workingDirectory),
  })
}

export function useEdgeFinderIndexStatus(workingDirectory?: string) {
  return useQuery({
    queryKey: ['edge-finder', 'index', 'status', workingDirectory],
    queryFn: () => api.edgeFinder.getIndexStatus(workingDirectory),
    refetchInterval: 5000,
  })
}

export function useEdgeFinderTrainingStatus() {
  return useQuery({
    queryKey: ['edge-finder', 'training', 'status'],
    queryFn: () => api.edgeFinder.getTrainingStatus(),
    refetchInterval: 2000, // Poll more frequently during training
  })
}

export function useBuildIndex() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: api.edgeFinder.buildIndex,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'index'] })
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'health'] })
    },
  })
}

export function useLoadIndex() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: api.edgeFinder.loadIndex,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'index'] })
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'health'] })
    },
  })
}

export function useStartTraining() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: api.edgeFinder.startTraining,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'training'] })
    },
  })
}

export function useStopTraining() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => api.edgeFinder.stopTraining(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'training'] })
    },
  })
}

export function useGenerateSessions() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: api.edgeFinder.generateSessions,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'sessions'] })
    },
  })
}

export function useEdgeInference() {
  return useMutation({
    mutationFn: api.edgeFinder.runInference,
  })
}

export function useChartInference() {
  return useMutation({
    mutationFn: api.edgeFinder.runChartInference,
  })
}

// --- Auto Setup Hooks ---

export function useAutoSetupStatus(workingDirectory?: string) {
  return useQuery({
    queryKey: ['edge-finder', 'auto-setup', 'status', workingDirectory],
    queryFn: () => api.edgeFinder.getAutoSetupStatus(workingDirectory),
    refetchInterval: (query) => {
      // Poll more frequently during training
      const status = query.state.data?.status
      if (status === 'training') return 2000
      if (status === 'building_index') return 1000
      return 5000
    },
  })
}

export function useAutoSetup() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: api.edgeFinder.autoSetup,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'auto-setup'] })
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'index'] })
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'models'] })
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'health'] })
    },
  })
}

// --- Model Management Hooks ---

export function useRenameModel() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ modelName, newName, workingDirectory }: {
      modelName: string;
      newName: string;
      workingDirectory?: string
    }) => api.edgeFinder.renameModel(modelName, newName, workingDirectory),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'models'] })
      queryClient.invalidateQueries({ queryKey: ['files'] })
    },
  })
}

export function useCopyModel() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ modelName, newName, workingDirectory }: {
      modelName: string;
      newName: string;
      workingDirectory?: string
    }) => api.edgeFinder.copyModel(modelName, newName, workingDirectory),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'models'] })
      queryClient.invalidateQueries({ queryKey: ['files'] })
    },
  })
}

export function useDeleteModel() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ modelName, workingDirectory }: {
      modelName: string;
      workingDirectory?: string
    }) => api.edgeFinder.deleteModel(modelName, workingDirectory),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'models'] })
      queryClient.invalidateQueries({ queryKey: ['files'] })
    },
  })
}

export function useDeleteParquet() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ pair, timeframe, workingDirectory }: {
      pair: string;
      timeframe: string;
      workingDirectory?: string;
    }) => api.edgeFinder.deleteParquet(pair, timeframe, workingDirectory),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['files'] })
      queryClient.invalidateQueries({ queryKey: ['instruments'] })
    },
  })
}

export function useDeleteSessions() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ pair, session_type, timeframe, workingDirectory }: {
      pair: string;
      session_type: string;
      timeframe: string;
      workingDirectory?: string;
    }) => api.edgeFinder.deleteSessions({ pair, session_type, timeframe, workingDirectory }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['files'] })
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'sessions'] })
    },
  })
}

export function useDeleteIndex() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ indexName, workingDirectory }: {
      indexName: string;
      workingDirectory?: string;
    }) => api.edgeFinder.deleteIndex(indexName, workingDirectory),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['files'] })
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'index'] })
    },
  })
}

// --- Bulk Delete Hooks ---

export function useDeleteAllParquets() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (workingDirectory?: string) => api.edgeFinder.deleteAllParquets(workingDirectory),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['files'] })
      queryClient.invalidateQueries({ queryKey: ['instruments'] })
    },
  })
}

export function useDeleteAllSessions() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (workingDirectory?: string) => api.edgeFinder.deleteAllSessions(workingDirectory),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['files'] })
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'sessions'] })
    },
  })
}

export function useDeleteAllModels() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (workingDirectory?: string) => api.edgeFinder.deleteAllModels(workingDirectory),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['files'] })
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'models'] })
    },
  })
}

export function useDeleteAllIndices() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (workingDirectory?: string) => api.edgeFinder.deleteAllIndices(workingDirectory),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['files'] })
      queryClient.invalidateQueries({ queryKey: ['edge-finder', 'index'] })
    },
  })
}

export function useDeleteAllFiles() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (workingDirectory?: string) => api.edgeFinder.deleteAllFiles(workingDirectory),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['files'] })
      queryClient.invalidateQueries({ queryKey: ['instruments'] })
      queryClient.invalidateQueries({ queryKey: ['edge-finder'] })
    },
  })
}

// --- File Listing Hook ---

export function useFiles(workingDirectory?: string) {
  return useQuery({
    queryKey: ['files', workingDirectory],
    queryFn: () => api.edgeFinder.getFiles(workingDirectory),
  })
}

// --- Edge Mining Hook ---

export function useEdgeMining() {
  return useMutation({
    mutationFn: api.edgeFinder.mineSession,
  })
}
