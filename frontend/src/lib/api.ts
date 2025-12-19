import { API_BASE_URL } from './constants';
import type {
  ChartResponse,
  InstrumentsResponse,
  ProcessRequest,
  ProcessResponse,
  ProcessStatus,
  SessionType,
  TimeframeType,
  EdgeFinderSessionStats,
  EdgeFinderModelInfo,
  EdgeFinderIndexStatus,
  EdgeFinderTrainingStatus,
  EdgeFinderHealthStatus,
  EdgeFinderInferenceResponse,
  EdgeFinderBuildIndexResponse,
  ProcessAllRequest,
  ProcessAllResponse,
  ProcessAllStatus,
  AvailablePairsResponse,
  AutoSetupRequest,
  AutoSetupStatus,
  FileListResponse,
  ModelActionResponse,
  MineSessionResponse,
} from '@/types';

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP error ${response.status}`);
  }

  return response.json();
}

export const api = {
  pipeline: {
    process: (request: ProcessRequest): Promise<ProcessResponse> =>
      fetchApi('/pipeline/process', {
        method: 'POST',
        body: JSON.stringify(request),
      }),

    getStatus: (): Promise<ProcessStatus> =>
      fetchApi('/pipeline/status'),

    // Combined pipeline endpoints
    getAvailablePairs: (workingDirectory?: string): Promise<AvailablePairsResponse> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/pipeline/available-pairs${params}`);
    },

    processAll: (request: ProcessAllRequest): Promise<ProcessAllResponse> =>
      fetchApi('/pipeline/process-all', {
        method: 'POST',
        body: JSON.stringify(request),
      }),

    getProcessAllStatus: (): Promise<ProcessAllStatus> =>
      fetchApi('/pipeline/process-all/status'),
  },

  instruments: {
    list: (workingDirectory?: string): Promise<InstrumentsResponse> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/instruments${params}`);
    },

    getDates: (pair: string, workingDirectory?: string): Promise<{ pair: string; dates: string[] }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/instruments/${pair}/dates${params}`);
    },
  },

  chart: {
    getData: (
      pair: string,
      params: {
        timeframe: TimeframeType;
        date: string;
        session: SessionType;
        workingDirectory?: string;
        barIndex?: number;
      }
    ): Promise<ChartResponse> => {
      const searchParams = new URLSearchParams({
        timeframe: params.timeframe,
        date: params.date,
        session: params.session,
      });
      if (params.workingDirectory) {
        searchParams.set('working_directory', params.workingDirectory);
      }
      if (params.barIndex !== undefined) {
        searchParams.set('bar_index', String(params.barIndex));
      }
      return fetchApi(`/chart/${pair}?${searchParams.toString()}`);
    },
  },

  edgeFinder: {
    // Health & Status
    health: (workingDirectory?: string): Promise<EdgeFinderHealthStatus> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/edge-finder/health${params}`);
    },

    // Sessions
    getSessionStats: (workingDirectory?: string): Promise<EdgeFinderSessionStats> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/edge-finder/sessions/stats${params}`);
    },

    listSessions: (
      workingDirectory?: string,
      filters?: { pair?: string; session_type?: string; timeframe?: string; limit?: number }
    ): Promise<{ total: number; sessions: string[] }> => {
      const searchParams = new URLSearchParams();
      if (workingDirectory) searchParams.set('working_directory', workingDirectory);
      if (filters?.pair) searchParams.set('pair', filters.pair);
      if (filters?.session_type) searchParams.set('session_type', filters.session_type);
      if (filters?.timeframe) searchParams.set('timeframe', filters.timeframe);
      if (filters?.limit) searchParams.set('limit', String(filters.limit));
      return fetchApi(`/edge-finder/sessions/list?${searchParams.toString()}`);
    },

    generateSessions: (params: {
      workingDirectory?: string;
      pair?: string;
      session_type?: string;
      timeframe?: string;
      max_sessions?: number;
    }): Promise<{ status: string; sessions_generated: number; errors: string[]; message: string }> =>
      fetchApi('/edge-finder/sessions/generate', {
        method: 'POST',
        body: JSON.stringify({
          working_directory: params.workingDirectory,
          pair: params.pair,
          session_type: params.session_type,
          timeframe: params.timeframe,
          max_sessions: params.max_sessions,
        }),
      }),

    // Models
    listModels: (workingDirectory?: string): Promise<{ models: EdgeFinderModelInfo[] }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/edge-finder/models${params}`);
    },

    getModelInfo: (modelName: string, workingDirectory?: string): Promise<EdgeFinderModelInfo> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/edge-finder/models/${modelName}${params}`);
    },

    // Training
    getTrainingStatus: (): Promise<EdgeFinderTrainingStatus> =>
      fetchApi('/edge-finder/training/status'),

    startTraining: (params: {
      workingDirectory?: string;
      model_name?: string;
      num_epochs?: number;
      batch_size?: number;
      learning_rate?: number;
      latent_dim?: number;
    }): Promise<{ status: string; message: string }> =>
      fetchApi('/edge-finder/training/start', {
        method: 'POST',
        body: JSON.stringify({
          working_directory: params.workingDirectory,
          model_name: params.model_name || 'vae_default',
          num_epochs: params.num_epochs || 100,
          batch_size: params.batch_size || 32,
          learning_rate: params.learning_rate || 0.001,
          latent_dim: params.latent_dim || 32,
        }),
      }),

    stopTraining: (): Promise<{ status: string; message: string }> =>
      fetchApi('/edge-finder/training/stop', { method: 'POST' }),

    // Index
    getIndexStatus: (workingDirectory?: string): Promise<EdgeFinderIndexStatus> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/edge-finder/index/status${params}`);
    },

    buildIndex: (params: {
      workingDirectory?: string;
      model_name?: string;
      pair?: string;
      session_type?: string;
      timeframe?: string;
      save_index?: boolean;
    }): Promise<EdgeFinderBuildIndexResponse> =>
      fetchApi('/edge-finder/index/build', {
        method: 'POST',
        body: JSON.stringify({
          working_directory: params.workingDirectory,
          model_name: params.model_name || 'vae_test',
          pair: params.pair,
          session_type: params.session_type,
          timeframe: params.timeframe,
          save_index: params.save_index ?? true,
        }),
      }),

    loadIndex: (params: {
      workingDirectory?: string;
      model_name?: string;
      index_name?: string;
    }): Promise<EdgeFinderBuildIndexResponse> => {
      const searchParams = new URLSearchParams();
      if (params.workingDirectory) searchParams.set('working_directory', params.workingDirectory);
      if (params.model_name) searchParams.set('model_name', params.model_name);
      if (params.index_name) searchParams.set('index_name', params.index_name);
      return fetchApi(`/edge-finder/index/load?${searchParams.toString()}`, { method: 'POST' });
    },

    // Inference
    runInference: (params: {
      matrix: number[][];
      k_neighbors?: number;
      unique_sessions?: boolean;
    }): Promise<EdgeFinderInferenceResponse> =>
      fetchApi('/edge-finder/inference', {
        method: 'POST',
        body: JSON.stringify({
          matrix: params.matrix,
          k_neighbors: params.k_neighbors || 500,
          unique_sessions: params.unique_sessions ?? true,
        }),
      }),

    // Chart Inference - runs inference on a stored session by chart identifiers
    runChartInference: (params: {
      pair: string;
      date: string;
      session: string;
      timeframe: string;
      workingDirectory?: string;
      bar_index?: number;
      k_neighbors?: number;
      unique_sessions?: boolean;
    }): Promise<EdgeFinderInferenceResponse> =>
      fetchApi('/edge-finder/inference/chart', {
        method: 'POST',
        body: JSON.stringify({
          pair: params.pair,
          date: params.date,
          session: params.session,
          timeframe: params.timeframe,
          working_directory: params.workingDirectory,
          bar_index: params.bar_index,
          k_neighbors: params.k_neighbors || 500,
          unique_sessions: params.unique_sessions ?? true,
        }),
      }),

    // Auto Setup
    autoSetup: (request: AutoSetupRequest): Promise<AutoSetupStatus> =>
      fetchApi('/edge-finder/auto-setup', {
        method: 'POST',
        body: JSON.stringify({
          working_directory: request.working_directory,
          model_name: request.model_name || 'vae_default',
          force_retrain: request.force_retrain || false,
          force_rebuild_index: request.force_rebuild_index || false,
          num_epochs: request.num_epochs || 100,
          latent_dim: request.latent_dim || 32,
          pair: request.pair,
          session_type: request.session_type,
          timeframe: request.timeframe,
        }),
      }),

    getAutoSetupStatus: (workingDirectory?: string): Promise<AutoSetupStatus> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/edge-finder/auto-setup/status${params}`);
    },

    // Model Management
    renameModel: (modelName: string, newName: string, workingDirectory?: string): Promise<ModelActionResponse> =>
      fetchApi(`/edge-finder/models/${modelName}/rename`, {
        method: 'POST',
        body: JSON.stringify({
          working_directory: workingDirectory,
          new_name: newName,
        }),
      }),

    copyModel: (modelName: string, newName: string, workingDirectory?: string): Promise<ModelActionResponse> =>
      fetchApi(`/edge-finder/models/${modelName}/copy`, {
        method: 'POST',
        body: JSON.stringify({
          working_directory: workingDirectory,
          new_name: newName,
        }),
      }),

    deleteModel: (modelName: string, workingDirectory?: string): Promise<ModelActionResponse> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/edge-finder/models/${modelName}${params}`, { method: 'DELETE' });
    },

    deleteParquet: (pair: string, timeframe: string, workingDirectory?: string): Promise<ModelActionResponse> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/edge-finder/parquets/${pair}/${timeframe}${params}`, { method: 'DELETE' });
    },

    deleteSessions: (params: {
      pair: string;
      session_type: string;
      timeframe: string;
      workingDirectory?: string;
    }): Promise<{ success: boolean; deleted_count: number; message: string }> => {
      const searchParams = new URLSearchParams({
        pair: params.pair,
        session_type: params.session_type,
        timeframe: params.timeframe,
      });
      if (params.workingDirectory) {
        searchParams.set('working_directory', params.workingDirectory);
      }
      return fetchApi(`/edge-finder/sessions?${searchParams.toString()}`, { method: 'DELETE' });
    },

    deleteIndex: (indexName: string, workingDirectory?: string): Promise<ModelActionResponse> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/edge-finder/indices/${indexName}${params}`, { method: 'DELETE' });
    },

    // Bulk Delete
    deleteAllParquets: (workingDirectory?: string): Promise<{ success: boolean; deleted_count: number; message: string }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/edge-finder/parquets/all${params}`, { method: 'DELETE' });
    },

    deleteAllSessions: (workingDirectory?: string): Promise<{ success: boolean; deleted_count: number; message: string }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/edge-finder/sessions/all${params}`, { method: 'DELETE' });
    },

    deleteAllModels: (workingDirectory?: string): Promise<{ success: boolean; deleted_count: number; message: string }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/edge-finder/models/all${params}`, { method: 'DELETE' });
    },

    deleteAllIndices: (workingDirectory?: string): Promise<{ success: boolean; deleted_count: number; message: string }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/edge-finder/indices/all${params}`, { method: 'DELETE' });
    },

    deleteAllFiles: (workingDirectory?: string): Promise<{ success: boolean; deleted_count: number; message: string }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/edge-finder/all${params}`, { method: 'DELETE' });
    },

    // File Listing
    getFiles: (workingDirectory?: string): Promise<FileListResponse> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/edge-finder/files${params}`);
    },

    // Edge Mining
    mineSession: (params: {
      pair: string;
      date: string;
      session: string;
      timeframe: string;
      workingDirectory?: string;
      k_neighbors?: number;
    }): Promise<MineSessionResponse> =>
      fetchApi('/edge-finder/mining/session', {
        method: 'POST',
        body: JSON.stringify({
          pair: params.pair,
          date: params.date,
          session: params.session,
          timeframe: params.timeframe,
          working_directory: params.workingDirectory,
          k_neighbors: params.k_neighbors || 50,
        }),
      }),
  },
};
