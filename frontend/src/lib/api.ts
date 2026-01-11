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
  DevSessionResponse,
  DevRunResponse,
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

  lstm: {
    getDataFiles: (workingDirectory?: string): Promise<{
      files: Array<{ name: string; pair: string | null; year: number | null; type: string; size_mb: number }>
    }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/lstm/data-files${params}`);
    },

    getParquets: (workingDirectory?: string): Promise<{
      parquets: Array<{
        name: string;
        pair: string | null;
        start_date: string | null;
        end_date: string | null;
        timeframe: string | null;
        adr_period: number | null;
        size_mb: number;
        rows: number;
      }>
    }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/lstm/parquets${params}`);
    },

    createParquet: (params: {
      pair: string;
      start_date: string;
      end_date: string;
      working_directory?: string;
    }): Promise<{
      status: string;
      message: string;
      filename: string | null;
      rows: number;
      size_mb: number;
    }> =>
      fetchApi('/lstm/create-parquet', {
        method: 'POST',
        body: JSON.stringify({
          pair: params.pair,
          start_date: params.start_date,
          end_date: params.end_date,
          working_directory: params.working_directory,
        }),
      }),

    deleteParquet: (filename: string, workingDirectory?: string): Promise<{
      status: string;
      message: string;
    }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/lstm/parquet/${filename}${params}`, { method: 'DELETE' });
    },

    deleteParquetsBatch: (params: {
      filenames: string[];
      working_directory?: string;
    }): Promise<{
      status: string;
      message: string;
      deleted: string[];
      errors: string[];
    }> =>
      fetchApi('/lstm/parquets/delete-batch', {
        method: 'POST',
        body: JSON.stringify({
          filenames: params.filenames,
          working_directory: params.working_directory,
        }),
      }),

    createFromFiles: (params: {
      files: string[];
      working_directory?: string;
      adr_period?: number;
      timeframes?: string[];
    }): Promise<{
      status: string;
      message: string;
      created: Array<{
        pair: string;
        filename: string;
        rows: number;
        size_mb: number;
        start_date: string;
        end_date: string;
        adr_period: number;
        timeframe: string;
        trimmed_days: number;
      }>;
      errors: string[];
    }> =>
      fetchApi('/lstm/create-from-files', {
        method: 'POST',
        body: JSON.stringify({
          files: params.files,
          working_directory: params.working_directory,
          adr_period: params.adr_period ?? 20,
          timeframes: params.timeframes ?? ['M5'],
        }),
      }),

    // Bridge endpoints
    getRawParquetsForBridge: (workingDirectory?: string): Promise<{
      parquets: Array<{
        name: string;
        pair: string | null;
        start_date: string | null;
        end_date: string | null;
        timeframe: string | null;
        adr_period: number | null;
        size_mb: number;
        rows: number;
      }>
    }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/lstm/bridge/raw-parquets${params}`);
    },

    getBridgedParquets: (workingDirectory?: string): Promise<{
      parquets: Array<{
        name: string;
        pair: string | null;
        start_date: string | null;
        end_date: string | null;
        timeframe: string | null;
        adr_period: number | null;
        size_mb: number;
        rows: number;
      }>
    }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/lstm/bridge/bridged-parquets${params}`);
    },

    bridgeFiles: (params: {
      filenames: string[];
      working_directory?: string;
    }): Promise<{
      status: string;
      message: string;
      results: Array<{
        status: string;
        message: string;
        output_filename: string | null;
        rows: number;
        days_processed: number;
        processing_time_seconds: number;
      }>;
      errors: string[];
    }> =>
      fetchApi('/lstm/bridge/batch', {
        method: 'POST',
        body: JSON.stringify({
          filenames: params.filenames,
          working_directory: params.working_directory,
        }),
      }),

    deleteBridgedParquet: (filename: string, workingDirectory?: string): Promise<{
      status: string;
      message: string;
    }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/lstm/bridge/bridged/${filename}${params}`, { method: 'DELETE' });
    },
  },

  transformer: {
    // Configuration defaults
    getConfigDefaults: (params: {
      session?: string;
      timeframe?: string;
      combine_sessions?: string;
    }): Promise<{
      sequence_length: number;
      session_hours: number;
      timeframe_minutes: number;
      available_sessions: string[];
      available_timeframes: string[];
    }> => {
      const searchParams = new URLSearchParams();
      if (params.session) searchParams.set('session', params.session);
      if (params.timeframe) searchParams.set('timeframe', params.timeframe);
      if (params.combine_sessions) searchParams.set('combine_sessions', params.combine_sessions);
      return fetchApi(`/transformer/config-defaults?${searchParams.toString()}`);
    },

    // Training status
    getStatus: (): Promise<{
      status: string;
      current_epoch: number;
      total_epochs: number;
      train_loss: number | null;
      val_loss: number | null;
      best_loss: number | null;
      epochs_without_improvement: number;
      learning_rate: number | null;
      model_name: string | null;
      message: string | null;
      // New fields
      elapsed_seconds: number;
      directional_accuracy: number | null;
      r_squared: number | null;
      max_error: number | null;
      grad_norm: number | null;
    }> => fetchApi('/transformer/status'),

    // Start training
    startTraining: (params: {
      target_session?: string;
      combine_sessions?: string;
      sequence_length?: number;
      batch_size?: number;
      d_model?: number;
      n_layers?: number;
      n_heads?: number;
      dropout_rate?: number;
      learning_rate?: number;
      num_epochs?: number;
      early_stopping_patience?: number;
      model_name?: string;
      save_to_models_folder?: boolean;
      working_directory?: string;
    }): Promise<{
      status: string;
      message: string;
      model_name: string | null;
    }> =>
      fetchApi('/transformer/start-training', {
        method: 'POST',
        body: JSON.stringify({
          target_session: params.target_session ?? 'lon',
          combine_sessions: params.combine_sessions,
          sequence_length: params.sequence_length ?? 64,
          batch_size: params.batch_size ?? 32,
          d_model: params.d_model ?? 128,
          n_layers: params.n_layers ?? 4,
          n_heads: params.n_heads ?? 4,
          dropout_rate: params.dropout_rate ?? 0.1,
          learning_rate: params.learning_rate ?? 0.0001,
          num_epochs: params.num_epochs ?? 100,
          early_stopping_patience: params.early_stopping_patience ?? 15,
          model_name: params.model_name,
          save_to_models_folder: params.save_to_models_folder ?? false,
          working_directory: params.working_directory,
        }),
      }),

    // Stop training
    stopTraining: (): Promise<{
      status: string;
      message: string;
    }> => fetchApi('/transformer/stop-training', { method: 'POST' }),

    // List models
    getModels: (workingDirectory?: string): Promise<{
      models: Array<{
        name: string;
        best_loss: number | null;
        epochs_trained: number;
        target_session: string | null;
        sequence_length: number | null;
        d_model: number | null;
        n_layers: number | null;
        created_at: string | null;
      }>;
    }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/transformer/models${params}`);
    },

    // Delete model
    deleteModel: (modelName: string, workingDirectory?: string): Promise<{
      status: string;
      message: string;
    }> =>
      fetchApi('/transformer/delete-model', {
        method: 'POST',
        body: JSON.stringify({
          model_name: modelName,
          working_directory: workingDirectory,
        }),
      }),

    // Delete report
    deleteReport: (filename: string, workingDirectory?: string): Promise<{
      status: string;
      message: string;
    }> =>
      fetchApi('/transformer/delete-report', {
        method: 'POST',
        body: JSON.stringify({
          filename: filename,
          working_directory: workingDirectory,
        }),
      }),

    // Delete multiple reports
    deleteReportsBulk: (filenames: string[], workingDirectory?: string): Promise<{
      status: string;
      message: string;
      deleted: number;
    }> =>
      fetchApi('/transformer/delete-reports-bulk', {
        method: 'POST',
        body: JSON.stringify({
          filenames: filenames,
          working_directory: workingDirectory,
        }),
      }),

    // Delete multiple models
    deleteModelsBulk: (modelNames: string[], workingDirectory?: string): Promise<{
      status: string;
      message: string;
      deleted: number;
    }> =>
      fetchApi('/transformer/delete-models-bulk', {
        method: 'POST',
        body: JSON.stringify({
          model_names: modelNames,
          working_directory: workingDirectory,
        }),
      }),

    // Parquet viewer endpoints
    getParquetFiles: (workingDirectory?: string): Promise<{
      files: Array<{
        name: string;
        path: string;
        rows: number;
        size_mb: number;
        has_mmlc_columns: boolean;
      }>;
    }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/transformer/parquet-files${params}`);
    },

    getParquetDates: (params: {
      filename: string;
      working_directory?: string;
    }): Promise<{
      filename: string;
      dates: string[];
      total_dates: number;
    }> => {
      const searchParams = new URLSearchParams();
      searchParams.set('filename', params.filename);
      if (params.working_directory) searchParams.set('working_directory', params.working_directory);
      return fetchApi(`/transformer/parquet-dates?${searchParams.toString()}`);
    },

    getParquetData: (params: {
      filename: string;
      working_directory?: string;
      session?: string;
      start_idx?: number;
      limit?: number;
      date?: string;
    }): Promise<{
      candles: Array<{
        timestamp: string;
        open: number;
        high: number;
        low: number;
        close: number;
      }>;
      states: Array<{
        level: number | null;
        event: string | null;
        dir: string | null;
        out_max_up: number | null;
        out_max_down: number | null;
      }>;
      total_rows: number;
      start_idx: number;
      session: string;
      date: string | null;
    }> => {
      const searchParams = new URLSearchParams();
      searchParams.set('filename', params.filename);
      if (params.working_directory) searchParams.set('working_directory', params.working_directory);
      if (params.session) searchParams.set('session', params.session);
      if (params.start_idx !== undefined) searchParams.set('start_idx', String(params.start_idx));
      if (params.limit !== undefined) searchParams.set('limit', String(params.limit));
      if (params.date) searchParams.set('date', params.date);
      return fetchApi(`/transformer/parquet-data?${searchParams.toString()}`);
    },

    // Queue endpoints
    getQueueStatus: (): Promise<{
      queue_running: boolean;
      current_card_id: string | null;
      cards: Array<{
        card_id: string;
        status: 'pending' | 'training' | 'completed' | 'error';
        model_name: string;
        current_epoch: number;
        total_epochs: number;
        train_loss: number | null;
        val_loss: number | null;
        best_loss: number | null;
        elapsed_seconds: number;
        directional_accuracy: number | null;
        r_squared: number | null;
        max_error: number | null;
        grad_norm: number | null;
        error_message: string | null;
        final_directional_accuracy: number | null;
        final_r_squared: number | null;
        final_max_error: number | null;
      }>;
    }> => fetchApi('/transformer/queue/status'),

    addToQueue: (params: {
      card_id: string;
      model_name: string;
      parquet_file: string;
      session_option: string;
      target_outcome?: string;
      sequence_length?: number;
      batch_size?: number;
      d_model?: number;
      n_layers?: number;
      n_heads?: number;
      dropout_rate?: number;
      learning_rate?: number;
      num_epochs?: number;
      early_stopping_patience?: number;
      save_to_models_folder?: boolean;
      working_directory?: string;
    }): Promise<{
      status: string;
      card_id: string;
      message: string | null;
    }> =>
      fetchApi('/transformer/queue/add', {
        method: 'POST',
        body: JSON.stringify({
          config: {
            card_id: params.card_id,
            model_name: params.model_name,
            parquet_file: params.parquet_file,
            session_option: params.session_option,
            target_outcome: params.target_outcome ?? 'max_up',
            sequence_length: params.sequence_length ?? 64,
            batch_size: params.batch_size ?? 32,
            d_model: params.d_model ?? 128,
            n_layers: params.n_layers ?? 4,
            n_heads: params.n_heads ?? 4,
            dropout_rate: params.dropout_rate ?? 0.1,
            learning_rate: params.learning_rate ?? 0.0001,
            num_epochs: params.num_epochs ?? 100,
            early_stopping_patience: params.early_stopping_patience ?? 15,
            save_to_models_folder: params.save_to_models_folder ?? true,
          },
          working_directory: params.working_directory,
        }),
      }),

    removeFromQueue: (cardId: string): Promise<{
      status: string;
      card_id: string | null;
      message: string | null;
    }> =>
      fetchApi('/transformer/queue/remove', {
        method: 'POST',
        body: JSON.stringify({ card_id: cardId }),
      }),

    startQueue: (workingDirectory?: string): Promise<{
      status: string;
      message: string | null;
    }> =>
      fetchApi('/transformer/queue/start', {
        method: 'POST',
        body: JSON.stringify({ working_directory: workingDirectory }),
      }),

    stopQueue: (): Promise<{
      status: string;
      message: string | null;
    }> => fetchApi('/transformer/queue/stop', { method: 'POST' }),

    clearQueue: (): Promise<{
      status: string;
      message: string;
    }> => fetchApi('/transformer/queue/clear', { method: 'POST' }),

    // Validation data generation
    generateValidationFromSource: (params: {
      source_parquet: string;
      test_type: 'sanity' | 'memory' | 'logic' | 'next' | 'next5' | 'close' | 'max_up' | 'max_down';
      target_session?: string;
      timeframe?: string;
      adr_value?: number;
      seed?: number;
      working_directory?: string;
    }): Promise<{
      status: string;
      message: string;
      output_file: string | null;
      rows: number;
      sessions_found: number;
    }> =>
      fetchApi('/transformer/validation/generate-from-source', {
        method: 'POST',
        body: JSON.stringify({
          source_parquet: params.source_parquet,
          test_type: params.test_type,
          target_session: params.target_session ?? 'lon',
          timeframe: params.timeframe ?? 'M5',
          adr_value: params.adr_value ?? 0.005,
          seed: params.seed ?? 42,
          working_directory: params.working_directory,
        }),
      }),

    // Reports endpoints
    getReports: (workingDirectory?: string): Promise<{
      reports: Array<{
        filename: string;
        model_name: string;
        timestamp: string;
        directional_accuracy: number;
        r_squared: number;
        best_loss: number;
        total_epochs: number;
        target_session: string | null;
        elapsed_seconds: number | null;
      }>;
    }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/transformer/reports${params}`);
    },

    getReportDetail: (filename: string, workingDirectory?: string): Promise<{
      report: {
        filename: string;
        model_name: string;
        timestamp: string;
        config: Record<string, unknown>;
        summary: Record<string, unknown>;
        epochs: Array<{
          epoch: number;
          train_loss: number;
          val_loss: number | null;
          directional_accuracy: number | null;
          r_squared: number | null;
          max_error: number | null;
          grad_norm: number | null;
        }>;
      };
    }> => {
      const params = workingDirectory
        ? `?working_directory=${encodeURIComponent(workingDirectory)}`
        : '';
      return fetchApi(`/transformer/reports/${encodeURIComponent(filename)}${params}`);
    },
  },

  mmlcDev: {
    loadSession: (params: {
      pair: string;
      date: string;
      session: SessionType;
      timeframe: TimeframeType;
      workingDirectory?: string;
    }): Promise<DevSessionResponse> => {
      const searchParams = new URLSearchParams({
        date: params.date,
        session: params.session,
        timeframe: params.timeframe,
      });
      if (params.workingDirectory) {
        searchParams.set('working_directory', params.workingDirectory);
      }
      return fetchApi(`/mmlc-dev/session/${params.pair}?${searchParams.toString()}`);
    },

    run: (params: {
      pair: string;
      date: string;
      session: SessionType;
      timeframe: TimeframeType;
      startBar?: number;
      endBar?: number;
      mode?: 'complete' | 'spline' | 'stitch';
      workingDirectory?: string;
    }): Promise<DevRunResponse> => {
      const searchParams = new URLSearchParams({
        date: params.date,
        session: params.session,
        timeframe: params.timeframe,
      });
      if (params.startBar !== undefined) {
        searchParams.set('start_bar', String(params.startBar));
      }
      if (params.endBar !== undefined) {
        searchParams.set('end_bar', String(params.endBar));
      }
      if (params.mode) {
        searchParams.set('mode', params.mode);
      }
      if (params.workingDirectory) {
        searchParams.set('working_directory', params.workingDirectory);
      }
      return fetchApi(`/mmlc-dev/run/${params.pair}?${searchParams.toString()}`);
    },
  },
};
