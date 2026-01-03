export interface CandleData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface WaveData {
  id: number;
  level: number;
  direction: 'up' | 'down';
  start_time: string;
  end_time: string;
  start_price: number;
  end_price: number;
  color: string;
  parent_id: number | null;
  is_spline?: boolean;  // True for intermediate developing leg lines
}

// MMLC Debug Snapshot Types
export interface WaveSnapshotData {
  level: number;           // 1-5 (L1 through L5)
  direction: number;       // +1 (UP) or -1 (DOWN)
  amplitude: number;       // end_price - start_price (signed)
  duration_bars: number;   // Number of bars since wave started
  start_bar_index: number; // Bar index when this wave started
}

export interface StackSnapshotData {
  bar_index: number;
  timestamp: string;
  close_price: number;
  waves: WaveSnapshotData[];  // Active waves (L1 first, deepest last)
  l1_count: number;  // Cumulative L1 leg count
  l2_count: number;  // Cumulative L2 leg count
  l3_count: number;  // Cumulative L3 leg count
  l4_count: number;  // Cumulative L4 leg count
  l5_count: number;  // Cumulative L5 leg count
}

export interface ChartResponse {
  pair: string;
  timeframe: string;
  date: string;
  session: string;
  candles: CandleData[];
  waveform: WaveData[];
  debug?: string[];
  snapshot?: StackSnapshotData;  // MMLC state at bar_index (if provided)
}

export interface InstrumentInfo {
  pair: string;
  timeframes: string[];
  start_date: string;
  end_date: string;
  file_count: number;
}

export interface InstrumentsResponse {
  instruments: InstrumentInfo[];
}

export interface ProcessRequest {
  working_directory?: string;
  display_interval?: string;
}

export interface ProcessResponse {
  status: string;
  instruments_processed: string[];
  errors: string[];
  message: string;
}

export interface ProcessStatus {
  is_processing: boolean;
  progress: number;
  current_file: string | null;
  message: string;
}

export type SessionType = 'full_day' | 'asia' | 'london' | 'ny';
export type TimeframeType = 'M1' | 'M5' | 'M10' | 'M15' | 'M30' | 'H1' | 'H4';

// Edge Finder Types
export interface MatchDetail {
  session_id: string;
  bar_index: number;
  distance: number;
  next_bar_move: number;
  session_drift: number;
  mae: number;
  mfe: number;
  session_progress: number;
}

export interface EdgeProbabilities {
  num_matches: number;
  avg_distance: number;
  next_bar_up_pct: number;
  next_bar_avg_move: number;
  next_bar_std_move: number;
  session_up_pct: number;
  session_avg_drift: number;
  session_std_drift: number;
  avg_mae: number;
  mae_p25: number;
  mae_p50: number;
  mae_p75: number;
  mae_p95: number;
  avg_mfe: number;
  mfe_p25: number;
  mfe_p50: number;
  mfe_p75: number;
  mfe_p95: number;
  risk_reward_ratio: number;
  avg_session_progress: number;
  top_10_avg_distance: number;
  top_matches?: MatchDetail[];
}

export interface EdgeFinderSessionStats {
  total_sessions: number;
  by_pair: Record<string, number>;
  by_session_type: Record<string, number>;
  by_timeframe: Record<string, number>;
}

export interface EdgeFinderModelInfo {
  model_name: string;
  latent_dim: number;
  hidden_dim: number;
  num_layers: number;
  bidirectional: boolean;
  total_parameters: number;
  trained_epochs: number;
  best_loss: number;
}

export interface EdgeFinderIndexStatus {
  is_loaded: boolean;
  num_vectors: number;
  num_sessions: number;
  latent_dim: number;
  model_name: string | null;
}

export interface EdgeFinderTrainingStatus {
  is_training: boolean;
  epoch: number;
  total_epochs: number;
  train_loss: number;
  val_loss: number;
  best_loss: number;
  progress: number;
  message: string;
  model_name: string | null;
}

export interface EdgeFinderHealthStatus {
  status: string;
  training_active: boolean;
  index_loaded: boolean;
  index_vectors: number;
}

export interface EdgeFinderInferenceResponse {
  status: string;
  edge: EdgeProbabilities | null;
  message: string;
}

export interface EdgeFinderBuildIndexResponse {
  status: string;
  num_vectors: number;
  num_sessions: number;
  message: string;
}

// Combined Pipeline Types
export interface ProcessAllRequest {
  working_directory?: string;
  pairs?: string[];
  session_type?: string;
  timeframe?: string;
  force_sessions?: boolean;
  max_sessions?: number;
}

export interface ProcessAllResponse {
  status: string;
  parquets_created: number;
  sessions_generated: number;
  sessions_skipped: number;
  total_time_seconds: number;
  errors: string[];
  message: string;
}

export interface ProcessAllStatus {
  is_processing: boolean;
  stage: string;
  progress: number;
  current_pair: string | null;
  pairs_completed: number;
  pairs_total: number;
  message: string;
}

export interface AvailablePairsResponse {
  pairs: string[];
  csv_counts: Record<string, number>;
}

// Auto Setup Types
export interface AutoSetupRequest {
  working_directory?: string;
  model_name?: string;
  force_retrain?: boolean;
  force_rebuild_index?: boolean;
  num_epochs?: number;
  latent_dim?: number;
  pair?: string;
  session_type?: string;
  timeframe?: string;
}

export interface AutoSetupStatus {
  status: string;  // "ready", "checking", "training", "building_index", "error"
  model_exists: boolean;
  model_name: string | null;
  index_loaded: boolean;
  num_vectors: number;
  num_sessions: number;
  message: string;
  // Training progress (if training)
  training_epoch: number;
  training_total_epochs: number;
  training_loss: number;
  // Index building progress (if building index)
  index_current_session: number;
  index_total_sessions: number;
  index_progress: number;
  // Training completion info (shown after training completes)
  last_training_completed: boolean;
  last_training_best_loss: number;
  last_training_epochs: number;
}

export interface IndexBuildingStatus {
  is_building: boolean;
  current_session: number;
  total_sessions: number;
  progress: number;
  message: string;
}

export interface GenerationStatus {
  is_generating: boolean;
  current_session: string | null;
  progress: number;
  message: string;
}

// File Listing Types
export interface ParquetFileInfo {
  pair: string;
  timeframe: string;
  file_name: string;
  size_mb: number;
}

export interface ModelSummary {
  model_name: string;
  latent_dim: number;
  trained_epochs: number;
  best_loss: number;
  is_active: boolean;
}

export interface IndexSummary {
  index_name: string;
  num_vectors: number;
  model_name: string | null;
}

export interface FileListResponse {
  parquets: ParquetFileInfo[];
  sessions_total: number;
  sessions_by_pair: Record<string, number>;
  sessions_by_session_type: Record<string, number>;
  sessions_by_timeframe: Record<string, number>;
  models: ModelSummary[];
  indices: IndexSummary[];
}

export interface ModelActionResponse {
  success: boolean;
  message: string;
}

// Edge Mining Types
export interface BarEdgeData {
  bar_index: number;
  next_bar_win_rate: number;
  next_bar_avg_move: number;
  next_bar_edge_score: number;
  session_bias: 'long' | 'short';
  session_win_rate: number;
  session_avg_mfe: number;
  session_avg_mae: number;
  session_risk_reward: number;
  session_edge_score: number;
  num_matches: number;
  avg_distance: number;
}

export interface EdgeGraphDataPoint {
  bar_index: number;
  session_score: number;
  next_bar_score: number;
}

export interface MineSessionResponse {
  status: string;
  graph_data: EdgeGraphDataPoint[];
  edge_table: BarEdgeData[];
  message: string;
}

// MMLC Dev Types
export interface DevSessionResponse {
  pair: string;
  date: string;
  session: string;
  timeframe: string;
  candles: CandleData[];
  total_bars: number;
}

export interface StitchAnnotation {
  bar: number;
  timestamp: string;
  price: number;
  level: number;
  is_high: boolean;
  text: string;
}

export interface SwingLabel {
  bar: number;
  timestamp: string;
  price: number;
  is_high: boolean;
  child_level: number;
  child_price: number;
  bars_ago: number;
}

export interface DebugSwingPoint {
  bar: number;
  price: number;
}

export interface DebugSplineSegment {
  start_bar: number;
  start_price: number;
  end_bar: number;
  end_price: number;
}

export interface DebugStitchLeg {
  start_bar: number;
  start_price: number;
  end_bar: number;
  end_price: number;
  direction: string;
}

export interface DebugCandle {
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface DebugLevelData {
  level: number;
  direction: string;
  high: number;
  high_bar: number;
  low: number;
  low_bar: number;
  origin_bar: number;
  origin_price: number;
  completed_waves: DebugSplineSegment[];
  spline_segments: DebugSplineSegment[];
  swing_points?: DebugSwingPoint[];  // Only for L1
  // Per-bar indexed arrays
  leg_history: number[];       // leg at bar i
  count_history: number[];     // count at bar i
  direction_history: number[]; // direction at bar i (+1/-1/0)
  // Current counters
  current_leg: number;
  current_count: number;
}

export interface DebugStitchSwing {
  bar: number;
  price: number;
  direction: number;  // +1 = UP, -1 = DOWN
}

export interface DebugState {
  mode: string;
  end_bar: number;
  current_candle: DebugCandle | null;
  levels: DebugLevelData[];
  stitch_permanent_legs: DebugStitchLeg[];
  stitch_swings: DebugStitchSwing[];
  prev_L1_Direction: string;
  num_waves_returned: number;
  L1_count: number;  // Number of extremes in current L1 direction
  L1_leg: number;  // Overall L1 swing number (never resets)
}

export interface DevRunResponse {
  waves: WaveData[];
  start_bar: number;
  end_bar: number;
  bars_processed: number;
  annotations: StitchAnnotation[];
  swing_labels: SwingLabel[];
  debug_state?: DebugState;
}
