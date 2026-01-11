import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { api } from '@/lib/api'
import {
  useTransformerModels,
  useDeleteTransformerModel,
  useDeleteTransformerReport,
  useDeleteTransformerReportsBulk,
  useDeleteTransformerModelsBulk,
  useQueueStatus,
  useAddToQueue,
  useStartQueue,
  useStopQueue,
  useClearQueue,
  useParquetFiles,
  useTransformerReports,
  useTransformerReportDetail,
} from '@/hooks/useTransformer'
import { usePersistedState } from '@/hooks/usePersistedSettings'
import {
  Loader2,
  Zap,
  CheckCircle2,
  AlertCircle,
  XCircle,
  Play,
  Square,
  CheckSquare,
  ChevronDown,
  ChevronRight,
  Trash2,
  Layers,
  Activity,
  Settings2,
  Database,
  Plus,
  Timer,
  FileText,
  TrendingUp,
  BarChart3,
  ChevronsDownUp,
  ChevronsUpDown,
  LineChart,
  ExternalLink,
} from 'lucide-react'

interface TransformerTabProps {
  workingDirectory: string
}

// Unified session options (6 total)
type SessionOption = 'asia' | 'lon' | 'ny' | 'day' | 'asia_lon' | 'lon_ny'

const SESSION_OPTIONS: { value: SessionOption; label: string }[] = [
  { value: 'asia', label: 'Asia' },
  { value: 'lon', label: 'London' },
  { value: 'ny', label: 'New York' },
  { value: 'day', label: 'Full Day' },
  { value: 'asia_lon', label: 'Asia+Lon' },
  { value: 'lon_ny', label: 'Lon+NY' },
]

// Target outcome options
type TargetOutcome = 'max_up' | 'max_down' | 'next' | 'next5' | 'sess' | 'validation'

const TARGET_OUTCOME_OPTIONS: { value: TargetOutcome; label: string; description: string }[] = [
  { value: 'max_up', label: 'Max Up', description: 'Maximum upward movement during session' },
  { value: 'max_down', label: 'Max Down', description: 'Maximum downward movement during session' },
  { value: 'next', label: 'Next Bar', description: 'Next bar price movement' },
  { value: 'next5', label: 'Next 5 Bars', description: 'Price movement 5 bars ahead' },
  { value: 'sess', label: 'Session Close', description: 'Session close price movement' },
  { value: 'validation', label: 'Validation', description: 'Use for validation parquets (uses max_up)' },
]

// Session durations in hours
const SESSION_HOURS: Record<SessionOption, number> = {
  asia: 9,
  lon: 9,
  ny: 9,
  day: 22,
  asia_lon: 17,
  lon_ny: 14,
}

// Bars per hour for each timeframe
const BARS_PER_HOUR: Record<string, number> = {
  M1: 60,
  M5: 12,
  M10: 6,
  M15: 4,
  M30: 2,
  H1: 1,
  H4: 0.25,
}

function extractTimeframeFromFilename(filename: string): string | null {
  const enrichedMatch = filename.match(/_ADR\d+_([A-Z0-9]+)(?:_bridged)?\.parquet$/i)
  if (enrichedMatch) return enrichedMatch[1].toUpperCase()
  const testMatch = filename.match(/^test_\w+_([A-Z0-9]+)\.parquet$/i)
  if (testMatch) return testMatch[1].toUpperCase()
  return null
}

function calculateSequenceLength(session: SessionOption, timeframe: string | null): number {
  if (!timeframe || !BARS_PER_HOUR[timeframe]) return 108
  return Math.floor(SESSION_HOURS[session] * BARS_PER_HOUR[timeframe])
}

interface TrainingCardConfig {
  id: string
  modelName: string
  parquetFile: string
  sessionOption: SessionOption
  targetOutcome: TargetOutcome
  sequenceLength: number
  batchSize: number
  numEpochs: number
  learningRate: number
  dModel: number
  nLayers: number
  nHeads: number
  dropoutRate: number
  earlyStoppingPatience: number
  saveToModels: boolean
  advancedOpen: boolean
}

function createDefaultCard(): TrainingCardConfig {
  return {
    id: crypto.randomUUID(),
    modelName: '',
    parquetFile: '',
    sessionOption: 'lon',
    targetOutcome: 'max_up',
    sequenceLength: 108,
    batchSize: 32,
    numEpochs: 100,
    learningRate: 0.0001,
    dModel: 128,
    nLayers: 4,
    nHeads: 4,
    dropoutRate: 0.1,
    earlyStoppingPatience: 15,
    saveToModels: true,
    advancedOpen: false,
  }
}

function formatElapsed(seconds: number): string {
  const hrs = Math.floor(seconds / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  if (hrs > 0) {
    return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

function formatReportTimestamp(timestamp: string): string {
  if (!timestamp || timestamp.length < 15) return timestamp
  const month = timestamp.slice(4, 6)
  const day = timestamp.slice(6, 8)
  const hour = timestamp.slice(9, 11)
  const min = timestamp.slice(11, 13)
  return `${month}/${day} ${hour}:${min}`
}

// Tooltip descriptions for report fields
const FIELD_TOOLTIPS: Record<string, string> = {
  // Summary fields
  total_epochs: 'Number of epochs completed. May be less than configured if early stopping triggered.',
  best_val_loss: 'Lowest validation loss achieved during training. The model checkpoint with this loss is saved.',
  final_directional_accuracy: 'Sign Accuracy on final epoch. Percentage of predictions with correct sign (+/-). Only meaningful for signed targets (deltas), not magnitude targets (max_up/max_down).',
  final_r_squared: 'R-squared on final epoch. How well predictions explain variance. Higher is better.',
  final_max_error: 'Largest prediction error seen during validation. Indicates worst-case performance.',
  elapsed_seconds: 'Total training time in seconds.',
  // Config fields
  model_name: 'Name used for saving model files and in reports.',
  target_session: 'Trading session the model was trained on (e.g., lon, asia, ny).',
  sequence_length: 'Number of bars (candles) the model sees as input context.',
  batch_size: 'Number of samples processed together per training step.',
  d_model: 'Internal representation size. Larger = more capacity.',
  n_layers: 'Number of transformer layers. More = deeper model.',
  n_heads: 'Number of attention heads per layer.',
  dropout_rate: 'Fraction of neurons disabled during training to prevent overfitting.',
  learning_rate: 'Step size for weight updates. Controls learning speed.',
  num_epochs: 'Maximum epochs configured for training.',
  early_stopping_patience: 'Epochs without improvement before stopping.',
  device: 'Hardware used for training (cuda = GPU, cpu = CPU).',
  embed_dim_level: 'Embedding dimension for MMLC level feature.',
  embed_dim_event: 'Embedding dimension for MMLC event feature.',
  embed_dim_direction: 'Embedding dimension for MMLC direction feature.',
  latent_dim: 'Size of the latent representation before prediction head.',
  save_every: 'How often (in epochs) to save checkpoints.',
  parquet_files: 'Training data files used.',
  combine_sessions: 'Whether multiple sessions were combined for training.',
}

// Interface for report data passed to popup
interface ReportPopupData {
  modelName: string
  filename: string
  timestamp: string
  directionalAccuracy: number
  rSquared: number
  bestLoss: number
  totalEpochs: number
  elapsedSeconds?: number
  targetSession?: string
  summary?: Record<string, unknown>
  config?: Record<string, unknown>
  epochs: Array<{
    epoch: number
    train_loss: number
    val_loss: number | null
    directional_accuracy: number | null
    r_squared: number | null
  }>
}

// Function to open a popup window with report details
function openReportPopup(data: ReportPopupData) {
  const popup = window.open('', '_blank', 'width=1200,height=1100,scrollbars=yes,resizable=yes')
  if (!popup) {
    alert('Popup blocked. Please allow popups for this site.')
    return
  }

  // Chart dimensions
  const width = 900
  const lossChartHeight = 250
  const qualityChartHeight = 200
  const padding = { top: 30, right: 70, bottom: 40, left: 70 }
  const lossChartInnerHeight = lossChartHeight - padding.top - padding.bottom
  const qualityChartInnerHeight = qualityChartHeight - padding.top - padding.bottom
  const chartWidth = width - padding.left - padding.right

  const normalize = (values: (number | null)[]): (number | null)[] => {
    const valid = values.filter((v): v is number => v !== null)
    if (valid.length === 0) return values
    const min = Math.min(...valid)
    const max = Math.max(...valid)
    const range = max - min || 1
    return values.map(v => v !== null ? (v - min) / range : null)
  }

  const trainLosses = data.epochs.map(e => e.train_loss)
  const valLosses = data.epochs.map(e => e.val_loss)
  const signAccs = data.epochs.map(e => e.directional_accuracy)
  const r2s = data.epochs.map(e => e.r_squared)

  const normTrain = normalize(trainLosses)
  const normVal = normalize(valLosses)
  const normSignAcc = normalize(signAccs)
  const normR2 = normalize(r2s)

  const createPath = (normalized: (number | null)[], chartHeight: number): string => {
    const points: string[] = []
    normalized.forEach((val, i) => {
      if (val !== null) {
        const x = padding.left + (i / (data.epochs.length - 1 || 1)) * chartWidth
        const y = padding.top + (1 - val) * chartHeight
        points.push(`${points.length === 0 ? 'M' : 'L'} ${x} ${y}`)
      }
    })
    return points.join(' ')
  }

  const getRange = (values: (number | null)[]): { min: number; max: number } => {
    const valid = values.filter((v): v is number => v !== null)
    if (valid.length === 0) return { min: 0, max: 1 }
    return { min: Math.min(...valid), max: Math.max(...valid) }
  }

  const trainRange = getRange(trainLosses)
  const signAccRange = getRange(signAccs)
  const r2Range = getRange(r2s)

  const formatElapsedTime = (seconds: number): string => {
    const hrs = Math.floor(seconds / 3600)
    const mins = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)
    if (hrs > 0) return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const createGridLines = (chartHeight: number) => [0, 0.25, 0.5, 0.75, 1].map(v =>
    `<line x1="${padding.left}" y1="${padding.top + v * chartHeight}" x2="${width - padding.right}" y2="${padding.top + v * chartHeight}" stroke="#444" stroke-opacity="0.3" />`
  ).join('')

  // Create summary rows
  const summaryRows = data.summary ? Object.entries(data.summary).map(([key, value]) =>
    `<tr>
      <td title="${FIELD_TOOLTIPS[key] || key.replace(/_/g, ' ')}">${key.replace(/_/g, ' ')}</td>
      <td>${typeof value === 'number' ? value.toFixed(4) : String(value)}</td>
    </tr>`
  ).join('') : ''

  // Create config rows
  const configRows = data.config ? Object.entries(data.config).map(([key, value]) =>
    `<tr>
      <td title="${FIELD_TOOLTIPS[key] || key.replace(/_/g, ' ')}">${key.replace(/_/g, ' ')}</td>
      <td>${typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(4)) : String(value)}</td>
    </tr>`
  ).join('') : ''

  // Create epoch rows
  const epochRows = data.epochs.map(ep =>
    `<tr>
      <td>${ep.epoch}</td>
      <td>${ep.train_loss.toFixed(4)}</td>
      <td class="val-loss">${ep.val_loss?.toFixed(4) || '—'}</td>
      <td class="sign-acc">${ep.directional_accuracy?.toFixed(1) || '—'}</td>
      <td class="r2">${ep.r_squared?.toFixed(3) || '—'}</td>
    </tr>`
  ).join('')

  // Tooltip content
  const lossTooltip = `Training Loss vs Validation Loss:
• Training Loss (amber): Measures how well the model fits the training data. Should decrease as model learns.
• Validation Loss (teal): Measures how well the model generalizes to unseen data. The "true" measure of quality.
• Both decreasing together = healthy learning
• Train ↓ but Val ↑ = overfitting (model memorizing, not generalizing)
• Best Loss saves the model checkpoint when validation loss is lowest.`

  const qualityTooltip = `Quality Metrics:
• R² (purple): How well predictions explain variance. Range 0-1, higher is better.
• Sign Accuracy (blue): Percentage of predictions with correct sign (+/-).
  Only meaningful for signed targets (deltas). Not applicable for magnitude targets (max_up/max_down).`

  const html = `
<!DOCTYPE html>
<html>
<head>
  <title>${data.modelName} - Training Report</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #0a0a0a;
      color: #e5e5e5;
      padding: 24px;
      line-height: 1.5;
    }
    .container { max-width: 1150px; margin: 0 auto; }
    h1 {
      font-size: 1.75rem;
      margin-bottom: 8px;
      color: #f5f5f5;
    }
    .subtitle {
      color: #888;
      font-size: 0.875rem;
      margin-bottom: 24px;
    }
    .key-metrics {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }
    .metric-card {
      padding: 20px;
      border-radius: 12px;
      text-align: center;
      border: 1px solid;
      cursor: help;
    }
    .metric-card.blue { background: rgba(59, 130, 246, 0.1); border-color: rgba(59, 130, 246, 0.3); }
    .metric-card.purple { background: rgba(168, 85, 247, 0.1); border-color: rgba(168, 85, 247, 0.3); }
    .metric-card.teal { background: rgba(20, 184, 166, 0.1); border-color: rgba(20, 184, 166, 0.3); }
    .metric-card.gray { background: rgba(100, 100, 100, 0.1); border-color: rgba(100, 100, 100, 0.3); }
    .metric-value { font-size: 2rem; font-weight: 700; font-family: monospace; }
    .metric-value.blue { color: #3b82f6; }
    .metric-value.purple { color: #a855f7; }
    .metric-value.teal { color: #14b8a6; }
    .metric-value.gray { color: #888; }
    .metric-label { font-size: 0.875rem; color: #888; margin-top: 4px; }
    .chart-container {
      background: #111;
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 16px;
      border: 1px solid #333;
    }
    .chart-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 12px;
    }
    .chart-title {
      font-size: 1rem;
      color: #e5e5e5;
    }
    .info-icon {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: #333;
      color: #888;
      font-size: 12px;
      cursor: help;
      font-style: normal;
    }
    .info-icon:hover { background: #444; color: #aaa; }
    .scale-toggle {
      margin-left: auto;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .scale-toggle button {
      padding: 4px 12px;
      border-radius: 6px;
      border: 1px solid #444;
      background: transparent;
      color: #888;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .scale-toggle button:hover { background: #333; color: #ccc; }
    .scale-toggle button.active {
      background: rgba(245, 158, 11, 0.2);
      border-color: rgba(245, 158, 11, 0.5);
      color: #f59e0b;
    }
    svg { display: block; margin: 0 auto; }
    .legend {
      display: flex;
      justify-content: center;
      gap: 32px;
      margin-top: 12px;
      font-size: 0.875rem;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: help;
    }
    .legend-line { width: 24px; height: 3px; border-radius: 2px; }
    .legend-line.dashed { border-top: 3px dashed; background: none; }
    .text-amber { color: #f59e0b; }
    .text-teal { color: #14b8a6; }
    .text-blue { color: #3b82f6; }
    .text-purple { color: #a855f7; }
    .bg-amber { background: #f59e0b; }
    .bg-teal { background: #14b8a6; }
    .bg-blue { background: #3b82f6; }
    .bg-purple { background: #a855f7; }
    .border-blue { border-color: #3b82f6; }
    .border-purple { border-color: #a855f7; }
    .data-sections {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 24px;
    }
    .data-section {
      background: #111;
      border-radius: 12px;
      padding: 20px;
      border: 1px solid #333;
    }
    .data-section h2 {
      font-size: 1rem;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid #333;
      color: #e5e5e5;
    }
    table { width: 100%; border-collapse: collapse; }
    td { padding: 6px 8px; font-size: 0.875rem; }
    tr:hover { background: rgba(255,255,255,0.03); }
    td:first-child { color: #888; cursor: help; }
    td:last-child { font-family: monospace; text-align: right; }
    .epoch-section {
      background: #111;
      border-radius: 12px;
      padding: 20px;
      border: 1px solid #333;
    }
    .epoch-section h2 {
      font-size: 1rem;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid #333;
      color: #e5e5e5;
    }
    .epoch-table-container {
      max-height: 400px;
      overflow-y: auto;
    }
    .epoch-table { width: 100%; border-collapse: collapse; }
    .epoch-table th {
      position: sticky;
      top: 0;
      background: #111;
      padding: 8px 12px;
      text-align: right;
      font-size: 0.875rem;
      color: #888;
      border-bottom: 1px solid #333;
      cursor: help;
    }
    .epoch-table th:first-child { text-align: left; }
    .epoch-table td {
      padding: 8px 12px;
      text-align: right;
      font-family: monospace;
      font-size: 0.875rem;
      border-bottom: 1px solid #222;
    }
    .epoch-table td:first-child { text-align: left; color: #888; }
    .epoch-table .val-loss { color: #14b8a6; }
    .epoch-table .sign-acc { color: #3b82f6; }
    .epoch-table .r2 { color: #a855f7; }
    .footer {
      text-align: center;
      color: #555;
      font-size: 0.75rem;
      margin-top: 24px;
      padding-top: 16px;
      border-top: 1px solid #222;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>${data.modelName}</h1>
    <div class="subtitle">
      ${data.targetSession ? `<span style="color: #f59e0b">${data.targetSession}</span> • ` : ''}
      ${data.timestamp} • ${data.filename}
    </div>

    <div class="key-metrics">
      <div class="metric-card blue" title="Sign Accuracy: Percentage of predictions where the model correctly predicted the sign (+/-) of the target. Only meaningful for signed targets (deltas). 50% = random guessing.">
        <div class="metric-value blue">${data.directionalAccuracy.toFixed(2)}%</div>
        <div class="metric-label">Sign Accuracy</div>
      </div>
      <div class="metric-card purple" title="R-squared: How well the model's predictions explain variance in actual outcomes. Range: -∞ to 1. Values >0 indicate better than predicting the mean. 1.0 = perfect predictions.">
        <div class="metric-value purple">${data.rSquared.toFixed(4)}</div>
        <div class="metric-label">R² Score</div>
      </div>
      <div class="metric-card teal" title="Best Loss: Lowest validation loss (MSE) achieved during training. Lower is better. This value determines which model checkpoint is saved.">
        <div class="metric-value teal">${data.bestLoss.toFixed(4)}</div>
        <div class="metric-label">Best Val Loss</div>
      </div>
      <div class="metric-card gray" title="Total training time and epochs completed">
        <div class="metric-value gray">${data.totalEpochs}</div>
        <div class="metric-label">Epochs${data.elapsedSeconds ? ` (${formatElapsedTime(data.elapsedSeconds)})` : ''}</div>
      </div>
    </div>

    <!-- Loss Chart -->
    <div class="chart-container">
      <div class="chart-header">
        <span class="chart-title">Loss Metrics</span>
        <span class="info-icon" title="${lossTooltip}">i</span>
        <div class="scale-toggle">
          <button id="btn-linear" class="active" onclick="setLossScale('linear')">Linear</button>
          <button id="btn-log" onclick="setLossScale('log')">Log</button>
        </div>
      </div>
      <div id="loss-chart-container">
        <svg id="loss-chart" width="${width}" height="${lossChartHeight}" style="background: #0a0a0a; border-radius: 8px;">
          ${createGridLines(lossChartInnerHeight)}
          <path id="train-path" d="${createPath(normTrain, lossChartInnerHeight)}" fill="none" stroke="#f59e0b" stroke-width="2.5" />
          <path id="val-path" d="${createPath(normVal, lossChartInnerHeight)}" fill="none" stroke="#14b8a6" stroke-width="2.5" />
          <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${lossChartHeight - padding.bottom}" stroke="#666" stroke-opacity="0.5" />
          <line x1="${padding.left}" y1="${lossChartHeight - padding.bottom}" x2="${width - padding.right}" y2="${lossChartHeight - padding.bottom}" stroke="#666" stroke-opacity="0.5" />
          <text x="${padding.left}" y="${lossChartHeight - 12}" fill="#888" font-size="12">1</text>
          <text x="${width - padding.right}" y="${lossChartHeight - 12}" fill="#888" font-size="12" text-anchor="end">${data.epochs.length}</text>
          <text x="${width / 2}" y="${lossChartHeight - 12}" fill="#888" font-size="12" text-anchor="middle">Epoch</text>
          <text id="y-max-label" x="${padding.left - 8}" y="${padding.top + 4}" fill="#f59e0b" font-size="11" text-anchor="end">${trainRange.max.toFixed(4)}</text>
          <text id="y-min-label" x="${padding.left - 8}" y="${lossChartHeight - padding.bottom}" fill="#f59e0b" font-size="11" text-anchor="end">${trainRange.min.toFixed(4)}</text>
        </svg>
      </div>
      <div class="legend">
        <div class="legend-item" title="Training loss (MSE) - measures how well the model fits training data. Should decrease over time.">
          <div class="legend-line bg-amber"></div>
          <span class="text-amber">Train Loss</span>
        </div>
        <div class="legend-item" title="Validation loss (MSE) - measures how well the model generalizes to unseen data. The true measure of model quality.">
          <div class="legend-line bg-teal"></div>
          <span class="text-teal">Val Loss</span>
        </div>
      </div>
    </div>

    <!-- Quality Metrics Chart -->
    <div class="chart-container">
      <div class="chart-header">
        <span class="chart-title">Quality Metrics</span>
        <span class="info-icon" title="${qualityTooltip}">i</span>
      </div>
      <svg width="${width}" height="${qualityChartHeight}" style="background: #0a0a0a; border-radius: 8px;">
        ${createGridLines(qualityChartInnerHeight)}
        <path d="${createPath(normSignAcc, qualityChartInnerHeight)}" fill="none" stroke="#3b82f6" stroke-width="2.5" />
        <path d="${createPath(normR2, qualityChartInnerHeight)}" fill="none" stroke="#a855f7" stroke-width="2.5" />
        <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${qualityChartHeight - padding.bottom}" stroke="#666" stroke-opacity="0.5" />
        <line x1="${padding.left}" y1="${qualityChartHeight - padding.bottom}" x2="${width - padding.right}" y2="${qualityChartHeight - padding.bottom}" stroke="#666" stroke-opacity="0.5" />
        <text x="${padding.left}" y="${qualityChartHeight - 12}" fill="#888" font-size="12">1</text>
        <text x="${width - padding.right}" y="${qualityChartHeight - 12}" fill="#888" font-size="12" text-anchor="end">${data.epochs.length}</text>
        <text x="${width / 2}" y="${qualityChartHeight - 12}" fill="#888" font-size="12" text-anchor="middle">Epoch</text>
        <text x="${padding.left - 8}" y="${padding.top + 4}" fill="#3b82f6" font-size="11" text-anchor="end">${signAccRange.max.toFixed(1)}%</text>
        <text x="${padding.left - 8}" y="${qualityChartHeight - padding.bottom}" fill="#3b82f6" font-size="11" text-anchor="end">${signAccRange.min.toFixed(1)}%</text>
        <text x="${width - padding.right + 8}" y="${padding.top + 4}" fill="#a855f7" font-size="11">${r2Range.max.toFixed(3)}</text>
        <text x="${width - padding.right + 8}" y="${qualityChartHeight - padding.bottom}" fill="#a855f7" font-size="11">${r2Range.min.toFixed(3)}</text>
      </svg>
      <div class="legend">
        <div class="legend-item" title="Sign Accuracy - percentage of predictions with correct sign (+/-). Only meaningful for signed targets like deltas. 50% = random.">
          <div class="legend-line bg-blue"></div>
          <span class="text-blue">Sign Acc %</span>
        </div>
        <div class="legend-item" title="R-squared - how well predictions explain variance. Range 0-1, higher is better.">
          <div class="legend-line bg-purple"></div>
          <span class="text-purple">R²</span>
        </div>
      </div>
    </div>

    <div class="data-sections">
      ${data.summary ? `
      <div class="data-section">
        <h2>Summary</h2>
        <table>${summaryRows}</table>
      </div>
      ` : ''}
      ${data.config ? `
      <div class="data-section">
        <h2>Configuration</h2>
        <table>${configRows}</table>
      </div>
      ` : ''}
    </div>

    ${data.epochs.length > 0 ? `
    <div class="epoch-section">
      <h2>Epoch History (${data.epochs.length} epochs)</h2>
      <div class="epoch-table-container">
        <table class="epoch-table">
          <thead>
            <tr>
              <th title="Epoch number">#</th>
              <th title="Training loss (MSE) - how well model fits training data">Train Loss</th>
              <th title="Validation loss (MSE) - how well model generalizes to unseen data">Val Loss</th>
              <th title="Sign accuracy on validation set - % of predictions with correct sign (+/-). Only meaningful for signed targets.">Sign Acc %</th>
              <th title="R-squared on validation set - prediction quality vs mean baseline">R²</th>
            </tr>
          </thead>
          <tbody>${epochRows}</tbody>
        </table>
      </div>
    </div>
    ` : ''}

    <div class="footer">${data.filename}</div>
  </div>

  <script>
    // Chart data and dimensions for dynamic updates
    const chartData = {
      trainLosses: ${JSON.stringify(trainLosses)},
      valLosses: ${JSON.stringify(valLosses)},
      padding: ${JSON.stringify(padding)},
      width: ${width},
      chartHeight: ${lossChartInnerHeight},
      fullHeight: ${lossChartHeight},
      numEpochs: ${data.epochs.length}
    };

    function setLossScale(scale) {
      const btnLinear = document.getElementById('btn-linear');
      const btnLog = document.getElementById('btn-log');
      const trainPath = document.getElementById('train-path');
      const valPath = document.getElementById('val-path');
      const yMaxLabel = document.getElementById('y-max-label');
      const yMinLabel = document.getElementById('y-min-label');

      // Update button states
      btnLinear.classList.toggle('active', scale === 'linear');
      btnLog.classList.toggle('active', scale === 'log');

      // Get valid values for range calculation
      const allLosses = [...chartData.trainLosses, ...chartData.valLosses].filter(v => v !== null && v > 0);
      if (allLosses.length === 0) return;

      let minVal, maxVal, transform;

      if (scale === 'log') {
        // Logarithmic scale
        const minLoss = Math.min(...allLosses);
        const maxLoss = Math.max(...allLosses);
        const logMin = Math.log10(Math.max(minLoss, 1e-10));
        const logMax = Math.log10(maxLoss);
        minVal = logMin;
        maxVal = logMax;
        transform = (v) => v !== null && v > 0 ? Math.log10(v) : null;
        yMaxLabel.textContent = maxLoss.toExponential(2);
        yMinLabel.textContent = minLoss.toExponential(2);
      } else {
        // Linear scale
        const minLoss = Math.min(...allLosses);
        const maxLoss = Math.max(...allLosses);
        minVal = minLoss;
        maxVal = maxLoss;
        transform = (v) => v;
        yMaxLabel.textContent = maxLoss.toFixed(4);
        yMinLabel.textContent = minLoss.toFixed(4);
      }

      // Normalize function
      const normalize = (values) => {
        const transformed = values.map(transform);
        const range = maxVal - minVal || 1;
        return transformed.map(v => v !== null ? (v - minVal) / range : null);
      };

      // Create path function
      const createPath = (normalized) => {
        const points = [];
        normalized.forEach((val, i) => {
          if (val !== null) {
            const x = chartData.padding.left + (i / (chartData.numEpochs - 1 || 1)) * (chartData.width - chartData.padding.left - chartData.padding.right);
            const y = chartData.padding.top + (1 - val) * chartData.chartHeight;
            points.push((points.length === 0 ? 'M ' : 'L ') + x + ' ' + y);
          }
        });
        return points.join(' ');
      };

      // Update paths
      trainPath.setAttribute('d', createPath(normalize(chartData.trainLosses)));
      valPath.setAttribute('d', createPath(normalize(chartData.valLosses)));
    }
  </script>
</body>
</html>
`

  popup.document.write(html)
  popup.document.close()
}

// Collapsible Section Header
function CollapsibleHeader({
  icon: Icon,
  title,
  count,
  isOpen,
  onToggle,
  accentColor = 'amber',
  statusBadge,
}: {
  icon: React.ElementType
  title: string
  count?: number
  isOpen: boolean
  onToggle: () => void
  accentColor?: 'amber' | 'teal' | 'cyan'
  statusBadge?: React.ReactNode
}) {
  const colorClasses = {
    amber: 'text-amber-400 border-amber-500/30 bg-amber-500/5',
    teal: 'text-teal-400 border-teal-500/30 bg-teal-500/5',
    cyan: 'text-cyan-400 border-cyan-500/30 bg-cyan-500/5',
  }

  return (
    <button
      onClick={onToggle}
      className={`w-full flex items-center gap-2 p-2 rounded-lg border transition-all duration-200 hover:bg-secondary/50 ${colorClasses[accentColor]}`}
    >
      {isOpen ? (
        <ChevronDown className="h-4 w-4 text-muted-foreground" />
      ) : (
        <ChevronRight className="h-4 w-4 text-muted-foreground" />
      )}
      <Icon className={`h-4 w-4 ${colorClasses[accentColor].split(' ')[0]}`} />
      <span className="text-sm font-semibold text-foreground/80">{title}</span>
      {count !== undefined && count > 0 && (
        <span className="text-xs px-1.5 py-0.5 rounded-full bg-secondary/50 text-muted-foreground">
          {count}
        </span>
      )}
      {statusBadge && <div className="ml-auto">{statusBadge}</div>}
    </button>
  )
}

export function TransformerTab({ workingDirectory }: TransformerTabProps) {
  const [trainingOpen, setTrainingOpen] = usePersistedState('transformerTrainingOpen', true)
  const [modelsOpen, setModelsOpen] = usePersistedState('transformerModelsOpen', false)
  const [reportsOpen, setReportsOpen] = usePersistedState('transformerReportsOpen', false)
  const [expandedReport, setExpandedReport] = useState<string | null>(null)
  const [selectedReports, setSelectedReports] = useState<Set<string>>(new Set())
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set())

  const [cards, setCards] = usePersistedState<TrainingCardConfig[]>(
    'lstmTrainingCards',
    [createDefaultCard()]
  )

  const { data: queueStatus } = useQueueStatus()
  const { data: modelsData, refetch: refetchModels } = useTransformerModels(workingDirectory)
  const { data: parquetFilesData } = useParquetFiles(workingDirectory)
  const { data: reportsData } = useTransformerReports(workingDirectory)
  const { data: reportDetailData } = useTransformerReportDetail(expandedReport, workingDirectory)

  const parquetFiles = parquetFilesData?.files || []
  const reports = reportsData?.reports || []
  const reportDetail = reportDetailData?.report

  const addToQueueMutation = useAddToQueue()
  const startQueueMutation = useStartQueue()
  const stopQueueMutation = useStopQueue()
  const clearQueueMutation = useClearQueue()
  const deleteModelMutation = useDeleteTransformerModel()
  const deleteReportMutation = useDeleteTransformerReport()
  const deleteReportsBulkMutation = useDeleteTransformerReportsBulk()
  const deleteModelsBulkMutation = useDeleteTransformerModelsBulk()

  const models = modelsData?.models || []
  const isQueueRunning = queueStatus?.queue_running || false
  const queueCards = queueStatus?.cards || []

  const addCard = () => setCards([...cards, createDefaultCard()])
  const removeCard = (id: string) => {
    if (cards.length > 1) setCards(cards.filter((c) => c.id !== id))
  }

  const updateCard = (id: string, updates: Partial<TrainingCardConfig>) => {
    setCards(cards.map((c) => {
      if (c.id !== id) return c
      const newCard = { ...c, ...updates }
      if ('parquetFile' in updates || 'sessionOption' in updates) {
        const tf = extractTimeframeFromFilename(newCard.parquetFile)
        newCard.sequenceLength = calculateSequenceLength(newCard.sessionOption, tf)
      }
      return newCard
    }))
  }

  const getAutoModelName = (card: TrainingCardConfig) => {
    const session = card.sessionOption.replace('_', '+')
    const tf = extractTimeframeFromFilename(card.parquetFile) || 'M5'
    return `ADR20_${tf}_${session}`
  }

  const getCardTimeframe = (card: TrainingCardConfig) => extractTimeframeFromFilename(card.parquetFile)
  const allCardsHaveParquet = cards.every((c) => c.parquetFile !== '')

  const handleStartAll = async () => {
    if (!allCardsHaveParquet) {
      alert('Please select a parquet file for all training configurations')
      return
    }
    for (const card of cards) {
      const modelName = card.modelName || getAutoModelName(card)
      await addToQueueMutation.mutateAsync({
        card_id: card.id,
        model_name: modelName,
        parquet_file: card.parquetFile,
        session_option: card.sessionOption,
        target_outcome: card.targetOutcome,
        sequence_length: card.sequenceLength,
        batch_size: card.batchSize,
        d_model: card.dModel,
        n_layers: card.nLayers,
        n_heads: card.nHeads,
        dropout_rate: card.dropoutRate,
        learning_rate: card.learningRate,
        num_epochs: card.numEpochs,
        early_stopping_patience: card.earlyStoppingPatience,
        save_to_models_folder: card.saveToModels,
        working_directory: workingDirectory,
      })
    }
    startQueueMutation.mutate(workingDirectory)
  }

  const handleDeleteModel = (name: string) => {
    if (confirm(`Delete model "${name}"?`)) {
      deleteModelMutation.mutate({ modelName: name, workingDirectory }, { onSuccess: () => refetchModels() })
    }
  }

  // Selection helpers for reports
  const toggleReportSelection = (filename: string) => {
    setSelectedReports(prev => {
      const next = new Set(prev)
      if (next.has(filename)) next.delete(filename)
      else next.add(filename)
      return next
    })
  }
  const selectAllReports = () => setSelectedReports(new Set(reports.map(r => r.filename)))
  const deselectAllReports = () => setSelectedReports(new Set())
  const deleteSelectedReports = () => {
    if (selectedReports.size === 0) return
    if (confirm(`Delete ${selectedReports.size} selected report(s)?`)) {
      deleteReportsBulkMutation.mutate(
        { filenames: Array.from(selectedReports), workingDirectory },
        { onSuccess: () => setSelectedReports(new Set()) }
      )
    }
  }

  // Selection helpers for models
  const toggleModelSelection = (name: string) => {
    setSelectedModels(prev => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }
  const selectAllModels = () => setSelectedModels(new Set(models.map(m => m.name)))
  const deselectAllModels = () => setSelectedModels(new Set())
  const deleteSelectedModels = () => {
    if (selectedModels.size === 0) return
    if (confirm(`Delete ${selectedModels.size} selected model(s)?`)) {
      deleteModelsBulkMutation.mutate(
        { modelNames: Array.from(selectedModels), workingDirectory },
        { onSuccess: () => { setSelectedModels(new Set()); refetchModels() } }
      )
    }
  }

  const getQueueStatusDisplay = () => {
    if (isQueueRunning) return { icon: Activity, color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/30', text: 'Running', animate: true }
    if (queueCards.some((c) => c.status === 'error')) return { icon: AlertCircle, color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/30', text: 'Error', animate: false }
    return { icon: CheckCircle2, color: 'text-teal-400', bg: 'bg-teal-500/10 border-teal-500/30', text: 'Ready', animate: false }
  }

  const queueStatusDisplay = getQueueStatusDisplay()
  const QueueStatusIcon = queueStatusDisplay.icon
  const currentTrainingCard = queueCards.find((c) => c.status === 'training')
  const pendingCount = queueCards.filter((c) => c.status === 'pending').length
  const completedCount = queueCards.filter((c) => c.status === 'completed').length

  const allOpen = trainingOpen && modelsOpen && reportsOpen
  const allClosed = !trainingOpen && !modelsOpen && !reportsOpen
  const expandAll = () => { setTrainingOpen(true); setModelsOpen(true); setReportsOpen(true) }
  const collapseAll = () => { setTrainingOpen(false); setModelsOpen(false); setReportsOpen(false) }

  const trainingStatusBadge = isQueueRunning ? (
    <span className="text-xs px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 font-medium flex items-center gap-1">
      <Activity className="h-3 w-3 animate-spin" />
      {pendingCount + 1} active
    </span>
  ) : queueCards.length > 0 ? (
    <span className="text-xs text-muted-foreground">{completedCount} done</span>
  ) : null

  return (
    <div className="space-y-2">
      {/* Expand/Collapse Controls */}
      <div className="flex justify-end gap-1">
        <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={expandAll} disabled={allOpen}>
          <ChevronsUpDown className="h-3.5 w-3.5 mr-1" />
          Expand
        </Button>
        <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={collapseAll} disabled={allClosed}>
          <ChevronsDownUp className="h-3.5 w-3.5 mr-1" />
          Collapse
        </Button>
      </div>

      {/* TRANSFORMER TRAINING SECTION */}
      <Card className="border-border/40 bg-gradient-to-br from-card to-card/90">
        <CollapsibleHeader
          icon={Zap}
          title="Transformer Training"
          isOpen={trainingOpen}
          onToggle={() => setTrainingOpen(!trainingOpen)}
          accentColor="amber"
          statusBadge={trainingStatusBadge}
        />

        {trainingOpen && (
          <CardContent className="pt-2 space-y-2">
            {/* Queue Status */}
            <div className={`flex items-center gap-2 p-2 rounded border ${queueStatusDisplay.bg} ${queueStatusDisplay.color}`}>
              <QueueStatusIcon className={`h-4 w-4 ${queueStatusDisplay.animate ? 'animate-spin' : ''}`} />
              <span className="text-sm font-medium">{queueStatusDisplay.text}</span>
              {currentTrainingCard && (
                <span className="text-xs opacity-70 ml-auto truncate max-w-[120px]">{currentTrainingCard.model_name}</span>
              )}
            </div>

            {/* Training Progress */}
            {currentTrainingCard && (
              <div className="space-y-1.5 p-2 rounded bg-secondary/30 border border-amber-500/30">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">
                    Epoch {currentTrainingCard.current_epoch}/{currentTrainingCard.total_epochs}
                  </span>
                  <span className="font-mono text-amber-400 flex items-center gap-1">
                    <Timer className="h-3.5 w-3.5" />
                    {formatElapsed(currentTrainingCard.elapsed_seconds)}
                  </span>
                </div>
                <div className="w-full bg-secondary rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-amber-500 to-amber-400 h-2 rounded-full transition-all"
                    style={{ width: `${currentTrainingCard.total_epochs > 0 ? (currentTrainingCard.current_epoch / currentTrainingCard.total_epochs) * 100 : 0}%` }}
                  />
                </div>
                <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-xs">
                  <div className="flex justify-between" title="MSE loss on training data. Measures how well model fits training samples.">
                    <span className="text-muted-foreground cursor-help">Train:</span>
                    <span className="font-mono">{currentTrainingCard.train_loss?.toFixed(4) || '—'}</span>
                  </div>
                  <div className="flex justify-between" title="MSE loss on held-out validation data. Measures generalization. Should decrease with Train.">
                    <span className="text-muted-foreground cursor-help">Val:</span>
                    <span className="font-mono">{currentTrainingCard.val_loss?.toFixed(4) || '—'}</span>
                  </div>
                  <div className="flex justify-between" title="Lowest validation loss achieved. Lower is better. Used to save best model checkpoint.">
                    <span className="text-muted-foreground cursor-help">Best:</span>
                    <span className="font-mono text-teal-400">{currentTrainingCard.best_loss?.toFixed(4) || '—'}</span>
                  </div>
                  <div className="flex justify-between" title="Sign Accuracy: % of predictions with correct sign (+/-). Only meaningful for signed targets (deltas). 50% = random.">
                    <span className="text-muted-foreground cursor-help">Sign Acc:</span>
                    <span className="font-mono text-blue-400">
                      {currentTrainingCard.directional_accuracy != null ? `${currentTrainingCard.directional_accuracy.toFixed(1)}%` : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between" title="R-squared: How well predictions explain variance. Range -∞ to 1. Values >0 beat mean prediction. 1.0 = perfect.">
                    <span className="text-muted-foreground cursor-help">R²:</span>
                    <span className="font-mono text-purple-400">{currentTrainingCard.r_squared?.toFixed(3) || '—'}</span>
                  </div>
                  <div className="flex justify-between" title="Gradient norm: Magnitude of gradients during backprop. Very high values may indicate training instability.">
                    <span className="text-muted-foreground cursor-help">Grad:</span>
                    <span className="font-mono">{currentTrainingCard.grad_norm?.toFixed(2) || '—'}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Training Cards */}
            {cards.map((card, idx) => (
              <div key={card.id} className="p-2 rounded bg-secondary/20 border border-border/30 space-y-2">
                <div className="flex items-center justify-between">
                  <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                    idx % 4 === 0 ? 'bg-blue-500/30 text-blue-300' :
                    idx % 4 === 1 ? 'bg-purple-500/30 text-purple-300' :
                    idx % 4 === 2 ? 'bg-teal-500/30 text-teal-300' :
                    'bg-amber-500/30 text-amber-300'
                  }`}>
                    Config {idx + 1}
                  </span>
                  {cards.length > 1 && !isQueueRunning && (
                    <Button size="sm" variant="ghost" className="h-5 w-5 p-0" onClick={() => removeCard(card.id)}>
                      <XCircle className="h-3.5 w-3.5 text-muted-foreground hover:text-red-400" />
                    </Button>
                  )}
                </div>

                <div title="Enriched parquet file containing MMLC states and outcome data. More rows = more training samples.">
                  <label className="text-xs text-muted-foreground flex items-center gap-1 mb-1 cursor-help">
                    <Database className="h-3.5 w-3.5" />
                    Training Data
                  </label>
                  <select
                    value={card.parquetFile}
                    onChange={(e) => updateCard(card.id, { parquetFile: e.target.value })}
                    disabled={isQueueRunning}
                    className={`w-full h-8 text-sm bg-background/50 border rounded px-2 font-mono ${
                      card.parquetFile === '' ? 'border-amber-500/50 text-muted-foreground' : 'border-border/50'
                    } ${isQueueRunning ? 'opacity-50' : ''}`}
                  >
                    <option value="">-- Select Parquet --</option>
                    {parquetFiles.map((pf) => (
                      <option key={pf.name} value={pf.name}>
                        {pf.name} ({pf.rows?.toLocaleString() || '?'} rows)
                      </option>
                    ))}
                  </select>
                </div>

                <div className="grid grid-cols-3 gap-1" title="Trading session to train on. Determines which time window's MMLC states and outcomes are used. Combined sessions (Asia+Lon, Lon+NY) span multiple sessions.">
                  {SESSION_OPTIONS.map((opt) => (
                    <label
                      key={opt.value}
                      className={`flex items-center justify-center py-1.5 rounded text-xs cursor-pointer transition-all ${
                        card.sessionOption === opt.value
                          ? 'bg-amber-500/20 border border-amber-500/50 text-amber-400 font-medium'
                          : 'bg-secondary/30 border border-border/30 text-muted-foreground hover:border-border'
                      } ${isQueueRunning ? 'opacity-50' : ''}`}
                    >
                      <input
                        type="radio"
                        name={`session-${card.id}`}
                        checked={card.sessionOption === opt.value}
                        onChange={() => updateCard(card.id, { sessionOption: opt.value })}
                        disabled={isQueueRunning}
                        className="sr-only"
                      />
                      {opt.label}
                    </label>
                  ))}
                </div>

                <div title="Target outcome to predict. Max Up/Down are magnitude targets. Next Bar and Session Close are signed deltas.">
                  <label className="text-xs text-muted-foreground flex items-center gap-1 mb-1 cursor-help">
                    <TrendingUp className="h-3.5 w-3.5" />
                    Target Outcome
                  </label>
                  <select
                    value={card.targetOutcome}
                    onChange={(e) => updateCard(card.id, { targetOutcome: e.target.value as TargetOutcome })}
                    disabled={isQueueRunning}
                    className={`w-full h-8 text-sm bg-background/50 border rounded px-2 font-mono border-border/50 ${isQueueRunning ? 'opacity-50' : ''}`}
                    title={TARGET_OUTCOME_OPTIONS.find(o => o.value === card.targetOutcome)?.description}
                  >
                    {TARGET_OUTCOME_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value} title={opt.description}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </div>

                <Input
                  value={card.modelName}
                  onChange={(e) => updateCard(card.id, { modelName: e.target.value })}
                  placeholder={getAutoModelName(card)}
                  disabled={isQueueRunning}
                  className="text-sm h-8 bg-background/50 border-border/50 font-mono"
                  title="Model name for saving. Leave blank to auto-generate based on timeframe and session. Used in saved model filename and reports."
                />

                {card.parquetFile && (
                  <div className="flex items-center gap-2 text-xs text-muted-foreground bg-secondary/20 px-2 py-1 rounded cursor-help" title="TF = Timeframe detected from filename. Bars = sequence length (number of bars the model sees as input context). Auto-calculated from session duration and timeframe.">
                    <span>TF: <span className="text-amber-400 font-mono">{getCardTimeframe(card) || '?'}</span></span>
                    <span>•</span>
                    <span>Bars: <span className="text-teal-400 font-mono">{card.sequenceLength}</span></span>
                  </div>
                )}

                <div className="grid grid-cols-3 gap-1">
                  <div title="Batch Size: Number of samples processed together before updating weights. Larger = faster but uses more GPU memory. Typical: 16-64.">
                    <label className="text-xs text-muted-foreground cursor-help">Batch</label>
                    <Input
                      type="number"
                      value={card.batchSize}
                      onChange={(e) => updateCard(card.id, { batchSize: parseInt(e.target.value) || 32 })}
                      disabled={isQueueRunning}
                      className="text-sm h-7 bg-background/50 border-border/50 font-mono px-1.5"
                    />
                  </div>
                  <div title="Epochs: Number of complete passes through all training data. More epochs = more learning, but risk of overfitting. Early stopping will halt if no improvement.">
                    <label className="text-xs text-muted-foreground cursor-help">Epochs</label>
                    <Input
                      type="number"
                      value={card.numEpochs}
                      onChange={(e) => updateCard(card.id, { numEpochs: parseInt(e.target.value) || 100 })}
                      disabled={isQueueRunning}
                      className="text-sm h-7 bg-background/50 border-border/50 font-mono px-1.5"
                    />
                  </div>
                  <div title="Learning Rate: How much to adjust weights each step. Too high = unstable, too low = slow learning. Typical: 0.0001 to 0.001.">
                    <label className="text-xs text-muted-foreground cursor-help">LR</label>
                    <Input
                      type="number"
                      value={card.learningRate}
                      onChange={(e) => updateCard(card.id, { learningRate: parseFloat(e.target.value) || 0.0001 })}
                      step="0.0001"
                      disabled={isQueueRunning}
                      className="text-sm h-7 bg-background/50 border-border/50 font-mono px-1.5"
                    />
                  </div>
                </div>

                <button
                  onClick={() => updateCard(card.id, { advancedOpen: !card.advancedOpen })}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                >
                  {card.advancedOpen ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
                  <Settings2 className="h-3.5 w-3.5" />
                  Advanced
                </button>

                {card.advancedOpen && (
                  <div className="grid grid-cols-3 gap-1">
                    <div title="Model Dimension: Size of internal representations. Larger = more capacity but slower and needs more data. Typical: 64-256.">
                      <label className="text-xs text-muted-foreground cursor-help">d_model</label>
                      <Input type="number" value={card.dModel} onChange={(e) => updateCard(card.id, { dModel: parseInt(e.target.value) || 128 })} disabled={isQueueRunning} className="text-sm h-7 bg-background/50 border-border/50 font-mono px-1.5" />
                    </div>
                    <div title="Number of Transformer Layers: Depth of the model. More layers = can learn more complex patterns but slower and risk of overfitting. Typical: 2-6.">
                      <label className="text-xs text-muted-foreground cursor-help">n_layers</label>
                      <Input type="number" value={card.nLayers} onChange={(e) => updateCard(card.id, { nLayers: parseInt(e.target.value) || 4 })} disabled={isQueueRunning} className="text-sm h-7 bg-background/50 border-border/50 font-mono px-1.5" />
                    </div>
                    <div title="Attention Heads: Number of parallel attention mechanisms per layer. More heads = can attend to different patterns simultaneously. Must divide d_model evenly. Typical: 2-8.">
                      <label className="text-xs text-muted-foreground cursor-help">n_heads</label>
                      <Input type="number" value={card.nHeads} onChange={(e) => updateCard(card.id, { nHeads: parseInt(e.target.value) || 4 })} disabled={isQueueRunning} className="text-sm h-7 bg-background/50 border-border/50 font-mono px-1.5" />
                    </div>
                    <div title="Dropout Rate: Fraction of neurons randomly disabled during training to prevent overfitting. Higher = more regularization. Typical: 0.1-0.3.">
                      <label className="text-xs text-muted-foreground cursor-help">Dropout</label>
                      <Input type="number" value={card.dropoutRate} onChange={(e) => updateCard(card.id, { dropoutRate: parseFloat(e.target.value) || 0.1 })} step="0.05" disabled={isQueueRunning} className="text-sm h-7 bg-background/50 border-border/50 font-mono px-1.5" />
                    </div>
                    <div title="Early Stopping Patience: Number of epochs without improvement before stopping. Prevents overfitting by stopping when validation loss plateaus. Typical: 10-20.">
                      <label className="text-xs text-muted-foreground cursor-help">Early Stop</label>
                      <Input type="number" value={card.earlyStoppingPatience} onChange={(e) => updateCard(card.id, { earlyStoppingPatience: parseInt(e.target.value) || 15 })} disabled={isQueueRunning} className="text-sm h-7 bg-background/50 border-border/50 font-mono px-1.5" />
                    </div>
                    <div className="flex items-end pb-1" title="Save Model: If checked, saves the best model checkpoint to the models folder for later use in predictions.">
                      <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer cursor-help">
                        <input type="checkbox" checked={card.saveToModels} onChange={(e) => updateCard(card.id, { saveToModels: e.target.checked })} disabled={isQueueRunning} className="h-3.5 w-3.5 rounded" />
                        Save
                      </label>
                    </div>
                  </div>
                )}
              </div>
            ))}

            {!isQueueRunning && (
              <Button variant="outline" size="sm" className="w-full h-7 text-xs border-dashed" onClick={addCard}>
                <Plus className="h-3.5 w-3.5 mr-1" />
                Add Configuration
              </Button>
            )}

            <div className="pt-1 space-y-1.5">
              {isQueueRunning ? (
                <Button variant="destructive" className="w-full h-9 text-sm" onClick={() => stopQueueMutation.mutate()}>
                  <Square className="mr-2 h-4 w-4" />
                  Stop Queue
                </Button>
              ) : (
                <Button
                  className={`w-full h-9 text-sm text-white shadow-lg ${
                    allCardsHaveParquet
                      ? 'bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400'
                      : 'bg-muted cursor-not-allowed'
                  }`}
                  onClick={handleStartAll}
                  disabled={!allCardsHaveParquet || addToQueueMutation.isPending || startQueueMutation.isPending}
                >
                  {addToQueueMutation.isPending || startQueueMutation.isPending ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Play className="mr-2 h-4 w-4" />
                  )}
                  {allCardsHaveParquet ? `Start All (${cards.length})` : 'Select Parquet Files'}
                </Button>
              )}
              {queueCards.length > 0 && !isQueueRunning && (
                <Button variant="outline" size="sm" className="w-full h-7 text-xs" onClick={() => clearQueueMutation.mutate()}>
                  Clear Queue
                </Button>
              )}
            </div>

            {queueCards.length > 0 && (
              <div className="space-y-1 pt-1 border-t border-border/30">
                <div className="text-xs text-muted-foreground flex items-center gap-1 py-1">
                  <Activity className="h-3.5 w-3.5" />
                  Queue ({queueCards.length})
                </div>
                <div className="space-y-1">
                  {queueCards.map((qc) => (
                    <div
                      key={qc.card_id}
                      className={`p-1.5 rounded border ${
                        qc.status === 'training' ? 'bg-amber-500/10 border-amber-500/30'
                          : qc.status === 'completed' ? 'bg-teal-500/10 border-teal-500/30'
                          : qc.status === 'error' ? 'bg-red-500/10 border-red-500/30'
                          : 'bg-secondary/30 border-border/30'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium truncate text-foreground/80">{qc.model_name}</span>
                        <span className={`text-xs ${
                          qc.status === 'training' ? 'text-amber-400'
                            : qc.status === 'completed' ? 'text-teal-400'
                            : qc.status === 'error' ? 'text-red-400'
                            : 'text-muted-foreground'
                        }`}>
                          {qc.status === 'training' ? `${qc.current_epoch}/${qc.total_epochs}` : qc.status}
                        </span>
                      </div>
                      {qc.status === 'completed' && qc.final_directional_accuracy != null && (
                        <div className="flex gap-2 text-xs mt-0.5">
                          <span className="text-blue-400">{qc.final_directional_accuracy.toFixed(1)}%</span>
                          <span className="text-purple-400">R²: {qc.final_r_squared?.toFixed(3)}</span>
                          <span className="text-muted-foreground">{formatElapsed(qc.elapsed_seconds)}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        )}
      </Card>

      {/* SAVED MODELS SECTION */}
      <Card className="border-border/40 bg-gradient-to-br from-card to-card/90">
        <CollapsibleHeader icon={Layers} title="Saved Models" count={models.length} isOpen={modelsOpen} onToggle={() => setModelsOpen(!modelsOpen)} accentColor="teal" />

        {modelsOpen && (
          <CardContent className="pt-2">
            {models.length === 0 ? (
              <div className="text-xs text-muted-foreground text-center py-3">No saved models yet</div>
            ) : (
              <div className="space-y-1.5">
                {/* Bulk selection controls */}
                <div className="flex items-center justify-between pb-1 border-b border-border/20">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => selectedModels.size === models.length ? deselectAllModels() : selectAllModels()}
                      className="p-0.5 hover:bg-secondary/50 rounded"
                      title={selectedModels.size === models.length ? "Deselect all" : "Select all"}
                    >
                      {selectedModels.size === models.length && models.length > 0 ? (
                        <CheckSquare className="h-4 w-4 text-teal-400" />
                      ) : (
                        <Square className="h-4 w-4 text-muted-foreground" />
                      )}
                    </button>
                    <span className="text-xs text-muted-foreground">
                      {selectedModels.size > 0 ? `${selectedModels.size} selected` : 'Select all'}
                    </span>
                  </div>
                  {selectedModels.size > 0 && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 px-2 text-xs text-red-400 hover:text-red-300 hover:bg-red-500/10"
                      onClick={deleteSelectedModels}
                      disabled={deleteModelsBulkMutation.isPending}
                    >
                      <Trash2 className="h-3 w-3 mr-1" />
                      Delete ({selectedModels.size})
                    </Button>
                  )}
                </div>
                {models.map((model) => (
                  <div key={model.name} className="flex items-center gap-2 p-2 rounded bg-secondary/30 border border-border/30 group hover:border-teal-500/30">
                    <button
                      onClick={() => toggleModelSelection(model.name)}
                      className="p-0.5 hover:bg-secondary/50 rounded shrink-0"
                    >
                      {selectedModels.has(model.name) ? (
                        <CheckSquare className="h-4 w-4 text-teal-400" />
                      ) : (
                        <Square className="h-4 w-4 text-muted-foreground" />
                      )}
                    </button>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium truncate text-foreground/80">{model.name}</div>
                      <div className="flex gap-2 text-xs text-muted-foreground">
                        {model.best_loss && <span className="text-teal-400">Loss: {model.best_loss.toFixed(4)}</span>}
                        {model.epochs_trained > 0 && <span>{model.epochs_trained} epochs</span>}
                        {model.target_session && <span>{model.target_session}</span>}
                      </div>
                    </div>
                    <Button size="sm" variant="ghost" className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100" onClick={() => handleDeleteModel(model.name)}>
                      <Trash2 className="h-3.5 w-3.5 text-red-400" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        )}
      </Card>

      {/* TRAINING REPORTS SECTION */}
      <Card className="border-border/40 bg-gradient-to-br from-card to-card/90">
        <CollapsibleHeader icon={FileText} title="Training Reports" count={reports.length} isOpen={reportsOpen} onToggle={() => setReportsOpen(!reportsOpen)} accentColor="cyan" />

        {reportsOpen && (
          <CardContent className="pt-2">
            {reports.length === 0 ? (
              <div className="text-xs text-muted-foreground text-center py-3">No training reports yet</div>
            ) : (
              <div className="space-y-1.5">
                {/* Bulk selection controls */}
                <div className="flex items-center justify-between pb-1 border-b border-border/20">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => selectedReports.size === reports.length ? deselectAllReports() : selectAllReports()}
                      className="p-0.5 hover:bg-secondary/50 rounded"
                      title={selectedReports.size === reports.length ? "Deselect all" : "Select all"}
                    >
                      {selectedReports.size === reports.length && reports.length > 0 ? (
                        <CheckSquare className="h-4 w-4 text-cyan-400" />
                      ) : (
                        <Square className="h-4 w-4 text-muted-foreground" />
                      )}
                    </button>
                    <span className="text-xs text-muted-foreground">
                      {selectedReports.size > 0 ? `${selectedReports.size} selected` : 'Select all'}
                    </span>
                  </div>
                  {selectedReports.size > 0 && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 px-2 text-xs text-red-400 hover:text-red-300 hover:bg-red-500/10"
                      onClick={deleteSelectedReports}
                      disabled={deleteReportsBulkMutation.isPending}
                    >
                      <Trash2 className="h-3 w-3 mr-1" />
                      Delete ({selectedReports.size})
                    </Button>
                  )}
                </div>
                {reports.map((report) => {
                  const isExpanded = expandedReport === report.filename
                  const detail = isExpanded ? reportDetail : null
                  return (
                    <div key={report.filename} className="group rounded border border-border/30 bg-secondary/20 overflow-hidden hover:border-cyan-500/30">
                      <div className="flex items-start">
                        <button
                          onClick={() => toggleReportSelection(report.filename)}
                          className="p-2 hover:bg-secondary/50 shrink-0"
                        >
                          {selectedReports.has(report.filename) ? (
                            <CheckSquare className="h-4 w-4 text-cyan-400" />
                          ) : (
                            <Square className="h-4 w-4 text-muted-foreground" />
                          )}
                        </button>
                        <button onClick={() => setExpandedReport(isExpanded ? null : report.filename)} className="flex-1 p-2 pl-0 text-left flex items-start gap-2">
                          <div className="pt-0.5">
                            {isExpanded ? <ChevronDown className="h-4 w-4 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 text-muted-foreground" />}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="text-base font-bold text-blue-400 font-mono cursor-help" title="Sign Accuracy: % of predictions with correct sign (+/-). Only meaningful for signed targets.">{report.directional_accuracy.toFixed(1)}%</span>
                              <span className="text-xs text-purple-400 font-mono cursor-help" title="R-squared: Prediction quality. Range -∞ to 1. Higher is better.">R² {report.r_squared.toFixed(3)}</span>
                              <span className="text-xs text-muted-foreground ml-auto cursor-help" title="Total epochs completed (may be less than configured if early stopping triggered)">{report.total_epochs} ep</span>
                              {report.elapsed_seconds != null && (
                                <span className="text-xs text-muted-foreground cursor-help" title="Total training time">{formatElapsed(report.elapsed_seconds)}</span>
                              )}
                            </div>
                            <div className="flex items-center gap-2 mt-0.5">
                              <span className="text-xs font-medium truncate text-foreground/80">{report.model_name}</span>
                              {report.target_session && <span className="text-xs text-amber-400/70">{report.target_session}</span>}
                            </div>
                            <div className="text-xs text-muted-foreground/70">{formatReportTimestamp(report.timestamp)}</div>
                          </div>
                        </button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-6 w-6 p-0 mt-2 mr-1 opacity-0 group-hover:opacity-100"
                          onClick={async () => {
                            try {
                              const detailData = await api.transformer.getReportDetail(report.filename, workingDirectory)
                              openReportPopup({
                                modelName: report.model_name,
                                filename: report.filename,
                                timestamp: formatReportTimestamp(report.timestamp),
                                directionalAccuracy: report.directional_accuracy,
                                rSquared: report.r_squared,
                                bestLoss: report.best_loss,
                                totalEpochs: report.total_epochs,
                                elapsedSeconds: report.elapsed_seconds ?? undefined,
                                targetSession: report.target_session ?? undefined,
                                summary: detailData.report.summary,
                                config: detailData.report.config,
                                epochs: detailData.report.epochs
                              })
                            } catch (err) {
                              console.error('Failed to load report detail:', err)
                            }
                          }}
                          title="Open detailed report in new window"
                        >
                          <ExternalLink className="h-3.5 w-3.5 text-cyan-400" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-6 w-6 p-0 mt-2 mr-2 opacity-0 group-hover:opacity-100"
                          onClick={() => {
                            if (confirm(`Delete report "${report.model_name}"?`)) {
                              deleteReportMutation.mutate({
                                filename: report.filename,
                                workingDirectory: workingDirectory,
                              })
                            }
                          }}
                          disabled={deleteReportMutation.isPending}
                          title="Delete report"
                        >
                          <Trash2 className="h-3.5 w-3.5 text-red-400" />
                        </Button>
                      </div>

                      {isExpanded && (
                        <div className="px-2 pb-2 border-t border-border/20">
                          <div className="bg-background/30 rounded p-2 mt-2 space-y-3">
                            {/* Key Metrics */}
                            <div className="grid grid-cols-3 gap-2">
                              <div
                                className="text-center p-2 rounded bg-blue-500/10 border border-blue-500/20 cursor-help"
                                title="Sign Accuracy: Percentage of predictions with correct sign (+/-). Only meaningful for signed targets (deltas), not magnitude targets. 50% = random guessing."
                              >
                                <TrendingUp className="h-4 w-4 text-blue-400 mx-auto mb-1" />
                                <div className="text-sm font-bold text-blue-400">{report.directional_accuracy.toFixed(2)}%</div>
                                <div className="text-xs text-muted-foreground">Sign Accuracy</div>
                              </div>
                              <div
                                className="text-center p-2 rounded bg-purple-500/10 border border-purple-500/20 cursor-help"
                                title="R-squared: How well the model's predictions explain variance in actual outcomes. Range: -∞ to 1. Values >0 indicate better than predicting the mean. 1.0 = perfect predictions."
                              >
                                <BarChart3 className="h-4 w-4 text-purple-400 mx-auto mb-1" />
                                <div className="text-sm font-bold text-purple-400">{report.r_squared.toFixed(4)}</div>
                                <div className="text-xs text-muted-foreground">R² Score</div>
                              </div>
                              <div
                                className="text-center p-2 rounded bg-teal-500/10 border border-teal-500/20 cursor-help"
                                title="Best Loss: Lowest validation loss (MSE) achieved during training. Lower is better. This value determines which model checkpoint is saved."
                              >
                                <Activity className="h-4 w-4 text-teal-400 mx-auto mb-1" />
                                <div className="text-sm font-bold text-teal-400">{report.best_loss.toFixed(4)}</div>
                                <div className="text-xs text-muted-foreground">Best Loss</div>
                              </div>
                            </div>

                            {/* Summary */}
                            {detail?.summary && (
                              <div>
                                <div className="text-xs font-semibold text-foreground/80 border-b border-border/30 pb-1 mb-1" title="Key metrics from the completed training run">Summary</div>
                                <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-xs">
                                  {Object.entries(detail.summary).map(([key, value]) => (
                                    <div key={key} className="flex justify-between cursor-help" title={FIELD_TOOLTIPS[key] || `${key.replace(/_/g, ' ')}`}>
                                      <span className="text-muted-foreground">{key.replace(/_/g, ' ')}:</span>
                                      <span className="font-mono">{typeof value === 'number' ? value.toFixed(4) : String(value)}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Config */}
                            {detail?.config && (
                              <div>
                                <div className="text-xs font-semibold text-foreground/80 border-b border-border/30 pb-1 mb-1" title="Model and training hyperparameters used for this run">Configuration</div>
                                <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-xs">
                                  {Object.entries(detail.config).map(([key, value]) => (
                                    <div key={key} className="flex justify-between cursor-help" title={FIELD_TOOLTIPS[key] || `${key.replace(/_/g, ' ')}`}>
                                      <span className="text-muted-foreground">{key.replace(/_/g, ' ')}:</span>
                                      <span className="font-mono truncate max-w-[100px]" title={String(value)}>
                                        {typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(4)) : String(value)}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Epoch History */}
                            {detail?.epochs && detail.epochs.length > 0 && (
                              <div>
                                <div className="text-xs font-semibold text-foreground/80 border-b border-border/30 pb-1 mb-1">
                                  Epoch History ({detail.epochs.length})
                                </div>
                                <div className="max-h-48 overflow-y-auto">
                                  <table className="w-full text-xs">
                                    <thead className="sticky top-0 bg-background/90">
                                      <tr className="text-muted-foreground">
                                        <th className="text-left py-1 px-1" title="Epoch number">#</th>
                                        <th className="text-right py-1 px-1 cursor-help" title="Training loss (MSE) - how well model fits training data">Train</th>
                                        <th className="text-right py-1 px-1 cursor-help" title="Validation loss (MSE) - how well model generalizes to unseen data">Val</th>
                                        <th className="text-right py-1 px-1 cursor-help" title="Sign Accuracy: % of predictions with correct sign (+/-). Only meaningful for signed targets.">Sign%</th>
                                        <th className="text-right py-1 px-1 cursor-help" title="R-squared on validation set - prediction quality vs mean baseline">R²</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {detail.epochs.map((ep) => (
                                        <tr key={ep.epoch} className="border-t border-border/10 hover:bg-secondary/30">
                                          <td className="py-0.5 px-1 text-muted-foreground">{ep.epoch}</td>
                                          <td className="py-0.5 px-1 text-right font-mono">{ep.train_loss.toFixed(4)}</td>
                                          <td className="py-0.5 px-1 text-right font-mono text-teal-400">{ep.val_loss?.toFixed(4) || '—'}</td>
                                          <td className="py-0.5 px-1 text-right font-mono text-blue-400">{ep.directional_accuracy?.toFixed(1) || '—'}</td>
                                          <td className="py-0.5 px-1 text-right font-mono text-purple-400">{ep.r_squared?.toFixed(3) || '—'}</td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              </div>
                            )}

                            <div className="text-xs text-muted-foreground/60 pt-1 border-t border-border/20">{report.filename}</div>
                          </div>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </CardContent>
        )}
      </Card>
    </div>
  )
}
