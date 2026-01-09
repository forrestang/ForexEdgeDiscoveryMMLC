import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import {
  useTransformerModels,
  useDeleteTransformerModel,
  useQueueStatus,
  useAddToQueue,
  useStartQueue,
  useStopQueue,
  useClearQueue,
  useParquetFiles,
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
  ChevronDown,
  ChevronRight,
  Trash2,
  Layers,
  Activity,
  Settings2,
  Database,
  Plus,
  Timer,
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
  { value: 'asia_lon', label: 'Asia + Lon' },
  { value: 'lon_ny', label: 'Lon + NY' },
]

// Session durations in hours
const SESSION_HOURS: Record<SessionOption, number> = {
  asia: 9,      // 00:00 - 09:00
  lon: 9,       // 08:00 - 17:00
  ny: 9,        // 13:00 - 22:00
  day: 22,      // 00:00 - 22:00
  asia_lon: 17, // 00:00 - 17:00
  lon_ny: 14,   // 08:00 - 22:00
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

// Extract timeframe from parquet filename (e.g., "EURUSD_..._ADR20_M5_bridged.parquet" -> "M5")
function extractTimeframeFromFilename(filename: string): string | null {
  const match = filename.match(/_ADR\d+_([A-Z0-9]+)(?:_bridged)?\.parquet$/i)
  return match ? match[1].toUpperCase() : null
}

// Calculate sequence length based on session and timeframe
function calculateSequenceLength(session: SessionOption, timeframe: string | null): number {
  if (!timeframe || !BARS_PER_HOUR[timeframe]) {
    return 108 // Default fallback
  }
  const hours = SESSION_HOURS[session]
  const barsPerHour = BARS_PER_HOUR[timeframe]
  return Math.floor(hours * barsPerHour)
}

// Training card configuration
interface TrainingCardConfig {
  id: string
  modelName: string
  parquetFile: string  // Selected parquet file for training
  sessionOption: SessionOption
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
    parquetFile: '',  // Must be selected by user
    sessionOption: 'lon',
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

// Format elapsed time as mm:ss or hh:mm:ss
function formatElapsed(seconds: number): string {
  const hrs = Math.floor(seconds / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)

  if (hrs > 0) {
    return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

export function TransformerTab({ workingDirectory }: TransformerTabProps) {
  // Persisted multi-card state for queue
  const [cards, setCards] = usePersistedState<TrainingCardConfig[]>(
    'lstmTrainingCards',
    [createDefaultCard()]
  )

  // Queries
  const { data: queueStatus } = useQueueStatus()
  const { data: modelsData, refetch: refetchModels } = useTransformerModels(workingDirectory)
  const { data: parquetFilesData } = useParquetFiles(workingDirectory)

  const parquetFiles = parquetFilesData?.files || []

  // Queue mutations
  const addToQueueMutation = useAddToQueue()
  const startQueueMutation = useStartQueue()
  const stopQueueMutation = useStopQueue()
  const clearQueueMutation = useClearQueue()

  // Other mutations
  const deleteModelMutation = useDeleteTransformerModel()

  const models = modelsData?.models || []
  const isQueueRunning = queueStatus?.queue_running || false
  const queueCards = queueStatus?.cards || []

  // Card management
  const addCard = () => {
    setCards([...cards, createDefaultCard()])
  }

  const removeCard = (id: string) => {
    if (cards.length > 1) {
      setCards(cards.filter((c) => c.id !== id))
    }
  }

  const updateCard = (id: string, updates: Partial<TrainingCardConfig>) => {
    setCards(cards.map((c) => {
      if (c.id !== id) return c

      const newCard = { ...c, ...updates }

      // Auto-calculate sequence length when parquet or session changes
      if ('parquetFile' in updates || 'sessionOption' in updates) {
        const tf = extractTimeframeFromFilename(newCard.parquetFile)
        newCard.sequenceLength = calculateSequenceLength(newCard.sessionOption, tf)
      }

      return newCard
    }))
  }

  // Auto-generate model name for a card
  const getAutoModelName = (card: TrainingCardConfig) => {
    const session = card.sessionOption.replace('_', '+')
    const tf = extractTimeframeFromFilename(card.parquetFile) || 'M5'
    return `ADR20_${tf}_${session}`
  }

  // Get timeframe from card's parquet file
  const getCardTimeframe = (card: TrainingCardConfig): string | null => {
    return extractTimeframeFromFilename(card.parquetFile)
  }

  // Check if all cards have a parquet file selected
  const allCardsHaveParquet = cards.every((c) => c.parquetFile !== '')

  // Handle starting queue with all cards
  const handleStartAll = async () => {
    // Validate all cards have parquet file selected
    if (!allCardsHaveParquet) {
      alert('Please select a parquet file for all training configurations')
      return
    }

    // Add all cards to queue
    for (const card of cards) {
      const modelName = card.modelName || getAutoModelName(card)
      await addToQueueMutation.mutateAsync({
        card_id: card.id,
        model_name: modelName,
        session_option: card.sessionOption,
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
    // Start the queue
    startQueueMutation.mutate(workingDirectory)
  }

  const handleStopQueue = () => {
    stopQueueMutation.mutate()
  }

  const handleClearQueue = () => {
    clearQueueMutation.mutate()
  }

  const handleDeleteModel = (name: string) => {
    if (confirm(`Delete model "${name}"? This cannot be undone.`)) {
      deleteModelMutation.mutate(
        { modelName: name, workingDirectory },
        { onSuccess: () => refetchModels() }
      )
    }
  }

  // Get status display for queue
  const getQueueStatusDisplay = () => {
    if (isQueueRunning) {
      return {
        icon: Activity,
        color: 'text-amber-400',
        bg: 'bg-amber-500/10 border-amber-500/30',
        text: 'Queue Running',
        animate: true,
      }
    }
    if (queueCards.some((c) => c.status === 'error')) {
      return {
        icon: AlertCircle,
        color: 'text-red-400',
        bg: 'bg-red-500/10 border-red-500/30',
        text: 'Error',
        animate: false,
      }
    }
    return {
      icon: CheckCircle2,
      color: 'text-teal-400',
      bg: 'bg-teal-500/10 border-teal-500/30',
      text: 'Ready',
      animate: false,
    }
  }

  const queueStatusDisplay = getQueueStatusDisplay()
  const QueueStatusIcon = queueStatusDisplay.icon

  // Find currently training card from queue
  const currentTrainingCard = queueCards.find((c) => c.status === 'training')
  const pendingCount = queueCards.filter((c) => c.status === 'pending').length
  const completedCount = queueCards.filter((c) => c.status === 'completed').length

  return (
    <div className="space-y-3">
      {/* Header Card - Queue Status */}
      <Card className="border-border/50 bg-gradient-to-br from-card to-card/80">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Zap className="h-4 w-4 text-amber-400" />
            <span className="bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text">
              Transformer Training
            </span>
            {queueCards.length > 0 && (
              <span className="text-xs text-muted-foreground ml-auto">
                {pendingCount} pending, {completedCount} done
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Queue Status Indicator */}
          <div
            className={`flex items-center gap-2 p-2.5 rounded-md border ${queueStatusDisplay.bg} ${queueStatusDisplay.color}`}
          >
            <QueueStatusIcon className={`h-4 w-4 ${queueStatusDisplay.animate ? 'animate-spin' : ''}`} />
            <span className="text-sm font-medium">{queueStatusDisplay.text}</span>
            {currentTrainingCard && (
              <span className="text-xs opacity-70 ml-auto truncate max-w-[120px]">
                {currentTrainingCard.model_name}
              </span>
            )}
          </div>

          {/* Currently Training Progress */}
          {currentTrainingCard && (
            <div className="space-y-2 p-3 rounded-md bg-secondary/30 border border-amber-500/30">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">
                  Epoch {currentTrainingCard.current_epoch}/{currentTrainingCard.total_epochs}
                </span>
                <span className="font-mono text-amber-400 flex items-center gap-1">
                  <Timer className="h-3 w-3" />
                  {formatElapsed(currentTrainingCard.elapsed_seconds)}
                </span>
              </div>
              <div className="w-full bg-secondary rounded-full h-1.5 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-amber-500 to-amber-400 h-1.5 rounded-full transition-all duration-300"
                  style={{
                    width: `${
                      currentTrainingCard.total_epochs > 0
                        ? (currentTrainingCard.current_epoch / currentTrainingCard.total_epochs) * 100
                        : 0
                    }%`,
                  }}
                />
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs mt-2">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Train:</span>
                  <span className="font-mono">{currentTrainingCard.train_loss?.toFixed(4) || '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Val:</span>
                  <span className="font-mono">{currentTrainingCard.val_loss?.toFixed(4) || '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Best:</span>
                  <span className="font-mono text-teal-400">
                    {currentTrainingCard.best_loss?.toFixed(4) || '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Dir Acc:</span>
                  <span className="font-mono text-blue-400">
                    {currentTrainingCard.directional_accuracy != null
                      ? `${currentTrainingCard.directional_accuracy.toFixed(1)}%`
                      : '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">R²:</span>
                  <span className="font-mono text-purple-400">
                    {currentTrainingCard.r_squared?.toFixed(3) || '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Grad Norm:</span>
                  <span className="font-mono">{currentTrainingCard.grad_norm?.toFixed(2) || '—'}</span>
                </div>
              </div>
            </div>
          )}

          {/* Training Cards */}
          {cards.map((card, idx) => (
            <div
              key={card.id}
              className="p-3 rounded-md bg-secondary/20 border border-border/30 space-y-3"
            >
              {/* Card Header */}
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">Config {idx + 1}</span>
                {cards.length > 1 && !isQueueRunning && (
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-5 w-5 p-0"
                    onClick={() => removeCard(card.id)}
                  >
                    <XCircle className="h-3 w-3 text-muted-foreground hover:text-red-400" />
                  </Button>
                )}
              </div>

              {/* Parquet File Selection */}
              <div className="space-y-1">
                <label className="text-[9px] text-muted-foreground flex items-center gap-1">
                  <Database className="h-3 w-3" />
                  Training Data
                </label>
                <select
                  value={card.parquetFile}
                  onChange={(e) => updateCard(card.id, { parquetFile: e.target.value })}
                  disabled={isQueueRunning}
                  className={`w-full h-7 text-xs bg-background/50 border rounded px-2 font-mono ${
                    card.parquetFile === ''
                      ? 'border-amber-500/50 text-muted-foreground'
                      : 'border-border/50 text-foreground'
                  } ${isQueueRunning ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <option value="">-- Select Parquet --</option>
                  {parquetFiles.map((pf) => (
                    <option key={pf.name} value={pf.name}>
                      {pf.name} ({pf.rows?.toLocaleString() || '?'} rows)
                    </option>
                  ))}
                </select>
              </div>

              {/* Session Selection - 3x2 Grid */}
              <div className="grid grid-cols-3 gap-1.5">
                {SESSION_OPTIONS.map((opt) => (
                  <label
                    key={opt.value}
                    className={`flex items-center justify-center p-1.5 rounded text-[10px] cursor-pointer transition-all ${
                      card.sessionOption === opt.value
                        ? 'bg-amber-500/20 border border-amber-500/50 text-amber-400'
                        : 'bg-secondary/30 border border-border/30 text-muted-foreground hover:border-border'
                    } ${isQueueRunning ? 'opacity-50 cursor-not-allowed' : ''}`}
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

              {/* Model Name */}
              <Input
                value={card.modelName}
                onChange={(e) => updateCard(card.id, { modelName: e.target.value })}
                placeholder={getAutoModelName(card)}
                disabled={isQueueRunning}
                className="text-xs h-7 bg-background/50 border-border/50 font-mono"
              />

              {/* Sequence length info - auto-calculated */}
              {card.parquetFile && (
                <div className="flex items-center gap-2 text-[9px] text-muted-foreground bg-secondary/20 px-2 py-1 rounded">
                  <span>TF: <span className="text-amber-400 font-mono">{getCardTimeframe(card) || '?'}</span></span>
                  <span>•</span>
                  <span>Bars: <span className="text-teal-400 font-mono">{card.sequenceLength}</span></span>
                  <span className="text-[8px] opacity-60">({SESSION_HOURS[card.sessionOption]}h × {BARS_PER_HOUR[getCardTimeframe(card) || 'M5'] || '?'}/h)</span>
                </div>
              )}

              {/* Core Parameters - 3 col grid (removed Seq since it's auto-calculated) */}
              <div className="grid grid-cols-3 gap-1.5">
                <div className="space-y-0.5">
                  <label className="text-[9px] text-muted-foreground">Batch</label>
                  <Input
                    type="number"
                    value={card.batchSize}
                    onChange={(e) =>
                      updateCard(card.id, { batchSize: parseInt(e.target.value) || 32 })
                    }
                    disabled={isQueueRunning}
                    className="text-[10px] h-6 bg-background/50 border-border/50 font-mono px-1"
                  />
                </div>
                <div className="space-y-0.5">
                  <label className="text-[9px] text-muted-foreground">Epochs</label>
                  <Input
                    type="number"
                    value={card.numEpochs}
                    onChange={(e) =>
                      updateCard(card.id, { numEpochs: parseInt(e.target.value) || 100 })
                    }
                    disabled={isQueueRunning}
                    className="text-[10px] h-6 bg-background/50 border-border/50 font-mono px-1"
                  />
                </div>
                <div className="space-y-0.5">
                  <label className="text-[9px] text-muted-foreground">LR</label>
                  <Input
                    type="number"
                    value={card.learningRate}
                    onChange={(e) =>
                      updateCard(card.id, { learningRate: parseFloat(e.target.value) || 0.0001 })
                    }
                    step="0.0001"
                    disabled={isQueueRunning}
                    className="text-[10px] h-6 bg-background/50 border-border/50 font-mono px-1"
                  />
                </div>
              </div>

              {/* Advanced Toggle */}
              <button
                onClick={() => updateCard(card.id, { advancedOpen: !card.advancedOpen })}
                className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground"
              >
                {card.advancedOpen ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
                <Settings2 className="h-3 w-3" />
                Advanced
              </button>

              {card.advancedOpen && (
                <div className="grid grid-cols-3 gap-1.5">
                  <div className="space-y-0.5">
                    <label className="text-[9px] text-muted-foreground">d_model</label>
                    <Input
                      type="number"
                      value={card.dModel}
                      onChange={(e) =>
                        updateCard(card.id, { dModel: parseInt(e.target.value) || 128 })
                      }
                      disabled={isQueueRunning}
                      className="text-[10px] h-6 bg-background/50 border-border/50 font-mono px-1"
                    />
                  </div>
                  <div className="space-y-0.5">
                    <label className="text-[9px] text-muted-foreground">n_layers</label>
                    <Input
                      type="number"
                      value={card.nLayers}
                      onChange={(e) =>
                        updateCard(card.id, { nLayers: parseInt(e.target.value) || 4 })
                      }
                      disabled={isQueueRunning}
                      className="text-[10px] h-6 bg-background/50 border-border/50 font-mono px-1"
                    />
                  </div>
                  <div className="space-y-0.5">
                    <label className="text-[9px] text-muted-foreground">n_heads</label>
                    <Input
                      type="number"
                      value={card.nHeads}
                      onChange={(e) =>
                        updateCard(card.id, { nHeads: parseInt(e.target.value) || 4 })
                      }
                      disabled={isQueueRunning}
                      className="text-[10px] h-6 bg-background/50 border-border/50 font-mono px-1"
                    />
                  </div>
                  <div className="space-y-0.5">
                    <label className="text-[9px] text-muted-foreground">Dropout</label>
                    <Input
                      type="number"
                      value={card.dropoutRate}
                      onChange={(e) =>
                        updateCard(card.id, { dropoutRate: parseFloat(e.target.value) || 0.1 })
                      }
                      step="0.05"
                      disabled={isQueueRunning}
                      className="text-[10px] h-6 bg-background/50 border-border/50 font-mono px-1"
                    />
                  </div>
                  <div className="space-y-0.5">
                    <label className="text-[9px] text-muted-foreground">Early Stop</label>
                    <Input
                      type="number"
                      value={card.earlyStoppingPatience}
                      onChange={(e) =>
                        updateCard(card.id, { earlyStoppingPatience: parseInt(e.target.value) || 15 })
                      }
                      disabled={isQueueRunning}
                      className="text-[10px] h-6 bg-background/50 border-border/50 font-mono px-1"
                    />
                  </div>
                  <div className="flex items-end pb-0.5">
                    <label className="flex items-center gap-1 text-[9px] text-muted-foreground cursor-pointer">
                      <input
                        type="checkbox"
                        checked={card.saveToModels}
                        onChange={(e) => updateCard(card.id, { saveToModels: e.target.checked })}
                        disabled={isQueueRunning}
                        className="h-3 w-3 rounded"
                      />
                      Save
                    </label>
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Add Card Button */}
          {!isQueueRunning && (
            <Button
              variant="outline"
              size="sm"
              className="w-full h-7 text-xs border-dashed"
              onClick={addCard}
            >
              <Plus className="h-3 w-3 mr-1" />
              Add Configuration
            </Button>
          )}

          {/* Action Buttons */}
          <div className="pt-1 space-y-2">
            {isQueueRunning ? (
              <Button variant="destructive" className="w-full h-9" onClick={handleStopQueue}>
                <Square className="mr-2 h-4 w-4" />
                Stop Queue
              </Button>
            ) : (
              <Button
                className={`w-full h-9 text-white shadow-lg ${
                  allCardsHaveParquet
                    ? 'bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400 shadow-amber-500/20'
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
              <Button
                variant="outline"
                size="sm"
                className="w-full h-7 text-xs"
                onClick={handleClearQueue}
              >
                Clear Queue
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Queue Progress Cards */}
      {queueCards.length > 0 && (
        <Card className="border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs flex items-center gap-2 text-muted-foreground">
              <Activity className="h-3.5 w-3.5" />
              Queue ({queueCards.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1.5 max-h-40 overflow-y-auto">
              {queueCards.map((qc) => (
                <div
                  key={qc.card_id}
                  className={`p-2 rounded-md border ${
                    qc.status === 'training'
                      ? 'bg-amber-500/10 border-amber-500/30'
                      : qc.status === 'completed'
                        ? 'bg-teal-500/10 border-teal-500/30'
                        : qc.status === 'error'
                          ? 'bg-red-500/10 border-red-500/30'
                          : 'bg-secondary/30 border-border/30'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium truncate">{qc.model_name}</span>
                    <span
                      className={`text-[10px] ${
                        qc.status === 'training'
                          ? 'text-amber-400'
                          : qc.status === 'completed'
                            ? 'text-teal-400'
                            : qc.status === 'error'
                              ? 'text-red-400'
                              : 'text-muted-foreground'
                      }`}
                    >
                      {qc.status === 'training'
                        ? `${qc.current_epoch}/${qc.total_epochs}`
                        : qc.status}
                    </span>
                  </div>
                  {qc.status === 'completed' && qc.final_directional_accuracy != null && (
                    <div className="flex gap-2 text-[10px] mt-1">
                      <span className="text-blue-400">
                        Dir: {qc.final_directional_accuracy.toFixed(1)}%
                      </span>
                      <span className="text-purple-400">R²: {qc.final_r_squared?.toFixed(3)}</span>
                      <span className="text-muted-foreground">
                        {formatElapsed(qc.elapsed_seconds)}
                      </span>
                    </div>
                  )}
                  {qc.status === 'error' && qc.error_message && (
                    <div className="text-[10px] text-red-400 mt-1 truncate">{qc.error_message}</div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Saved Models Card */}
      {models.length > 0 && (
        <Card className="border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs flex items-center gap-2 text-muted-foreground">
              <Layers className="h-3.5 w-3.5" />
              Saved Models
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1.5 max-h-40 overflow-y-auto">
              {models.map((model) => (
                <div
                  key={model.name}
                  className="flex items-center justify-between p-2 rounded-md bg-secondary/30 border border-border/30 group"
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium truncate">{model.name}</div>
                    <div className="flex gap-2 text-[10px] text-muted-foreground">
                      {model.best_loss && (
                        <span className="text-teal-400">Loss: {model.best_loss.toFixed(4)}</span>
                      )}
                      {model.epochs_trained > 0 && <span>{model.epochs_trained} epochs</span>}
                      {model.target_session && <span>{model.target_session}</span>}
                    </div>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={() => handleDeleteModel(model.name)}
                    disabled={deleteModelMutation.isPending}
                  >
                    <Trash2 className="h-3 w-3 text-red-400" />
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
