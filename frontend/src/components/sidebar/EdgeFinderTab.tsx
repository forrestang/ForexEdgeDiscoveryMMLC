import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import {
  useAutoSetupStatus,
  useAutoSetup,
  useStopTraining,
  useEdgeFinderModels,
  useCopyModel,
  useDeleteModel,
} from '@/hooks/useEdgeFinder'
import { usePersistedState } from '@/hooks/usePersistedSettings'
import {
  Loader2,
  Brain,
  CheckCircle2,
  AlertCircle,
  XCircle,
  Play,
  Square,
  ChevronDown,
  ChevronRight,
  Copy,
  Trash2,
  RefreshCw,
} from 'lucide-react'
import type { EdgeProbabilities } from '@/types'

interface EdgeFinderTabProps {
  workingDirectory: string
  edgeProbabilities: EdgeProbabilities | null
  onEdgeProbabilitiesChange: (edge: EdgeProbabilities | null) => void
  kNeighbors: number
  onKNeighborsChange: (k: number) => void
}

export function EdgeFinderTab({
  workingDirectory,
  edgeProbabilities,
  kNeighbors,
  onKNeighborsChange,
}: EdgeFinderTabProps) {
  // Persisted settings
  const [modelName, setModelName] = usePersistedState('modelName', 'vae_default')
  const [trainingEpochs, setTrainingEpochs] = usePersistedState('trainingEpochs', 100)
  const [latentDim, setLatentDim] = usePersistedState('latentDim', 32)
  const [advancedOpen, setAdvancedOpen] = usePersistedState('advancedEdgeFinderOpen', false)

  // Local state
  const [saveAsName, setSaveAsName] = useState('')

  // Queries
  const { data: status, isLoading: statusLoading } = useAutoSetupStatus(workingDirectory)
  const { data: modelsData } = useEdgeFinderModels(workingDirectory)

  // Mutations
  const autoSetupMutation = useAutoSetup()
  const stopTrainingMutation = useStopTraining()
  const copyModelMutation = useCopyModel()
  const deleteModelMutation = useDeleteModel()

  const models = modelsData?.models || []

  const handleSetup = (forceRetrain = false) => {
    autoSetupMutation.mutate({
      working_directory: workingDirectory,
      model_name: modelName,
      force_retrain: forceRetrain,
      num_epochs: trainingEpochs,
      latent_dim: latentDim,
    })
  }

  const handleStopTraining = () => {
    stopTrainingMutation.mutate()
  }

  const handleSaveAs = () => {
    if (!saveAsName || !status?.model_name) return
    copyModelMutation.mutate({
      modelName: status.model_name,
      newName: saveAsName,
      workingDirectory,
    })
    setSaveAsName('')
  }

  const handleDeleteModel = (name: string) => {
    if (confirm(`Delete model "${name}"? This cannot be undone.`)) {
      deleteModelMutation.mutate({
        modelName: name,
        workingDirectory,
      })
    }
  }

  // Determine status display
  const getStatusDisplay = () => {
    if (statusLoading) {
      return { icon: Loader2, color: 'text-muted-foreground', text: 'Loading...', animate: true }
    }

    switch (status?.status) {
      case 'ready':
        return { icon: CheckCircle2, color: 'text-green-500', text: 'Ready for Inference', animate: false }
      case 'training':
        return { icon: Brain, color: 'text-yellow-500', text: 'Training in Progress', animate: true }
      case 'building_index':
        return { icon: RefreshCw, color: 'text-blue-500', text: 'Building Index', animate: true }
      case 'loading':
        return { icon: Loader2, color: 'text-purple-500', text: 'Loading Model & Index', animate: true }
      case 'checking':
        return { icon: AlertCircle, color: 'text-yellow-500', text: status.message, animate: false }
      case 'error':
        return { icon: XCircle, color: 'text-red-500', text: status.message, animate: false }
      default:
        return { icon: AlertCircle, color: 'text-muted-foreground', text: 'Unknown', animate: false }
    }
  }

  const statusDisplay = getStatusDisplay()
  const StatusIcon = statusDisplay.icon

  return (
    <div className="space-y-4">
      {/* Main Status Card */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Edge Finder
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Status Indicator */}
          <div className={`flex items-center gap-2 p-3 rounded-lg bg-secondary/50 ${statusDisplay.color}`}>
            <StatusIcon className={`h-5 w-5 ${statusDisplay.animate ? 'animate-spin' : ''}`} />
            <span className="font-medium">{statusDisplay.text}</span>
          </div>

          {/* Model Info (when ready) */}
          {status?.status === 'ready' && status.model_name && (
            <div className="space-y-1 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Model:</span>
                <span className="font-medium">{status.model_name}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Vectors:</span>
                <span>{status.num_vectors.toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Sessions:</span>
                <span>{status.num_sessions}</span>
              </div>
            </div>
          )}

          {/* Training Complete Summary */}
          {status?.status === 'ready' && status.last_training_completed && (
            <div className="p-3 rounded-lg bg-green-500/10 border border-green-500/30 space-y-1">
              <div className="flex items-center gap-2 text-green-500 font-medium text-sm">
                <CheckCircle2 className="h-4 w-4" />
                Training Complete
              </div>
              <div className="text-xs text-muted-foreground">
                Epochs: {status.last_training_epochs} | Best Loss: {status.last_training_best_loss.toFixed(4)}
              </div>
            </div>
          )}

          {/* Training Progress */}
          {status?.status === 'training' && (
            <div className="space-y-2">
              <div className="w-full bg-secondary rounded-full h-2">
                <div
                  className="bg-yellow-500 h-2 rounded-full transition-all"
                  style={{ width: `${(status.training_epoch / status.training_total_epochs) * 100}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Training: Epoch {status.training_epoch}/{status.training_total_epochs}</span>
                <span>Loss: {status.training_loss.toFixed(4)}</span>
              </div>
            </div>
          )}

          {/* Index Building Progress */}
          {status?.status === 'building_index' && (
            <div className="space-y-2">
              <div className="w-full bg-secondary rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all"
                  style={{ width: `${(status.index_progress || 0) * 100}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Building index: {status.index_current_session || 0}/{status.index_total_sessions || 0} sessions</span>
                <span>{Math.round((status.index_progress || 0) * 100)}%</span>
              </div>
            </div>
          )}

          {/* Loading Progress */}
          {status?.status === 'loading' && (
            <div className="space-y-2">
              <div className="w-full bg-secondary rounded-full h-2">
                <div
                  className="bg-purple-500 h-2 rounded-full transition-all"
                  style={{ width: `${(status.index_progress || 0) * 100}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{status.message || 'Loading...'}</span>
                <span>{Math.round((status.index_progress || 0) * 100)}%</span>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-2">
            {status?.status === 'training' ? (
              <Button
                variant="destructive"
                className="w-full"
                onClick={handleStopTraining}
                disabled={stopTrainingMutation.isPending}
              >
                <Square className="mr-2 h-4 w-4" />
                Stop Training
              </Button>
            ) : status?.status === 'building_index' ? (
              <Button
                variant="outline"
                className="w-full"
                disabled={true}
              >
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Building Index...
              </Button>
            ) : status?.status === 'loading' ? (
              <Button
                variant="outline"
                className="w-full"
                disabled={true}
              >
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading...
              </Button>
            ) : status?.status === 'ready' ? (
              <Button
                variant="outline"
                className="w-full"
                onClick={() => handleSetup(true)}
                disabled={autoSetupMutation.isPending}
              >
                <RefreshCw className="mr-2 h-4 w-4" />
                Retrain Model
              </Button>
            ) : (
              <Button
                className="w-full"
                onClick={() => handleSetup(false)}
                disabled={autoSetupMutation.isPending || status?.status === 'training' || status?.status === 'building_index' || status?.status === 'loading'}
              >
                {autoSetupMutation.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Play className="mr-2 h-4 w-4" />
                )}
                Setup Model
              </Button>
            )}
          </div>

          {/* Save Model As */}
          {status?.status === 'ready' && status.model_name && (
            <div className="flex gap-2">
              <Input
                value={saveAsName}
                onChange={(e) => setSaveAsName(e.target.value)}
                placeholder="Save model as..."
                className="text-xs"
              />
              <Button
                size="sm"
                variant="outline"
                onClick={handleSaveAs}
                disabled={!saveAsName || copyModelMutation.isPending}
              >
                <Copy className="h-4 w-4" />
              </Button>
            </div>
          )}

          {/* Advanced Options */}
          <div className="border-t pt-2">
            <button
              onClick={() => setAdvancedOpen(!advancedOpen)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
            >
              {advancedOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
              Advanced Options
            </button>
            {advancedOpen && (
              <div className="mt-3 space-y-3">
                {/* Model Name */}
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Model Name</label>
                  <Input
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    className="text-xs"
                  />
                </div>

                {/* Training Epochs */}
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Training Epochs</label>
                  <Input
                    type="number"
                    value={trainingEpochs}
                    onChange={(e) => setTrainingEpochs(parseInt(e.target.value) || 100)}
                    min={10}
                    max={500}
                    className="text-xs"
                  />
                </div>

                {/* Latent Dimension */}
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Latent Dimension</label>
                  <Input
                    type="number"
                    value={latentDim}
                    onChange={(e) => setLatentDim(parseInt(e.target.value) || 32)}
                    min={8}
                    max={128}
                    className="text-xs"
                  />
                </div>

                {/* K-Neighbors */}
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">K-Neighbors</label>
                  <Input
                    type="number"
                    value={kNeighbors}
                    onChange={(e) => onKNeighborsChange(parseInt(e.target.value) || 500)}
                    min={10}
                    max={1000}
                    className="text-xs"
                  />
                </div>

                {/* Saved Models List */}
                {models.length > 0 && (
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">Saved Models</label>
                    <div className="space-y-1 max-h-32 overflow-y-auto">
                      {models.map((model) => (
                        <div
                          key={model.model_name}
                          className={`flex items-center justify-between p-2 rounded text-xs ${
                            model.is_active ? 'bg-primary/10 border border-primary/30' : 'bg-secondary/50'
                          }`}
                        >
                          <div>
                            <span className="font-medium">{model.model_name}</span>
                            <span className="text-muted-foreground ml-2">
                              ({model.trained_epochs} epochs)
                            </span>
                          </div>
                          <div className="flex gap-1">
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-6 w-6 p-0"
                              onClick={() => handleDeleteModel(model.model_name)}
                              disabled={model.is_active}
                            >
                              <Trash2 className="h-3 w-3 text-red-500" />
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Edge Probabilities Display */}
      {edgeProbabilities && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Edge Probabilities</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Pattern Matches:</span>
                <span className="font-medium">500 → {edgeProbabilities.num_matches} unique</span>
              </div>

              <div className="border-t border-border pt-2 mt-2">
                <div className="text-muted-foreground mb-1">Next Bar</div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Up %:</span>
                  <span
                    className={
                      edgeProbabilities.next_bar_up_pct > 0.55
                        ? 'text-green-500'
                        : edgeProbabilities.next_bar_up_pct < 0.45
                        ? 'text-red-500'
                        : ''
                    }
                  >
                    {(edgeProbabilities.next_bar_up_pct * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Avg Move:</span>
                  <span>{edgeProbabilities.next_bar_avg_move.toFixed(3)} ATR</span>
                </div>
              </div>

              <div className="border-t border-border pt-2 mt-2">
                <div className="text-muted-foreground mb-1">Session</div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Up %:</span>
                  <span
                    className={
                      edgeProbabilities.session_up_pct > 0.55
                        ? 'text-green-500'
                        : edgeProbabilities.session_up_pct < 0.45
                        ? 'text-red-500'
                        : ''
                    }
                  >
                    {(edgeProbabilities.session_up_pct * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Avg Drift:</span>
                  <span>{edgeProbabilities.session_avg_drift.toFixed(3)} ATR</span>
                </div>
              </div>

              <div className="border-t border-border pt-2 mt-2">
                <div className="text-muted-foreground mb-1">Risk/Reward</div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Avg MAE:</span>
                  <span className="text-red-400">{edgeProbabilities.avg_mae.toFixed(2)} ATR</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Avg MFE:</span>
                  <span className="text-green-400">{edgeProbabilities.avg_mfe.toFixed(2)} ATR</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">R:R Ratio:</span>
                  <span
                    className={
                      edgeProbabilities.risk_reward_ratio > 1.5
                        ? 'text-green-500 font-medium'
                        : edgeProbabilities.risk_reward_ratio < 0.8
                        ? 'text-red-500'
                        : ''
                    }
                  >
                    {edgeProbabilities.risk_reward_ratio.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Ready Status */}
      {status?.status === 'ready' && !edgeProbabilities && (
        <Card>
          <CardContent className="p-3">
            <div className="flex items-center gap-2 text-xs text-green-500">
              <CheckCircle2 className="h-4 w-4" />
              Index ready - Load a chart to see edge probabilities
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
