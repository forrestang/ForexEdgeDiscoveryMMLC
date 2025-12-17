import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { TIMEFRAME_OPTIONS, SESSION_OPTIONS } from '@/lib/constants'
import { useAvailablePairs, useProcessAll, useProcessAllStatus } from '@/hooks/usePipeline'
import { usePersistedState } from '@/hooks/usePersistedSettings'
import { Loader2, Play, CheckCircle, AlertCircle, ChevronDown, ChevronRight } from 'lucide-react'

interface PipelineTabProps {
  workingDirectory: string
  setWorkingDirectory: (dir: string) => void
}

export function PipelineTab({ workingDirectory, setWorkingDirectory }: PipelineTabProps) {
  // Persisted settings
  const [selectedPairs, setSelectedPairs] = usePersistedState<string[]>('selectedPairs', [])
  const [sessionType, setSessionType] = usePersistedState('sessionType', 'ny')
  const [timeframe, setTimeframe] = usePersistedState('timeframe', 'M10')
  const [advancedOpen, setAdvancedOpen] = usePersistedState('advancedPipelineOpen', false)

  // Advanced options
  const [forceSessions, setForceSessions] = useState(false)

  // Queries
  const { data: availablePairs, isLoading: pairsLoading } = useAvailablePairs(workingDirectory)
  const { data: status } = useProcessAllStatus()
  const processAllMutation = useProcessAll()

  // Auto-select all pairs when first loaded
  useEffect(() => {
    if (availablePairs?.pairs && selectedPairs.length === 0) {
      setSelectedPairs(availablePairs.pairs)
    }
  }, [availablePairs?.pairs])

  const handlePairToggle = (pair: string) => {
    setSelectedPairs(prev =>
      prev.includes(pair)
        ? prev.filter(p => p !== pair)
        : [...prev, pair]
    )
  }

  const handleSelectAll = () => {
    if (availablePairs?.pairs) {
      setSelectedPairs(availablePairs.pairs)
    }
  }

  const handleSelectNone = () => {
    setSelectedPairs([])
  }

  const handleProcess = () => {
    processAllMutation.mutate({
      working_directory: workingDirectory,
      pairs: selectedPairs.length > 0 ? selectedPairs : undefined,
      session_type: sessionType,
      timeframe: timeframe,
      force_sessions: forceSessions,
    })
  }

  const isProcessing = status?.is_processing || processAllMutation.isPending

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Data Pipeline</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Working Directory */}
          <div className="space-y-2">
            <label className="text-xs text-muted-foreground">Working Directory</label>
            <Input
              value={workingDirectory}
              onChange={(e) => setWorkingDirectory(e.target.value)}
              placeholder="C:\Users\lawfp\Desktop\Data4"
              className="text-xs"
            />
          </div>

          {/* Pair Selection */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs text-muted-foreground">Currency Pairs</label>
              <div className="flex gap-1">
                <button
                  onClick={handleSelectAll}
                  className="text-xs text-primary hover:underline"
                >
                  All
                </button>
                <span className="text-xs text-muted-foreground">|</span>
                <button
                  onClick={handleSelectNone}
                  className="text-xs text-primary hover:underline"
                >
                  None
                </button>
              </div>
            </div>
            {pairsLoading ? (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" />
                Loading pairs...
              </div>
            ) : availablePairs?.pairs && availablePairs.pairs.length > 0 ? (
              <div className="grid grid-cols-2 gap-1">
                {availablePairs.pairs.map(pair => (
                  <label
                    key={pair}
                    className="flex items-center gap-2 text-xs cursor-pointer hover:bg-secondary/50 p-1 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={selectedPairs.includes(pair)}
                      onChange={() => handlePairToggle(pair)}
                      className="rounded"
                    />
                    <span>{pair}</span>
                    <span className="text-muted-foreground">
                      ({availablePairs.csv_counts[pair]} files)
                    </span>
                  </label>
                ))}
              </div>
            ) : (
              <div className="text-xs text-muted-foreground">
                No CSV files found. Check your working directory.
              </div>
            )}
          </div>

          {/* Session Type */}
          <div className="space-y-2">
            <label className="text-xs text-muted-foreground">Session Type</label>
            <Select
              value={sessionType}
              onChange={(e) => setSessionType(e.target.value)}
              options={SESSION_OPTIONS.map(s => ({ value: s.value, label: s.label }))}
            />
          </div>

          {/* Timeframe */}
          <div className="space-y-2">
            <label className="text-xs text-muted-foreground">Timeframe</label>
            <Select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              options={TIMEFRAME_OPTIONS.map(t => ({ value: t.value, label: t.label }))}
            />
          </div>

          {/* Advanced Options */}
          <div className="border-t pt-2">
            <button
              onClick={() => setAdvancedOpen(!advancedOpen)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
            >
              {advancedOpen ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronRight className="h-3 w-3" />
              )}
              Advanced Options
            </button>
            {advancedOpen && (
              <div className="mt-2 space-y-2 pl-4">
                <label className="flex items-center gap-2 text-xs cursor-pointer">
                  <input
                    type="checkbox"
                    checked={forceSessions}
                    onChange={(e) => setForceSessions(e.target.checked)}
                    className="rounded"
                  />
                  Force regenerate sessions
                </label>
              </div>
            )}
          </div>

          {/* Process Button */}
          <Button
            onClick={handleProcess}
            disabled={isProcessing || selectedPairs.length === 0}
            className="w-full"
          >
            {isProcessing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Process Data
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Status Card */}
      {(status?.is_processing || status?.stage === 'complete') && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Status</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {status?.is_processing ? (
              <>
                <div className="flex items-center gap-2 text-sm">
                  <Loader2 className="h-4 w-4 animate-spin text-primary" />
                  <span>{status.message}</span>
                </div>
                <div className="text-xs text-muted-foreground">
                  Stage: {status.stage} | {status.pairs_completed}/{status.pairs_total} pairs
                </div>
                {status.current_pair && (
                  <div className="text-xs text-muted-foreground">
                    Current: {status.current_pair}
                  </div>
                )}
                <div className="h-2 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all duration-300"
                    style={{ width: `${status.progress * 100}%` }}
                  />
                </div>
              </>
            ) : status?.stage === 'complete' ? (
              <>
                <div className="flex items-center gap-2 text-sm">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>{status.message}</span>
                </div>
                <div className="text-xs text-muted-foreground">
                  Processed {status.pairs_total} pairs
                </div>
              </>
            ) : null}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
