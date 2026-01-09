import { useState, useMemo } from 'react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { usePersistedState } from '@/hooks/usePersistedSettings'
import {
  useLSTMDataFiles,
  useLSTMParquets,
  useDeleteLSTMParquet,
  useDeleteLSTMParquetsBatch,
  useCreateFromFiles,
  useRawParquetsForBridge,
  useBridgedParquets,
  useBridgeFiles,
  useDeleteBridgedParquet,
} from '@/hooks/useLSTM'
import { TransformerTab } from '@/components/sidebar/TransformerTab'
import { ParquetViewer } from '@/components/chart/ParquetViewer'
import {
  Settings,
  Brain,
  Database,
  FileText,
  Loader2,
  Trash2,
  Play,
  RefreshCw,
  CheckSquare,
  Square,
  ChevronDown,
  ChevronRight,
  Link,
  Zap,
  ChevronsDownUp,
  ChevronsUpDown,
  FlaskConical,
  Beaker,
} from 'lucide-react'

interface FileInfo {
  name: string
  pair: string | null
  year: number | null
  type: string
  size_mb: number
}

export function LSTMPage() {
  // Persisted tab selection
  const [activeTab, setActiveTab] = usePersistedState('lstmActiveTab', 'pipeline')

  // Persisted settings
  const [workingDirectory, setWorkingDirectory] = usePersistedState(
    'lstmWorkingDirectory',
    'C:\\Users\\lawfp\\Desktop\\Data4'
  )
  const [adrPeriod, setAdrPeriod] = usePersistedState('lstmAdrPeriod', 20)
  const [selectedTimeframes, setSelectedTimeframes] = usePersistedState<string[]>('lstmTimeframes', ['M5'])

  // Available timeframes
  const TIMEFRAMES = ['M1', 'M5', 'M10', 'M15', 'M30', 'H1', 'H4']

  // Selected files state
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set())

  // Persisted collapsible sections (stage config areas)
  const [stage1Collapsed, setStage1Collapsed] = usePersistedState('lstmStage1Collapsed', false)
  const [stage2Collapsed, setStage2Collapsed] = usePersistedState('lstmStage2Collapsed', false)

  // Persisted data panel collapsible subsections
  const [stage1DataCollapsed, setStage1DataCollapsed] = usePersistedState('lstmStage1DataCollapsed', false)
  const [stage2DataCollapsed, setStage2DataCollapsed] = usePersistedState('lstmStage2DataCollapsed', false)

  // Stage 2: Bridge selection (input selection for bridging)
  const [selectedRawParquets, setSelectedRawParquets] = useState<Set<string>>(new Set())

  // Data panel selection (for mass delete)
  const [selectedStage1Parquets, setSelectedStage1Parquets] = useState<Set<string>>(new Set())
  const [selectedBridgedParquets, setSelectedBridgedParquets] = useState<Set<string>>(new Set())

  // Persisted validation data generation collapsed state
  const [validationCollapsed, setValidationCollapsed] = usePersistedState('lstmValidationCollapsed', false)

  // Data queries
  const {
    data: dataFilesResponse,
    isLoading: filesLoading,
    refetch: refetchFiles,
  } = useLSTMDataFiles(workingDirectory)
  const {
    data: parquetsResponse,
    isLoading: parquetsLoading,
    refetch: refetchParquets,
  } = useLSTMParquets(workingDirectory)

  // Mutations
  const createFromFilesMutation = useCreateFromFiles()
  const deleteParquetMutation = useDeleteLSTMParquet()
  const deleteParquetsBatchMutation = useDeleteLSTMParquetsBatch()

  // Stage 2: Bridge queries and mutations
  const {
    data: rawParquetsResponse,
    isLoading: rawParquetsLoading,
    refetch: refetchRawParquets,
  } = useRawParquetsForBridge(workingDirectory)
  const {
    data: bridgedParquetsResponse,
    isLoading: bridgedParquetsLoading,
    refetch: refetchBridgedParquets,
  } = useBridgedParquets(workingDirectory)
  const bridgeFilesMutation = useBridgeFiles()
  const deleteBridgedMutation = useDeleteBridgedParquet()

  // Group files by pair
  const filesByPair = useMemo(() => {
    if (!dataFilesResponse?.files) return new Map<string, FileInfo[]>()

    const grouped = new Map<string, FileInfo[]>()
    for (const file of dataFilesResponse.files) {
      const pair = file.pair || 'Unknown'
      if (!grouped.has(pair)) {
        grouped.set(pair, [])
      }
      grouped.get(pair)!.push(file)
    }

    // Sort files within each pair by year
    for (const files of grouped.values()) {
      files.sort((a, b) => (a.year || 0) - (b.year || 0))
    }

    return grouped
  }, [dataFilesResponse])

  // Get sorted pair names
  const sortedPairs = useMemo(() => {
    return Array.from(filesByPair.keys()).sort()
  }, [filesByPair])

  // Selection helpers
  const toggleFile = (filename: string) => {
    setSelectedFiles((prev) => {
      const next = new Set(prev)
      if (next.has(filename)) {
        next.delete(filename)
      } else {
        next.add(filename)
      }
      return next
    })
  }

  const selectAllForPair = (pair: string) => {
    const files = filesByPair.get(pair) || []
    setSelectedFiles((prev) => {
      const next = new Set(prev)
      for (const file of files) {
        next.add(file.name)
      }
      return next
    })
  }

  const selectNoneForPair = (pair: string) => {
    const files = filesByPair.get(pair) || []
    setSelectedFiles((prev) => {
      const next = new Set(prev)
      for (const file of files) {
        next.delete(file.name)
      }
      return next
    })
  }

  const isPairFullySelected = (pair: string) => {
    const files = filesByPair.get(pair) || []
    return files.length > 0 && files.every((f) => selectedFiles.has(f.name))
  }

  const isPairPartiallySelected = (pair: string) => {
    const files = filesByPair.get(pair) || []
    const selectedCount = files.filter((f) => selectedFiles.has(f.name)).length
    return selectedCount > 0 && selectedCount < files.length
  }

  const getSelectedCountForPair = (pair: string) => {
    const files = filesByPair.get(pair) || []
    return files.filter((f) => selectedFiles.has(f.name)).length
  }

  const toggleTimeframe = (tf: string) => {
    setSelectedTimeframes((prev) => {
      if (prev.includes(tf)) {
        // Don't allow empty - keep at least one
        if (prev.length === 1) return prev
        return prev.filter((t) => t !== tf)
      }
      return [...prev, tf]
    })
  }

  const handleCreateParquets = () => {
    if (selectedFiles.size === 0 || selectedTimeframes.length === 0) return

    createFromFilesMutation.mutate({
      files: Array.from(selectedFiles),
      working_directory: workingDirectory,
      adr_period: adrPeriod,
      timeframes: selectedTimeframes,
    })
  }

  const handleDeleteParquet = (filename: string) => {
    if (confirm(`Delete "${filename}"? This cannot be undone.`)) {
      deleteParquetMutation.mutate({
        filename,
        workingDirectory,
      })
    }
  }

  const handleRefresh = () => {
    refetchFiles()
    refetchParquets()
    refetchRawParquets()
    refetchBridgedParquets()
  }

  // Stage 2: Bridge helpers
  const toggleRawParquet = (filename: string) => {
    setSelectedRawParquets((prev) => {
      const next = new Set(prev)
      if (next.has(filename)) {
        next.delete(filename)
      } else {
        next.add(filename)
      }
      return next
    })
  }

  const selectAllRawParquets = () => {
    if (rawParquetsResponse?.parquets) {
      setSelectedRawParquets(new Set(rawParquetsResponse.parquets.map((p) => p.name)))
    }
  }

  const selectNoneRawParquets = () => {
    setSelectedRawParquets(new Set())
  }

  const handleBridgeParquets = () => {
    if (selectedRawParquets.size === 0) return
    bridgeFilesMutation.mutate({
      filenames: Array.from(selectedRawParquets),
      working_directory: workingDirectory,
    })
  }

  const handleDeleteBridged = (filename: string) => {
    if (confirm(`Delete "${filename}"? This cannot be undone.`)) {
      deleteBridgedMutation.mutate({
        filename,
        workingDirectory,
      })
    }
  }

  // Data panel selection helpers - Stage 1 (raw parquets)
  const toggleStage1Parquet = (filename: string) => {
    setSelectedStage1Parquets((prev) => {
      const next = new Set(prev)
      if (next.has(filename)) {
        next.delete(filename)
      } else {
        next.add(filename)
      }
      return next
    })
  }

  const selectAllStage1Parquets = () => {
    if (parquetsResponse?.parquets) {
      setSelectedStage1Parquets(new Set(parquetsResponse.parquets.map((p) => p.name)))
    }
  }

  const selectNoneStage1Parquets = () => {
    setSelectedStage1Parquets(new Set())
  }

  // Data panel selection helpers - Stage 2 (bridged parquets)
  const toggleBridgedParquet = (filename: string) => {
    setSelectedBridgedParquets((prev) => {
      const next = new Set(prev)
      if (next.has(filename)) {
        next.delete(filename)
      } else {
        next.add(filename)
      }
      return next
    })
  }

  const selectAllBridgedParquets = () => {
    if (bridgedParquetsResponse?.parquets) {
      setSelectedBridgedParquets(new Set(bridgedParquetsResponse.parquets.map((p) => p.name)))
    }
  }

  const selectNoneBridgedParquets = () => {
    setSelectedBridgedParquets(new Set())
  }

  // Unified selection controls
  const totalSelectedFiles = selectedStage1Parquets.size + selectedBridgedParquets.size

  const selectAllDataFiles = () => {
    selectAllStage1Parquets()
    selectAllBridgedParquets()
  }

  const selectNoneDataFiles = () => {
    selectNoneStage1Parquets()
    selectNoneBridgedParquets()
  }

  const handleDeleteSelectedFiles = async () => {
    if (totalSelectedFiles === 0) return
    if (!confirm(`Delete ${totalSelectedFiles} file${totalSelectedFiles !== 1 ? 's' : ''}? This cannot be undone.`)) {
      return
    }

    // Delete Stage 1 parquets
    if (selectedStage1Parquets.size > 0) {
      deleteParquetsBatchMutation.mutate(
        {
          filenames: Array.from(selectedStage1Parquets),
          workingDirectory,
        },
        {
          onSuccess: () => {
            setSelectedStage1Parquets(new Set())
          },
        }
      )
    }

    // Delete bridged parquets one by one (no batch endpoint yet)
    for (const filename of selectedBridgedParquets) {
      deleteBridgedMutation.mutate({ filename, workingDirectory })
    }
    setSelectedBridgedParquets(new Set())
  }

  const isBridging = bridgeFilesMutation.isPending
  const isDeletingBatch = deleteParquetsBatchMutation.isPending

  const isCreating = createFromFilesMutation.isPending

  // Collapse all stages toggle
  const allStagesCollapsed = stage1Collapsed && stage2Collapsed && validationCollapsed
  const toggleAllStages = () => {
    const newState = !allStagesCollapsed
    setStage1Collapsed(newState)
    setStage2Collapsed(newState)
    setValidationCollapsed(newState)
  }

  // Count selected pairs for summary
  const selectedPairsCount = useMemo(() => {
    const pairs = new Set<string>()
    for (const filename of selectedFiles) {
      const file = dataFilesResponse?.files.find((f) => f.name === filename)
      if (file?.pair) pairs.add(file.pair)
    }
    return pairs.size
  }, [selectedFiles, dataFilesResponse])

  const canCreate = selectedFiles.size > 0 && selectedTimeframes.length > 0 && !isCreating
  const totalParquetsToCreate = selectedPairsCount * selectedTimeframes.length

  return (
    <PanelGroup direction="horizontal" className="h-full">
      {/* Resizable Sidebar */}
      <Panel defaultSize={25} minSize={15} maxSize={50}>
        <div className="h-full flex flex-col bg-card border-r border-border">
          <div className="p-4 border-b border-border">
            <h1 className="sidebar-header-title">LSTM</h1>
            <p className="sidebar-header-subtitle">Symbolic Time Series Analysis</p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col min-h-0">
            <TabsList className="w-full justify-start rounded-none border-b border-border bg-transparent p-0">
              <TabsTrigger
                value="pipeline"
                className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary"
              >
                <Settings className="h-4 w-4 mr-1" />
                Pipeline
              </TabsTrigger>
              <TabsTrigger
                value="training"
                className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary"
              >
                <Brain className="h-4 w-4 mr-1" />
                Training
              </TabsTrigger>
            </TabsList>

            <div className="flex-1 overflow-auto">
              {/* Pipeline Tab */}
              <TabsContent value="pipeline" className="mt-0 h-full flex flex-col">
                {/* Pipeline Toolbar */}
                <div className="flex items-center justify-end gap-2 px-4 pt-3 pb-1">
                  <button
                    onClick={toggleAllStages}
                    className="flex items-center gap-1.5 px-2 py-1 text-[10px] text-muted-foreground/70 hover:text-foreground rounded-md border border-transparent hover:border-border/50 hover:bg-secondary/30 transition-all duration-200"
                    title={allStagesCollapsed ? 'Expand all stages' : 'Collapse all stages'}
                  >
                    {allStagesCollapsed ? (
                      <ChevronsUpDown className="h-3 w-3" />
                    ) : (
                      <ChevronsDownUp className="h-3 w-3" />
                    )}
                    <span className="font-medium tracking-wide">
                      {allStagesCollapsed ? 'Expand' : 'Collapse'}
                    </span>
                  </button>
                  <button
                    onClick={handleRefresh}
                    className="p-1.5 rounded-md text-muted-foreground/70 hover:text-foreground border border-transparent hover:border-border/50 hover:bg-secondary/30 transition-all duration-200"
                    title="Refresh data"
                  >
                    <RefreshCw className="h-3 w-3" />
                  </button>
                </div>

                {/* Pipeline Controls Section */}
                <div className="p-4 pt-2 space-y-4">
                  <div className="flex items-center justify-between">
                    <button
                      onClick={() => setStage1Collapsed(!stage1Collapsed)}
                      className="flex items-center gap-2 hover:text-foreground transition-colors group"
                    >
                      <span className="text-muted-foreground group-hover:text-foreground transition-colors">
                        {stage1Collapsed ? (
                          <ChevronRight className="h-4 w-4" />
                        ) : (
                          <ChevronDown className="h-4 w-4" />
                        )}
                      </span>
                      <Settings className="h-4 w-4 text-primary" />
                      <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground group-hover:text-foreground transition-colors">
                        Stage 1: Data Stitching
                      </span>
                    </button>
                  </div>

                  {/* Collapsible Content */}
                  {!stage1Collapsed && (
                    <>
                      {/* Working Directory */}
                      <div className="space-y-1.5">
                        <label className="text-xs text-muted-foreground">Working Directory</label>
                        <Input
                          value={workingDirectory}
                          onChange={(e) => setWorkingDirectory(e.target.value)}
                          placeholder="C:\Users\...\Data4"
                          className="text-xs h-8 font-mono"
                        />
                      </div>

                      {/* ADR Period */}
                      <div className="space-y-1.5">
                        <label className="text-xs text-muted-foreground">ADR Lookback Period (days)</label>
                        <Input
                          type="number"
                          min={1}
                          max={100}
                          value={adrPeriod}
                          onChange={(e) => setAdrPeriod(Math.max(1, Math.min(100, parseInt(e.target.value) || 20)))}
                          className="text-xs h-8 w-24 font-mono"
                        />
                      </div>

                      {/* Timeframe Selection */}
                      <div className="space-y-1.5">
                        <label className="text-xs text-muted-foreground">Timeframes to Create</label>
                        <div className="flex flex-wrap gap-1.5">
                          {TIMEFRAMES.map((tf) => {
                            const isSelected = selectedTimeframes.includes(tf)
                            return (
                              <button
                                key={tf}
                                onClick={() => toggleTimeframe(tf)}
                                className={`px-2 py-1 text-xs rounded border transition-colors ${
                                  isSelected
                                    ? 'bg-primary/20 border-primary/50 text-primary'
                                    : 'bg-background/50 border-border/50 text-muted-foreground hover:border-primary/30 hover:text-foreground'
                                }`}
                              >
                                {tf}
                              </button>
                            )
                          })}
                        </div>
                      </div>

                      {/* Source Files Selection */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <FileText className="h-3.5 w-3.5 text-amber-500/80" />
                            <span className="text-xs font-medium text-foreground/80">Select Source Files</span>
                          </div>
                          {selectedFiles.size > 0 && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/20 text-primary">
                              {selectedFiles.size} files → {totalParquetsToCreate} parquet{totalParquetsToCreate !== 1 ? 's' : ''}
                            </span>
                          )}
                        </div>

                        <div className="space-y-2 max-h-[280px] overflow-auto pr-1">
                          {filesLoading ? (
                            <div className="flex items-center gap-2 text-xs text-muted-foreground p-3">
                              <Loader2 className="h-3 w-3 animate-spin" />
                              Loading files...
                            </div>
                          ) : sortedPairs.length > 0 ? (
                            sortedPairs.map((pair) => {
                              const files = filesByPair.get(pair) || []
                              const isFullySelected = isPairFullySelected(pair)
                              const isPartiallySelected = isPairPartiallySelected(pair)
                              const selectedCount = getSelectedCountForPair(pair)

                              return (
                                <div
                                  key={pair}
                                  className="rounded-md border border-border/40 bg-background/30 overflow-hidden"
                                >
                                  {/* Pair Header */}
                                  <div className="flex items-center justify-between px-2.5 py-1.5 bg-background/50 border-b border-border/30">
                                    <div className="flex items-center gap-2">
                                      <button
                                        onClick={() => isFullySelected ? selectNoneForPair(pair) : selectAllForPair(pair)}
                                        className="text-muted-foreground hover:text-foreground transition-colors"
                                      >
                                        {isFullySelected ? (
                                          <CheckSquare className="h-3.5 w-3.5 text-primary" />
                                        ) : isPartiallySelected ? (
                                          <div className="h-3.5 w-3.5 border border-primary rounded-sm bg-primary/30" />
                                        ) : (
                                          <Square className="h-3.5 w-3.5" />
                                        )}
                                      </button>
                                      <span className="text-xs font-semibold text-foreground/90">{pair}</span>
                                      <span className="text-[10px] text-muted-foreground">
                                        {selectedCount > 0 ? `${selectedCount}/${files.length}` : files.length}
                                      </span>
                                    </div>
                                    <div className="flex items-center gap-1">
                                      <button
                                        onClick={() => selectAllForPair(pair)}
                                        className="text-[9px] px-1 py-0.5 text-muted-foreground hover:text-primary transition-colors"
                                      >
                                        All
                                      </button>
                                      <span className="text-muted-foreground/40">|</span>
                                      <button
                                        onClick={() => selectNoneForPair(pair)}
                                        className="text-[9px] px-1 py-0.5 text-muted-foreground hover:text-primary transition-colors"
                                      >
                                        None
                                      </button>
                                    </div>
                                  </div>

                                  {/* Files List */}
                                  <div className="p-1.5 space-y-0.5">
                                    {files.map((file) => {
                                      const isSelected = selectedFiles.has(file.name)
                                      return (
                                        <label
                                          key={file.name}
                                          className={`flex items-center justify-between px-2 py-1 rounded cursor-pointer transition-colors ${
                                            isSelected
                                              ? 'bg-primary/10 border border-primary/30'
                                              : 'hover:bg-secondary/30 border border-transparent'
                                          }`}
                                        >
                                          <div className="flex items-center gap-2">
                                            <input
                                              type="checkbox"
                                              checked={isSelected}
                                              onChange={() => toggleFile(file.name)}
                                              className="rounded border-border text-primary focus:ring-primary/50 h-3 w-3"
                                            />
                                            <span className={`text-xs ${file.type === 'test' ? 'text-purple-400/80' : 'text-foreground/70'}`}>
                                              {file.year || file.name.split('.')[0]}
                                            </span>
                                            {file.type === 'test' && (
                                              <span className="text-[8px] px-1 py-0.5 rounded bg-purple-500/20 text-purple-400/80 uppercase">
                                                test
                                              </span>
                                            )}
                                          </div>
                                          <span className="text-[10px] text-muted-foreground/50">
                                            {file.size_mb.toFixed(1)}MB
                                          </span>
                                        </label>
                                      )
                                    })}
                                  </div>
                                </div>
                              )
                            })
                          ) : (
                            <div className="text-xs text-muted-foreground/60 p-3 text-center border border-dashed border-border/30 rounded">
                              No CSV files found in data folder
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Create Button */}
                      <Button
                        onClick={handleCreateParquets}
                        disabled={!canCreate}
                        className="w-full h-9 text-xs"
                      >
                        {isCreating ? (
                          <>
                            <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
                            Creating Parquets...
                          </>
                        ) : (
                          <>
                            <Play className="mr-2 h-3.5 w-3.5" />
                            {totalParquetsToCreate > 0
                              ? `Create ${totalParquetsToCreate} Parquet${totalParquetsToCreate !== 1 ? 's' : ''} (${selectedPairsCount} pair${selectedPairsCount !== 1 ? 's' : ''} × ${selectedTimeframes.length} tf)`
                              : 'Create Parquets'}
                          </>
                        )}
                      </Button>

                      {/* Status Messages */}
                      {createFromFilesMutation.isSuccess && createFromFilesMutation.data.created.length > 0 && (
                        <div className="text-xs space-y-1">
                          {createFromFilesMutation.data.created.map((p) => (
                            <div key={p.filename} className="text-emerald-500/90 bg-emerald-500/10 px-2 py-1.5 rounded border border-emerald-500/20">
                              <div>✓ {p.pair} {p.timeframe}: {p.rows.toLocaleString()} rows</div>
                              <div className="text-emerald-500/60 text-[10px]">
                                {p.start_date} → {p.end_date} | ADR{p.adr_period} | {p.trimmed_days}d trimmed
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                      {createFromFilesMutation.isSuccess && createFromFilesMutation.data.errors.length > 0 && (
                        <div className="text-xs space-y-1">
                          {createFromFilesMutation.data.errors.map((err, i) => (
                            <div key={i} className="text-red-500/90 bg-red-500/10 px-2 py-1.5 rounded border border-red-500/20">
                              ✗ {err}
                            </div>
                          ))}
                        </div>
                      )}
                      {createFromFilesMutation.isError && (
                        <div className="text-xs text-red-500/90 bg-red-500/10 px-2 py-1.5 rounded border border-red-500/20">
                          Error: {createFromFilesMutation.error.message}
                        </div>
                      )}
                    </>
                  )}
                </div>

                {/* Stage 2: Bridge Section */}
                <div className="p-4 space-y-4 border-t border-border">
                  <div className="flex items-center justify-between">
                    <button
                      onClick={() => setStage2Collapsed(!stage2Collapsed)}
                      className="flex items-center gap-2 hover:text-foreground transition-colors group"
                    >
                      <span className="text-muted-foreground group-hover:text-foreground transition-colors">
                        {stage2Collapsed ? (
                          <ChevronRight className="h-4 w-4" />
                        ) : (
                          <ChevronDown className="h-4 w-4" />
                        )}
                      </span>
                      <Link className="h-4 w-4 text-cyan-500" />
                      <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground group-hover:text-foreground transition-colors">
                        Stage 2: Bridge (MMLC Enrichment)
                      </span>
                    </button>
                  </div>

                  {!stage2Collapsed && (
                    <>
                      {/* Raw Parquets Selection */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Database className="h-3.5 w-3.5 text-emerald-500/80" />
                            <span className="text-xs font-medium text-foreground/80">Select Raw Parquets to Enrich</span>
                          </div>
                          {selectedRawParquets.size > 0 && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-400">
                              {selectedRawParquets.size} selected
                            </span>
                          )}
                        </div>

                        <div className="flex items-center gap-2 mb-2">
                          <button
                            onClick={selectAllRawParquets}
                            className="text-[10px] px-2 py-1 text-muted-foreground hover:text-primary border border-border/50 rounded hover:border-primary/50 transition-colors"
                          >
                            Select All
                          </button>
                          <button
                            onClick={selectNoneRawParquets}
                            className="text-[10px] px-2 py-1 text-muted-foreground hover:text-primary border border-border/50 rounded hover:border-primary/50 transition-colors"
                          >
                            Select None
                          </button>
                        </div>

                        <div className="space-y-1 max-h-[200px] overflow-auto pr-1">
                          {rawParquetsLoading ? (
                            <div className="flex items-center gap-2 text-xs text-muted-foreground p-3">
                              <Loader2 className="h-3 w-3 animate-spin" />
                              Loading raw parquets...
                            </div>
                          ) : rawParquetsResponse?.parquets && rawParquetsResponse.parquets.length > 0 ? (
                            rawParquetsResponse.parquets.map((parquet) => {
                              const isSelected = selectedRawParquets.has(parquet.name)
                              return (
                                <label
                                  key={parquet.name}
                                  className={`flex items-center justify-between px-2.5 py-2 rounded cursor-pointer transition-colors ${
                                    isSelected
                                      ? 'bg-cyan-500/10 border border-cyan-500/30'
                                      : 'bg-background/30 border border-border/20 hover:border-cyan-500/20'
                                  }`}
                                >
                                  <div className="flex items-center gap-2">
                                    <input
                                      type="checkbox"
                                      checked={isSelected}
                                      onChange={() => toggleRawParquet(parquet.name)}
                                      className="rounded border-border text-cyan-500 focus:ring-cyan-500/50 h-3 w-3"
                                    />
                                    <div className="flex flex-col">
                                      <div className="flex items-center gap-2">
                                        <span className="text-xs font-medium text-foreground/90">
                                          {parquet.pair || 'Unknown'}
                                        </span>
                                        {parquet.timeframe && (
                                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/20 text-primary">
                                            {parquet.timeframe}
                                          </span>
                                        )}
                                        {parquet.adr_period && (
                                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400">
                                            ADR{parquet.adr_period}
                                          </span>
                                        )}
                                      </div>
                                      <span className="text-[10px] text-muted-foreground/60">
                                        {parquet.start_date} → {parquet.end_date} | {parquet.rows.toLocaleString()} rows
                                      </span>
                                    </div>
                                  </div>
                                  <span className="text-[10px] text-muted-foreground/50">
                                    {parquet.size_mb.toFixed(1)}MB
                                  </span>
                                </label>
                              )
                            })
                          ) : (
                            <div className="text-xs text-muted-foreground/60 p-3 text-center border border-dashed border-border/30 rounded">
                              No raw parquets available. Create some in Stage 1 first.
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Bridge Button */}
                      <Button
                        onClick={handleBridgeParquets}
                        disabled={selectedRawParquets.size === 0 || isBridging}
                        className="w-full h-9 text-xs bg-cyan-600 hover:bg-cyan-700"
                      >
                        {isBridging ? (
                          <>
                            <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
                            Enriching with MMLC...
                          </>
                        ) : (
                          <>
                            <Zap className="mr-2 h-3.5 w-3.5" />
                            {selectedRawParquets.size > 0
                              ? `Enrich ${selectedRawParquets.size} Parquet${selectedRawParquets.size !== 1 ? 's' : ''} with MMLC`
                              : 'Enrich Parquets with MMLC'}
                          </>
                        )}
                      </Button>

                      {/* Bridge Status Messages */}
                      {bridgeFilesMutation.isSuccess && bridgeFilesMutation.data.results.length > 0 && (
                        <div className="text-xs space-y-1">
                          {bridgeFilesMutation.data.results.map((r, i) => (
                            <div
                              key={i}
                              className={`px-2 py-1.5 rounded border ${
                                r.status === 'success'
                                  ? 'text-cyan-500/90 bg-cyan-500/10 border-cyan-500/20'
                                  : 'text-red-500/90 bg-red-500/10 border-red-500/20'
                              }`}
                            >
                              {r.status === 'success' ? '✓' : '✗'} {r.output_filename || r.message}
                              {r.status === 'success' && (
                                <span className="text-cyan-500/60 ml-2">
                                  {r.days_processed} days, {r.processing_time_seconds.toFixed(1)}s
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                      {bridgeFilesMutation.isError && (
                        <div className="text-xs text-red-500/90 bg-red-500/10 px-2 py-1.5 rounded border border-red-500/20">
                          Error: {bridgeFilesMutation.error.message}
                        </div>
                      )}
                    </>
                  )}
                </div>

                {/* Stage 3: Generate Validation Data */}
                <div className="p-4 space-y-4 border-t border-border">
                  <div className="flex items-center justify-between">
                    <button
                      onClick={() => setValidationCollapsed(!validationCollapsed)}
                      className="flex items-center gap-2 hover:text-foreground transition-colors group"
                    >
                      <span className="text-muted-foreground group-hover:text-foreground transition-colors">
                        {validationCollapsed ? (
                          <ChevronRight className="h-4 w-4" />
                        ) : (
                          <ChevronDown className="h-4 w-4" />
                        )}
                      </span>
                      <Beaker className="h-4 w-4 text-purple-500" />
                      <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground group-hover:text-foreground transition-colors">
                        Stage 3: Generate Validation Data
                      </span>
                    </button>
                  </div>

                  {!validationCollapsed && (
                    <div className="space-y-3">
                      <p className="text-xs text-muted-foreground">
                        Generate synthetic validation data from bridged parquets to test model performance.
                      </p>

                      {/* Placeholder for validation data generation controls */}
                      <div className="text-xs text-muted-foreground/60 p-4 text-center border border-dashed border-purple-500/30 rounded bg-purple-500/5">
                        <FlaskConical className="h-8 w-8 mx-auto mb-2 text-purple-500/40" />
                        <p>Validation data generation coming soon</p>
                        <p className="text-[10px] mt-1">Will use bridged parquets to create mock data for model validation</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Pipeline Data Panel - All Outputs */}
                <div className="flex-1 p-4 overflow-auto space-y-3 border-t-2 border-border bg-background/20">
                  {/* Header with selection count */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Database className="h-4 w-4 text-primary" />
                      <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                        Pipeline Data
                      </span>
                      <span className="text-xs text-muted-foreground">
                        ({(parquetsResponse?.parquets?.length || 0) + (bridgedParquetsResponse?.parquets?.length || 0)})
                      </span>
                    </div>
                    {totalSelectedFiles > 0 && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400">
                        {totalSelectedFiles} selected
                      </span>
                    )}
                  </div>

                  {/* Global selection controls */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <button
                        onClick={selectAllDataFiles}
                        className="text-[10px] px-2 py-1 text-muted-foreground hover:text-primary border border-border/50 rounded hover:border-primary/50 transition-colors"
                      >
                        Select All
                      </button>
                      <button
                        onClick={selectNoneDataFiles}
                        className="text-[10px] px-2 py-1 text-muted-foreground hover:text-primary border border-border/50 rounded hover:border-primary/50 transition-colors"
                      >
                        Select None
                      </button>
                    </div>
                    <button
                      onClick={handleDeleteSelectedFiles}
                      disabled={totalSelectedFiles === 0 || isDeletingBatch}
                      className="flex items-center gap-1 text-[10px] px-2 py-1 text-red-400 hover:text-red-300 border border-red-500/30 rounded hover:border-red-500/50 hover:bg-red-500/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isDeletingBatch ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <Trash2 className="h-3 w-3" />
                      )}
                      Delete Selected
                    </button>
                  </div>

                  {/* Scrollable content area */}
                  <div className="space-y-3 overflow-auto pr-1" style={{ maxHeight: 'calc(100% - 80px)' }}>
                    {/* Stage 1: Raw Parquets */}
                    <div className="rounded-md border border-border/30 overflow-hidden">
                      <button
                        onClick={() => setStage1DataCollapsed(!stage1DataCollapsed)}
                        className="flex items-center justify-between w-full px-3 py-2 bg-emerald-500/5 hover:bg-emerald-500/10 transition-colors"
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground">
                            {stage1DataCollapsed ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                          </span>
                          <Settings className="h-3.5 w-3.5 text-emerald-500" />
                          <span className="text-xs font-medium text-foreground/80">Stage 1: Raw Parquets</span>
                          <span className="text-[10px] text-muted-foreground">
                            ({parquetsResponse?.parquets?.length || 0})
                          </span>
                        </div>
                        {selectedStage1Parquets.size > 0 && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400">
                            {selectedStage1Parquets.size} selected
                          </span>
                        )}
                      </button>

                      {!stage1DataCollapsed && (
                        <div className="p-2 space-y-1 max-h-[200px] overflow-auto">
                          {parquetsLoading ? (
                            <div className="flex items-center gap-2 text-xs text-muted-foreground p-2">
                              <Loader2 className="h-3 w-3 animate-spin" />
                              Loading...
                            </div>
                          ) : parquetsResponse?.parquets && parquetsResponse.parquets.length > 0 ? (
                            parquetsResponse.parquets.map((parquet) => {
                              const isSelected = selectedStage1Parquets.has(parquet.name)
                              return (
                                <label
                                  key={parquet.name}
                                  className={`flex items-center justify-between px-2.5 py-2 rounded cursor-pointer transition-colors ${
                                    isSelected
                                      ? 'bg-emerald-500/10 border border-emerald-500/30'
                                      : 'bg-background/30 border border-border/20 hover:border-emerald-500/20'
                                  }`}
                                >
                                  <div className="flex items-center gap-2">
                                    <input
                                      type="checkbox"
                                      checked={isSelected}
                                      onChange={() => toggleStage1Parquet(parquet.name)}
                                      className="rounded border-border text-emerald-500 focus:ring-emerald-500/50 h-3 w-3"
                                    />
                                    <div className="flex flex-col">
                                      <div className="flex items-center gap-2">
                                        <span className="text-xs font-medium text-emerald-500/90">
                                          {parquet.pair || 'Unknown'}
                                        </span>
                                        {parquet.timeframe && (
                                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/20 text-primary">
                                            {parquet.timeframe}
                                          </span>
                                        )}
                                        {parquet.adr_period && (
                                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400">
                                            ADR{parquet.adr_period}
                                          </span>
                                        )}
                                      </div>
                                      <span className="text-[10px] text-muted-foreground/60">
                                        {parquet.start_date} → {parquet.end_date} | {parquet.rows.toLocaleString()} rows
                                      </span>
                                    </div>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <span className="text-[10px] text-muted-foreground/50">
                                      {parquet.size_mb.toFixed(1)}MB
                                    </span>
                                    <button
                                      onClick={(e) => {
                                        e.preventDefault()
                                        e.stopPropagation()
                                        handleDeleteParquet(parquet.name)
                                      }}
                                      className="p-1 rounded hover:bg-red-500/20 text-red-500/70 hover:text-red-500 transition-all"
                                      title="Delete"
                                    >
                                      <Trash2 className="h-3 w-3" />
                                    </button>
                                  </div>
                                </label>
                              )
                            })
                          ) : (
                            <div className="text-xs text-muted-foreground/60 p-2 text-center">
                              No raw parquets created yet
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Stage 2: Bridged Parquets */}
                    <div className="rounded-md border border-border/30 overflow-hidden">
                      <button
                        onClick={() => setStage2DataCollapsed(!stage2DataCollapsed)}
                        className="flex items-center justify-between w-full px-3 py-2 bg-cyan-500/5 hover:bg-cyan-500/10 transition-colors"
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground">
                            {stage2DataCollapsed ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                          </span>
                          <Link className="h-3.5 w-3.5 text-cyan-500" />
                          <span className="text-xs font-medium text-foreground/80">Stage 2: Bridged Parquets</span>
                          <span className="text-[10px] text-muted-foreground">
                            ({bridgedParquetsResponse?.parquets?.length || 0})
                          </span>
                        </div>
                        {selectedBridgedParquets.size > 0 && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-400">
                            {selectedBridgedParquets.size} selected
                          </span>
                        )}
                      </button>

                      {!stage2DataCollapsed && (
                        <div className="p-2 space-y-1 max-h-[200px] overflow-auto">
                          {bridgedParquetsLoading ? (
                            <div className="flex items-center gap-2 text-xs text-muted-foreground p-2">
                              <Loader2 className="h-3 w-3 animate-spin" />
                              Loading...
                            </div>
                          ) : bridgedParquetsResponse?.parquets && bridgedParquetsResponse.parquets.length > 0 ? (
                            bridgedParquetsResponse.parquets.map((parquet) => {
                              const isSelected = selectedBridgedParquets.has(parquet.name)
                              return (
                                <label
                                  key={parquet.name}
                                  className={`flex items-center justify-between px-2.5 py-2 rounded cursor-pointer transition-colors ${
                                    isSelected
                                      ? 'bg-cyan-500/10 border border-cyan-500/30'
                                      : 'bg-background/30 border border-border/20 hover:border-cyan-500/20'
                                  }`}
                                >
                                  <div className="flex items-center gap-2">
                                    <input
                                      type="checkbox"
                                      checked={isSelected}
                                      onChange={() => toggleBridgedParquet(parquet.name)}
                                      className="rounded border-border text-cyan-500 focus:ring-cyan-500/50 h-3 w-3"
                                    />
                                    <div className="flex flex-col">
                                      <div className="flex items-center gap-2">
                                        <span className="text-xs font-medium text-cyan-500/90">
                                          {parquet.pair || parquet.name.replace('_bridged.parquet', '')}
                                        </span>
                                        {parquet.timeframe && (
                                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/20 text-primary">
                                            {parquet.timeframe}
                                          </span>
                                        )}
                                        {parquet.adr_period && (
                                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400">
                                            ADR{parquet.adr_period}
                                          </span>
                                        )}
                                      </div>
                                      <span className="text-[10px] text-muted-foreground/60">
                                        {parquet.start_date && parquet.end_date
                                          ? `${parquet.start_date} → ${parquet.end_date} | `
                                          : ''
                                        }{parquet.rows.toLocaleString()} rows
                                      </span>
                                    </div>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <span className="text-[10px] text-muted-foreground/50">
                                      {parquet.size_mb.toFixed(1)}MB
                                    </span>
                                    <button
                                      onClick={(e) => {
                                        e.preventDefault()
                                        e.stopPropagation()
                                        handleDeleteBridged(parquet.name)
                                      }}
                                      className="p-1 rounded hover:bg-red-500/20 text-red-500/70 hover:text-red-500 transition-all"
                                      title="Delete"
                                    >
                                      <Trash2 className="h-3 w-3" />
                                    </button>
                                  </div>
                                </label>
                              )
                            })
                          ) : (
                            <div className="text-xs text-muted-foreground/60 p-2 text-center">
                              No bridged parquets yet
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Delete status messages */}
                  {deleteParquetsBatchMutation.isSuccess && (
                    <div className="text-xs text-emerald-500/90 bg-emerald-500/10 px-2 py-1.5 rounded border border-emerald-500/20">
                      ✓ Deleted {deleteParquetsBatchMutation.data.deleted.length} file(s)
                    </div>
                  )}
                  {deleteParquetsBatchMutation.isError && (
                    <div className="text-xs text-red-500/90 bg-red-500/10 px-2 py-1.5 rounded border border-red-500/20">
                      Error: {deleteParquetsBatchMutation.error.message}
                    </div>
                  )}
                </div>
              </TabsContent>

              {/* Training Tab */}
              <TabsContent value="training" className="mt-0 p-4 overflow-auto">
                <TransformerTab workingDirectory={workingDirectory} />
              </TabsContent>
            </div>
          </Tabs>
        </div>
      </Panel>

      {/* Resize Handle */}
      <PanelResizeHandle className="w-1 bg-border hover:bg-primary transition-colors cursor-col-resize" />

      {/* Main Content Area */}
      <Panel defaultSize={75} minSize={50}>
        <div className="h-full bg-background p-4">
          {activeTab === 'training' ? (
            <ParquetViewer workingDirectory={workingDirectory} />
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-card border border-border mb-4">
                  <Brain className="h-8 w-8 text-primary/60" />
                </div>
                <h2 className="text-lg font-semibold text-foreground/80 mb-1">LSTM Module</h2>
                <p className="text-sm text-muted-foreground">Symbolic Time Series Analysis</p>
                <p className="text-xs text-muted-foreground/60 mt-2">Use the Pipeline tab to create training data</p>
              </div>
            </div>
          )}
        </div>
      </Panel>
    </PanelGroup>
  )
}
