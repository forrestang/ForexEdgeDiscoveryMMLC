import { useState } from 'react'
import {
  useFiles,
  useDeleteModel,
  useAutoSetup,
  useDeleteParquet,
  useDeleteSessions,
  useDeleteIndex,
  useDeleteAllParquets,
  useDeleteAllSessions,
  useDeleteAllModels,
  useDeleteAllIndices,
  useDeleteAllFiles,
} from '@/hooks/useEdgeFinder'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  Loader2,
  Database,
  FileArchive,
  Brain,
  Search,
  Trash2,
  Play,
  RefreshCw,
} from 'lucide-react'

interface DataTabProps {
  workingDirectory: string
}

type TabType = 'parquets' | 'sessions' | 'models' | 'indices'

export function DataTab({ workingDirectory }: DataTabProps) {
  const [activeTab, setActiveTab] = useState<TabType>('parquets')
  const [sessionDeletePair, setSessionDeletePair] = useState<string>('')
  const [sessionDeleteType, setSessionDeleteType] = useState<string>('')
  const [sessionDeleteTimeframe, setSessionDeleteTimeframe] = useState<string>('')

  const { data, isLoading, error, refetch } = useFiles(workingDirectory)
  const deleteModelMutation = useDeleteModel()
  const autoSetupMutation = useAutoSetup()
  const deleteParquetMutation = useDeleteParquet()
  const deleteSessionsMutation = useDeleteSessions()
  const deleteIndexMutation = useDeleteIndex()
  const deleteAllParquetsMutation = useDeleteAllParquets()
  const deleteAllSessionsMutation = useDeleteAllSessions()
  const deleteAllModelsMutation = useDeleteAllModels()
  const deleteAllIndicesMutation = useDeleteAllIndices()
  const deleteAllFilesMutation = useDeleteAllFiles()

  const handleDeleteModel = (modelName: string) => {
    if (confirm(`Delete model "${modelName}"? This cannot be undone.`)) {
      deleteModelMutation.mutate({
        modelName,
        workingDirectory,
      })
    }
  }

  const handleLoadModel = (modelName: string) => {
    autoSetupMutation.mutate({
      working_directory: workingDirectory,
      model_name: modelName,
    })
  }

  const handleDeleteParquet = (pair: string, timeframe: string) => {
    if (confirm(`Delete parquet file "${pair}_${timeframe}.parquet"? This cannot be undone.`)) {
      deleteParquetMutation.mutate({
        pair,
        timeframe,
        workingDirectory,
      })
    }
  }

  const handleDeleteSessions = () => {
    if (!sessionDeletePair || !sessionDeleteType || !sessionDeleteTimeframe) {
      alert('Please select pair, session type, and timeframe to delete sessions.')
      return
    }
    const count = estimateSessionCount()
    if (confirm(`Delete approximately ${count} session files for ${sessionDeletePair}/${sessionDeleteType}/${sessionDeleteTimeframe}? This cannot be undone.`)) {
      deleteSessionsMutation.mutate({
        pair: sessionDeletePair,
        session_type: sessionDeleteType,
        timeframe: sessionDeleteTimeframe,
        workingDirectory,
      }, {
        onSuccess: () => {
          setSessionDeletePair('')
          setSessionDeleteType('')
          setSessionDeleteTimeframe('')
        }
      })
    }
  }

  const handleDeleteIndex = (indexName: string) => {
    if (confirm(`Delete index "${indexName}"? This cannot be undone.`)) {
      deleteIndexMutation.mutate({
        indexName,
        workingDirectory,
      })
    }
  }

  const estimateSessionCount = () => {
    if (!data || !sessionDeletePair || !sessionDeleteType || !sessionDeleteTimeframe) return 0
    // Rough estimate based on available data
    const pairCount = data.sessions_by_pair?.[sessionDeletePair] || 0
    const typeCount = data.sessions_by_session_type?.[sessionDeleteType] || 0
    const tfCount = data.sessions_by_timeframe?.[sessionDeleteTimeframe] || 0
    const total = data.sessions_total || 1
    // Estimate intersection
    return Math.round((pairCount * typeCount * tfCount) / (total * total)) || 0
  }

  // Bulk delete handlers
  const handleDeleteAllParquets = () => {
    const count = data?.parquets?.length || 0
    if (count === 0) return
    if (confirm(`Delete ALL ${count} parquet files? This cannot be undone.`)) {
      deleteAllParquetsMutation.mutate(workingDirectory)
    }
  }

  const handleDeleteAllSessions = () => {
    const count = data?.sessions_total || 0
    if (count === 0) return
    if (confirm(`Delete ALL ${count} session files? This cannot be undone.`)) {
      deleteAllSessionsMutation.mutate(workingDirectory)
    }
  }

  const handleDeleteAllModels = () => {
    const count = data?.models?.length || 0
    if (count === 0) return
    if (confirm(`Delete ALL ${count} models? This cannot be undone.`)) {
      deleteAllModelsMutation.mutate(workingDirectory)
    }
  }

  const handleDeleteAllIndices = () => {
    const count = data?.indices?.length || 0
    if (count === 0) return
    if (confirm(`Delete ALL ${count} indices? This cannot be undone.`)) {
      deleteAllIndicesMutation.mutate(workingDirectory)
    }
  }

  const handleDeleteAllFiles = () => {
    const totalFiles = (data?.parquets?.length || 0) + (data?.sessions_total || 0) + (data?.models?.length || 0) + (data?.indices?.length || 0)
    if (totalFiles === 0) return
    if (confirm(`DELETE EVERYTHING?\n\nThis will delete:\n- ${data?.parquets?.length || 0} parquet files\n- ${data?.sessions_total || 0} session files\n- ${data?.models?.length || 0} models\n- ${data?.indices?.length || 0} indices\n\nCSV source files will NOT be deleted.\n\nThis cannot be undone!`)) {
      deleteAllFilesMutation.mutate(workingDirectory)
    }
  }

  const isAnyBulkDeleting = deleteAllParquetsMutation.isPending || deleteAllSessionsMutation.isPending || deleteAllModelsMutation.isPending || deleteAllIndicesMutation.isPending || deleteAllFilesMutation.isPending

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-8 text-red-400 text-sm">
        Error loading files: {error.message}
      </div>
    )
  }

  const tabs: { id: TabType; label: string; icon: typeof Database; count: number }[] = [
    { id: 'parquets', label: 'Parquets', icon: FileArchive, count: data?.parquets?.length || 0 },
    { id: 'sessions', label: 'Sessions', icon: Database, count: data?.sessions_total || 0 },
    { id: 'models', label: 'Models', icon: Brain, count: data?.models?.length || 0 },
    { id: 'indices', label: 'Indices', icon: Search, count: data?.indices?.length || 0 },
  ]

  return (
    <div className="space-y-4">
      {/* Tab Navigation */}
      <div className="flex gap-1 p-1 bg-secondary/50 rounded-lg">
        {tabs.map((tab) => {
          const Icon = tab.icon
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 flex items-center justify-center gap-1 py-1.5 px-2 rounded text-xs transition-colors ${
                activeTab === tab.id
                  ? 'bg-background shadow-sm'
                  : 'hover:bg-background/50'
              }`}
            >
              <Icon className="h-3 w-3" />
              <span className="hidden sm:inline">{tab.label}</span>
              <span className="text-muted-foreground">({tab.count})</span>
            </button>
          )
        })}
      </div>

      {/* Action Buttons */}
      <div className="flex justify-between items-center">
        <Button
          size="sm"
          variant="destructive"
          onClick={handleDeleteAllFiles}
          disabled={isAnyBulkDeleting || ((data?.parquets?.length || 0) + (data?.sessions_total || 0) + (data?.models?.length || 0) + (data?.indices?.length || 0)) === 0}
          className="text-xs"
        >
          {deleteAllFilesMutation.isPending ? (
            <Loader2 className="h-3 w-3 animate-spin mr-1" />
          ) : (
            <Trash2 className="h-3 w-3 mr-1" />
          )}
          Delete All Files
        </Button>
        <Button size="sm" variant="ghost" onClick={() => refetch()}>
          <RefreshCw className="h-3 w-3 mr-1" />
          Refresh
        </Button>
      </div>

      {/* Tab Content */}
      {activeTab === 'parquets' && (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm">Cached Parquet Files</CardTitle>
              {(data?.parquets?.length || 0) > 0 && (
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 text-xs text-red-500 hover:text-red-600"
                  onClick={handleDeleteAllParquets}
                  disabled={deleteAllParquetsMutation.isPending}
                >
                  {deleteAllParquetsMutation.isPending ? (
                    <Loader2 className="h-3 w-3 animate-spin mr-1" />
                  ) : (
                    <Trash2 className="h-3 w-3 mr-1" />
                  )}
                  Delete All
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {!data?.parquets?.length ? (
              <div className="text-center py-4 text-muted-foreground text-sm">
                <FileArchive className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No parquet files found.</p>
                <p className="text-xs mt-1">Use the Pipeline tab to process data.</p>
              </div>
            ) : (
              <div className="space-y-1 max-h-64 overflow-y-auto">
                {data.parquets.map((file) => (
                  <div
                    key={`${file.pair}_${file.timeframe}`}
                    className="flex items-center justify-between p-2 rounded bg-secondary/30 text-xs"
                  >
                    <div>
                      <span className="font-medium">{file.pair}</span>
                      <span className="text-muted-foreground ml-2">{file.timeframe}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground">{file.size_mb.toFixed(1)} MB</span>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 w-6 p-0 text-red-500 hover:text-red-600"
                        onClick={() => handleDeleteParquet(file.pair, file.timeframe)}
                        disabled={deleteParquetMutation.isPending}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {activeTab === 'sessions' && (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm">Session Datasets</CardTitle>
              {(data?.sessions_total || 0) > 0 && (
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 text-xs text-red-500 hover:text-red-600"
                  onClick={handleDeleteAllSessions}
                  disabled={deleteAllSessionsMutation.isPending}
                >
                  {deleteAllSessionsMutation.isPending ? (
                    <Loader2 className="h-3 w-3 animate-spin mr-1" />
                  ) : (
                    <Trash2 className="h-3 w-3 mr-1" />
                  )}
                  Delete All
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="text-2xl font-bold text-center">
              {data?.sessions_total?.toLocaleString() || 0}
              <span className="text-sm font-normal text-muted-foreground ml-2">sessions</span>
            </div>

            {data?.sessions_by_pair && Object.keys(data.sessions_by_pair).length > 0 && (
              <div className="space-y-1">
                <div className="text-xs text-muted-foreground">By Pair</div>
                <div className="flex flex-wrap gap-1">
                  {Object.entries(data.sessions_by_pair).map(([pair, count]) => (
                    <span key={pair} className="px-2 py-1 bg-secondary rounded text-xs">
                      {pair}: {count}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {data?.sessions_by_session_type && Object.keys(data.sessions_by_session_type).length > 0 && (
              <div className="space-y-1">
                <div className="text-xs text-muted-foreground">By Session Type</div>
                <div className="flex flex-wrap gap-1">
                  {Object.entries(data.sessions_by_session_type).map(([type, count]) => (
                    <span key={type} className="px-2 py-1 bg-secondary rounded text-xs">
                      {type}: {count}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {data?.sessions_by_timeframe && Object.keys(data.sessions_by_timeframe).length > 0 && (
              <div className="space-y-1">
                <div className="text-xs text-muted-foreground">By Timeframe</div>
                <div className="flex flex-wrap gap-1">
                  {Object.entries(data.sessions_by_timeframe).map(([tf, count]) => (
                    <span key={tf} className="px-2 py-1 bg-secondary rounded text-xs">
                      {tf}: {count}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Bulk Delete Section */}
            {data?.sessions_total > 0 && (
              <div className="border-t pt-3 mt-3 space-y-2">
                <div className="text-xs text-muted-foreground flex items-center gap-1">
                  <Trash2 className="h-3 w-3" />
                  Bulk Delete Sessions
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <select
                    value={sessionDeletePair}
                    onChange={(e) => setSessionDeletePair(e.target.value)}
                    className="text-xs p-1.5 rounded bg-secondary border-none"
                  >
                    <option value="">Pair</option>
                    {Object.keys(data.sessions_by_pair || {}).map((pair) => (
                      <option key={pair} value={pair}>{pair}</option>
                    ))}
                  </select>
                  <select
                    value={sessionDeleteType}
                    onChange={(e) => setSessionDeleteType(e.target.value)}
                    className="text-xs p-1.5 rounded bg-secondary border-none"
                  >
                    <option value="">Session</option>
                    {Object.keys(data.sessions_by_session_type || {}).map((type) => (
                      <option key={type} value={type}>{type}</option>
                    ))}
                  </select>
                  <select
                    value={sessionDeleteTimeframe}
                    onChange={(e) => setSessionDeleteTimeframe(e.target.value)}
                    className="text-xs p-1.5 rounded bg-secondary border-none"
                  >
                    <option value="">Timeframe</option>
                    {Object.keys(data.sessions_by_timeframe || {}).map((tf) => (
                      <option key={tf} value={tf}>{tf}</option>
                    ))}
                  </select>
                </div>
                <Button
                  size="sm"
                  variant="destructive"
                  className="w-full h-7 text-xs"
                  onClick={handleDeleteSessions}
                  disabled={!sessionDeletePair || !sessionDeleteType || !sessionDeleteTimeframe || deleteSessionsMutation.isPending}
                >
                  {deleteSessionsMutation.isPending ? (
                    <Loader2 className="h-3 w-3 animate-spin mr-1" />
                  ) : (
                    <Trash2 className="h-3 w-3 mr-1" />
                  )}
                  Delete Selected Sessions
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {activeTab === 'models' && (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm">Trained Models</CardTitle>
              {(data?.models?.length || 0) > 0 && (
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 text-xs text-red-500 hover:text-red-600"
                  onClick={handleDeleteAllModels}
                  disabled={deleteAllModelsMutation.isPending}
                >
                  {deleteAllModelsMutation.isPending ? (
                    <Loader2 className="h-3 w-3 animate-spin mr-1" />
                  ) : (
                    <Trash2 className="h-3 w-3 mr-1" />
                  )}
                  Delete All
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {!data?.models?.length ? (
              <div className="text-center py-4 text-muted-foreground text-sm">
                <Brain className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No trained models found.</p>
                <p className="text-xs mt-1">Use the Edge Finder tab to train a model.</p>
              </div>
            ) : (
              <div className="space-y-2">
                {data.models.map((model) => (
                  <div
                    key={model.model_name}
                    className={`p-3 rounded border ${
                      model.is_active
                        ? 'bg-primary/10 border-primary/30'
                        : 'bg-secondary/30 border-transparent'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium text-sm flex items-center gap-2">
                          {model.model_name}
                          {model.is_active && (
                            <span className="text-xs text-primary">(Active)</span>
                          )}
                        </div>
                        <div className="text-xs text-muted-foreground mt-1">
                          {model.trained_epochs} epochs | Latent: {model.latent_dim} | Loss: {model.best_loss.toFixed(4)}
                        </div>
                      </div>
                      <div className="flex gap-1">
                        {!model.is_active && (
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-7"
                            onClick={() => handleLoadModel(model.model_name)}
                            disabled={autoSetupMutation.isPending}
                          >
                            <Play className="h-3 w-3 mr-1" />
                            Load
                          </Button>
                        )}
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-7 text-red-500 hover:text-red-600"
                          onClick={() => handleDeleteModel(model.model_name)}
                          disabled={model.is_active || deleteModelMutation.isPending}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {activeTab === 'indices' && (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm">Vector Indices</CardTitle>
              {(data?.indices?.length || 0) > 0 && (
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 text-xs text-red-500 hover:text-red-600"
                  onClick={handleDeleteAllIndices}
                  disabled={deleteAllIndicesMutation.isPending}
                >
                  {deleteAllIndicesMutation.isPending ? (
                    <Loader2 className="h-3 w-3 animate-spin mr-1" />
                  ) : (
                    <Trash2 className="h-3 w-3 mr-1" />
                  )}
                  Delete All
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {!data?.indices?.length ? (
              <div className="text-center py-4 text-muted-foreground text-sm">
                <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No indices found.</p>
                <p className="text-xs mt-1">Build an index in the Edge Finder tab.</p>
              </div>
            ) : (
              <div className="space-y-2">
                {data.indices.map((index) => (
                  <div
                    key={index.index_name}
                    className="p-3 rounded bg-secondary/30 flex items-center justify-between"
                  >
                    <div>
                      <div className="font-medium text-sm">{index.index_name}</div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {index.num_vectors.toLocaleString()} vectors
                        {index.model_name && ` | Model: ${index.model_name}`}
                      </div>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 text-red-500 hover:text-red-600"
                      onClick={() => handleDeleteIndex(index.index_name)}
                      disabled={deleteIndexMutation.isPending}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
