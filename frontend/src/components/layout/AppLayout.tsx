import { useEffect, useState, useRef } from 'react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import { Sidebar } from './Sidebar'
import { CandlestickChart } from '@/components/chart/CandlestickChart'
import { EdgeGraph } from '@/components/chart/EdgeGraph'
import { BarNavigator } from '@/components/chart/BarNavigator'
import { MatchedSessionModal } from '@/components/chart/MatchedSessionModal'
import { useChartInference, useEdgeMining } from '@/hooks/useEdgeFinder'
import { useEdgeFinderIndexStatus } from '@/hooks/useEdgeFinder'
import type { ChartSettings } from '@/App'
import type { EdgeProbabilities, EdgeGraphDataPoint, BarEdgeData } from '@/types'

interface AppLayoutProps {
  workingDirectory: string
  setWorkingDirectory: (dir: string) => void
  chartSettings: ChartSettings
  setChartSettings: React.Dispatch<React.SetStateAction<ChartSettings>>
  edgeProbabilities: EdgeProbabilities | null
  setEdgeProbabilities: (edge: EdgeProbabilities | null) => void
  selectedBarIndex: number | null
  setSelectedBarIndex: (index: number | null) => void
  totalBars: number | null
  setTotalBars: (count: number | null) => void
  kNeighbors: number
  setKNeighbors: (k: number) => void
}

export function AppLayout({
  workingDirectory,
  setWorkingDirectory,
  chartSettings,
  setChartSettings,
  edgeProbabilities,
  setEdgeProbabilities,
  selectedBarIndex,
  setSelectedBarIndex,
  totalBars,
  setTotalBars,
  kNeighbors,
  setKNeighbors,
}: AppLayoutProps) {
  const chartInference = useChartInference()
  const edgeMining = useEdgeMining()
  const { data: indexStatus } = useEdgeFinderIndexStatus(workingDirectory)
  const [isMatchModalOpen, setIsMatchModalOpen] = useState(false)

  // Edge mining state
  const [miningGraphData, setMiningGraphData] = useState<EdgeGraphDataPoint[] | null>(null)
  const [miningEdgeTable, setMiningEdgeTable] = useState<BarEdgeData[] | null>(null)
  const [miningError, setMiningError] = useState<string | null>(null)

  // Track last mined session to avoid re-mining on bar selection changes
  const lastMinedSessionRef = useRef<string | null>(null)

  // Reset bar selection and mining data when chart settings change (loading new session)
  useEffect(() => {
    setSelectedBarIndex(null)
    setTotalBars(null)
    // Clear mining data when session changes
    setMiningGraphData(null)
    setMiningEdgeTable(null)
    setMiningError(null)
  }, [chartSettings.pair, chartSettings.date, chartSettings.session, chartSettings.timeframe])

  // Trigger inference when chart settings or selected bar change and index is loaded
  useEffect(() => {
    // Only run inference if we have valid chart settings and index is loaded
    if (!chartSettings.pair || !chartSettings.date || !indexStatus?.is_loaded) {
      setEdgeProbabilities(null)
      return
    }

    // Map session type to backend format
    const sessionMap: Record<string, string> = {
      'full_day': 'full_day',
      'asia': 'asia',
      'london': 'london',
      'ny': 'ny',
    }

    chartInference.mutate(
      {
        pair: chartSettings.pair,
        date: chartSettings.date,
        session: sessionMap[chartSettings.session] || 'full_day',
        timeframe: chartSettings.timeframe,
        workingDirectory,
        bar_index: selectedBarIndex ?? undefined,
        k_neighbors: kNeighbors,
      },
      {
        onSuccess: (response) => {
          if (response.status === 'success' && response.edge) {
            setEdgeProbabilities(response.edge)
          } else {
            setEdgeProbabilities(null)
          }
        },
        onError: () => {
          setEdgeProbabilities(null)
        },
      }
    )
  }, [chartSettings.pair, chartSettings.date, chartSettings.session, chartSettings.timeframe, workingDirectory, indexStatus?.is_loaded, selectedBarIndex, kNeighbors])

  // Auto-mine edge scores when chart settings change (not on bar selection changes)
  useEffect(() => {
    // Only mine if we have valid chart settings and index is loaded
    if (!chartSettings.pair || !chartSettings.date || !indexStatus?.is_loaded) {
      return
    }

    // Create session key to track what we've mined
    const sessionKey = `${chartSettings.pair}_${chartSettings.date}_${chartSettings.session}_${chartSettings.timeframe}`

    // Skip if we've already mined this session
    if (lastMinedSessionRef.current === sessionKey) {
      return
    }

    // Mark as mining
    lastMinedSessionRef.current = sessionKey
    setMiningError(null)

    // Map session type to backend format
    const sessionMap: Record<string, string> = {
      'full_day': 'full_day',
      'asia': 'asia',
      'london': 'london',
      'ny': 'ny',
    }

    edgeMining.mutate(
      {
        pair: chartSettings.pair,
        date: chartSettings.date,
        session: sessionMap[chartSettings.session] || 'full_day',
        timeframe: chartSettings.timeframe,
        workingDirectory,
        k_neighbors: 50, // Fixed k for mining (different from per-bar inference)
      },
      {
        onSuccess: (response) => {
          if (response.status === 'success') {
            setMiningGraphData(response.graph_data)
            setMiningEdgeTable(response.edge_table)
            setMiningError(null)
          } else {
            setMiningGraphData(null)
            setMiningEdgeTable(null)
            setMiningError(response.message)
          }
        },
        onError: (error) => {
          setMiningGraphData(null)
          setMiningEdgeTable(null)
          setMiningError(error.message)
        },
      }
    )
  }, [chartSettings.pair, chartSettings.date, chartSettings.session, chartSettings.timeframe, workingDirectory, indexStatus?.is_loaded])

  // Determine error message for stats panel
  const getErrorMessage = () => {
    if (!indexStatus?.is_loaded) {
      return 'Index not loaded. Go to Edge tab to load a model.'
    }
    if (chartInference.error) {
      return chartInference.error.message
    }
    return null
  }

  return (
    <PanelGroup direction="horizontal" className="h-full">
      <Panel defaultSize={25} minSize={15} maxSize={40}>
        <Sidebar
          workingDirectory={workingDirectory}
          setWorkingDirectory={setWorkingDirectory}
          chartSettings={chartSettings}
          setChartSettings={setChartSettings}
          edgeProbabilities={edgeProbabilities}
          setEdgeProbabilities={setEdgeProbabilities}
          kNeighbors={kNeighbors}
          setKNeighbors={setKNeighbors}
          selectedBarIndex={selectedBarIndex}
          totalBars={totalBars}
          isInferenceLoading={chartInference.isPending}
          inferenceError={getErrorMessage()}
          onViewMatches={() => setIsMatchModalOpen(true)}
        />
      </Panel>

      <PanelResizeHandle className="w-1 bg-border hover:bg-primary transition-colors cursor-col-resize" />

      <Panel defaultSize={75} minSize={50}>
        <PanelGroup direction="vertical" className="h-full">
          <Panel defaultSize={70} minSize={30}>
            <div className="h-full bg-background p-4 flex flex-col">
              <CandlestickChart
                chartSettings={chartSettings}
                workingDirectory={workingDirectory}
                selectedBarIndex={selectedBarIndex}
                onBarSelect={setSelectedBarIndex}
                onTotalBarsChange={setTotalBars}
              />
              {totalBars !== null && (
                <BarNavigator
                  selectedBarIndex={selectedBarIndex}
                  totalBars={totalBars}
                  onBarSelect={setSelectedBarIndex}
                />
              )}
            </div>
          </Panel>

          <PanelResizeHandle className="h-1 bg-border hover:bg-primary transition-colors cursor-row-resize" />

          <Panel defaultSize={30} minSize={15} maxSize={50}>
            <EdgeGraph
              graphData={miningGraphData}
              edgeTable={miningEdgeTable}
              selectedBarIndex={selectedBarIndex}
              onBarSelect={setSelectedBarIndex}
              isLoading={edgeMining.isPending}
              error={miningError}
            />
          </Panel>
        </PanelGroup>
      </Panel>

      {/* Matched Session Modal */}
      <MatchedSessionModal
        isOpen={isMatchModalOpen}
        onClose={() => setIsMatchModalOpen(false)}
        matches={edgeProbabilities?.top_matches || []}
        workingDirectory={workingDirectory}
      />
    </PanelGroup>
  )
}
