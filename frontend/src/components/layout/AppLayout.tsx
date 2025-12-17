import { useEffect, useState } from 'react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import { Sidebar } from './Sidebar'
import { CandlestickChart } from '@/components/chart/CandlestickChart'
import { EdgeStatsPanel } from '@/components/chart/EdgeStatsPanel'
import { BarNavigator } from '@/components/chart/BarNavigator'
import { MatchedSessionModal } from '@/components/chart/MatchedSessionModal'
import { useChartInference } from '@/hooks/useEdgeFinder'
import { useEdgeFinderIndexStatus } from '@/hooks/useEdgeFinder'
import type { ChartSettings } from '@/App'
import type { EdgeProbabilities } from '@/types'

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
  const { data: indexStatus } = useEdgeFinderIndexStatus(workingDirectory)
  const [isMatchModalOpen, setIsMatchModalOpen] = useState(false)

  // Reset bar selection when chart settings change (loading new session)
  useEffect(() => {
    setSelectedBarIndex(null)
    setTotalBars(null)
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
        />
      </Panel>

      <PanelResizeHandle className="w-1 bg-border hover:bg-primary transition-colors cursor-col-resize" />

      <Panel defaultSize={75} minSize={50}>
        <PanelGroup direction="vertical" className="h-full">
          <Panel defaultSize={75} minSize={40}>
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

          <Panel defaultSize={25} minSize={10} maxSize={50}>
            <EdgeStatsPanel
              edge={edgeProbabilities}
              isLoading={chartInference.isPending}
              error={getErrorMessage()}
              selectedBarIndex={selectedBarIndex}
              totalBars={totalBars}
              kNeighbors={kNeighbors}
              onViewMatches={() => setIsMatchModalOpen(true)}
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
