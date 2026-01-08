import { useState } from 'react'
import { AppLayout } from '@/components/layout/AppLayout'
import { TopNavigation } from '@/components/layout/TopNavigation'
import { MMLCDevPage } from '@/pages/MMLCDevPage'
import { LSTMPage } from '@/pages/LSTMPage'
import { STPMPage } from '@/pages/STPMPage'
import { DebugPanelPage } from '@/pages/DebugPanelPage'
import { StatsPanelPage } from '@/pages/StatsPanelPage'
import { useKNeighbors, useMainTab } from '@/hooks/usePersistedSettings'
import type { SessionType, TimeframeType, EdgeProbabilities } from '@/types'
import { DEFAULT_WORKING_DIRECTORY } from '@/lib/constants'

export interface ChartSettings {
  pair: string | null
  date: string | null
  session: SessionType
  timeframe: TimeframeType
}

function App() {
  // Check URL params for popup pages
  const urlParams = new URLSearchParams(window.location.search)
  const pageParam = urlParams.get('page')
  const isDebugPanel = pageParam === 'debug-panel'
  const isStatsPanel = pageParam === 'stats-panel'

  const [activeTab, setActiveTab] = useMainTab()
  const [workingDirectory, setWorkingDirectory] = useState(DEFAULT_WORKING_DIRECTORY)
  const [chartSettings, setChartSettings] = useState<ChartSettings>({
    pair: null,
    date: null,
    session: 'full_day',
    timeframe: 'M5',
  })
  const [edgeProbabilities, setEdgeProbabilities] = useState<EdgeProbabilities | null>(null)
  const [selectedBarIndex, setSelectedBarIndex] = useState<number | null>(null)
  const [totalBars, setTotalBars] = useState<number | null>(null)
  const [kNeighbors, setKNeighbors] = useKNeighbors()

  // Popup windows
  if (isDebugPanel) {
    return <DebugPanelPage />
  }
  if (isStatsPanel) {
    return <StatsPanelPage />
  }

  const renderContent = () => {
    switch (activeTab) {
      case 'ae-knn':
        return (
          <AppLayout
            workingDirectory={workingDirectory}
            setWorkingDirectory={setWorkingDirectory}
            chartSettings={chartSettings}
            setChartSettings={setChartSettings}
            edgeProbabilities={edgeProbabilities}
            setEdgeProbabilities={setEdgeProbabilities}
            selectedBarIndex={selectedBarIndex}
            setSelectedBarIndex={setSelectedBarIndex}
            totalBars={totalBars}
            setTotalBars={setTotalBars}
            kNeighbors={kNeighbors}
            setKNeighbors={setKNeighbors}
          />
        )
      case 'lstm':
        return <LSTMPage />
      case 'stpm':
        return <STPMPage />
      case 'sandbox':
        return <MMLCDevPage />
      default:
        return null
    }
  }

  return (
    <div className="h-screen flex flex-col">
      <TopNavigation activeTab={activeTab} onTabChange={setActiveTab} />
      <div className="flex-1 min-h-0">
        {renderContent()}
      </div>
    </div>
  )
}

export default App
