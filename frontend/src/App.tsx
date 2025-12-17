import { useState } from 'react'
import { AppLayout } from '@/components/layout/AppLayout'
import { useKNeighbors } from '@/hooks/usePersistedSettings'
import type { SessionType, TimeframeType, EdgeProbabilities } from '@/types'
import { DEFAULT_WORKING_DIRECTORY } from '@/lib/constants'

export interface ChartSettings {
  pair: string | null
  date: string | null
  session: SessionType
  timeframe: TimeframeType
}

function App() {
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
}

export default App
