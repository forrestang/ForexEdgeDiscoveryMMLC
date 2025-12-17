import { useState } from 'react'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { PipelineTab } from '@/components/sidebar/PipelineTab'
import { DataTab } from '@/components/sidebar/DataTab'
import { ExploreTab } from '@/components/sidebar/ExploreTab'
import { EdgeFinderTab } from '@/components/sidebar/EdgeFinderTab'
import { Settings, Database, Search, Brain } from 'lucide-react'
import type { ChartSettings } from '@/App'
import type { EdgeProbabilities } from '@/types'

interface SidebarProps {
  workingDirectory: string
  setWorkingDirectory: (dir: string) => void
  chartSettings: ChartSettings
  setChartSettings: React.Dispatch<React.SetStateAction<ChartSettings>>
  edgeProbabilities: EdgeProbabilities | null
  setEdgeProbabilities: (edge: EdgeProbabilities | null) => void
  kNeighbors: number
  setKNeighbors: (k: number) => void
}

export function Sidebar({
  workingDirectory,
  setWorkingDirectory,
  chartSettings,
  setChartSettings,
  edgeProbabilities,
  setEdgeProbabilities,
  kNeighbors,
  setKNeighbors,
}: SidebarProps) {
  const [activeTab, setActiveTab] = useState('explore')

  return (
    <div className="h-full flex flex-col bg-card border-r border-border">
      <div className="p-4 border-b border-border">
        <h1 className="text-lg font-semibold">Waveform Analyzer</h1>
        <p className="text-xs text-muted-foreground">Recursive Session Analysis</p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
        <TabsList className="w-full justify-start rounded-none border-b border-border bg-transparent p-0">
          <TabsTrigger
            value="pipeline"
            className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary"
          >
            <Settings className="h-4 w-4 mr-1" />
            Pipeline
          </TabsTrigger>
          <TabsTrigger
            value="data"
            className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary"
          >
            <Database className="h-4 w-4 mr-1" />
            Data
          </TabsTrigger>
          <TabsTrigger
            value="explore"
            className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary"
          >
            <Search className="h-4 w-4 mr-1" />
            Explore
          </TabsTrigger>
          <TabsTrigger
            value="edge"
            className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary"
          >
            <Brain className="h-4 w-4 mr-1" />
            Edge
          </TabsTrigger>
        </TabsList>

        <div className="flex-1 overflow-auto p-4">
          <TabsContent value="pipeline" className="mt-0">
            <PipelineTab
              workingDirectory={workingDirectory}
              setWorkingDirectory={setWorkingDirectory}
            />
          </TabsContent>

          <TabsContent value="data" className="mt-0">
            <DataTab
              workingDirectory={workingDirectory}
              setChartSettings={setChartSettings}
            />
          </TabsContent>

          <TabsContent value="explore" className="mt-0">
            <ExploreTab
              workingDirectory={workingDirectory}
              chartSettings={chartSettings}
              setChartSettings={setChartSettings}
            />
          </TabsContent>

          <TabsContent value="edge" className="mt-0">
            <EdgeFinderTab
              workingDirectory={workingDirectory}
              edgeProbabilities={edgeProbabilities}
              onEdgeProbabilitiesChange={setEdgeProbabilities}
              kNeighbors={kNeighbors}
              onKNeighborsChange={setKNeighbors}
            />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  )
}
