import { useState } from 'react'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Settings, Database, Brain } from 'lucide-react'

export function LSTMPage() {
  const [activeTab, setActiveTab] = useState('pipeline')

  return (
    <div className="h-full flex">
      {/* Sidebar */}
      <div className="w-80 h-full flex flex-col bg-card border-r border-border">
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
              value="data"
              className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary"
            >
              <Database className="h-4 w-4 mr-1" />
              Data
            </TabsTrigger>
            <TabsTrigger
              value="training"
              className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary"
            >
              <Brain className="h-4 w-4 mr-1" />
              Training
            </TabsTrigger>
          </TabsList>

          <div className="flex-1 overflow-auto p-4">
            <TabsContent value="pipeline" className="mt-0">
              <div className="text-muted-foreground text-sm">
                Pipeline configuration will go here.
              </div>
            </TabsContent>

            <TabsContent value="data" className="mt-0">
              <div className="text-muted-foreground text-sm">
                Data management will go here.
              </div>
            </TabsContent>

            <TabsContent value="training" className="mt-0">
              <div className="text-muted-foreground text-sm">
                Training configuration will go here.
              </div>
            </TabsContent>
          </div>
        </Tabs>
      </div>

      {/* Main content area */}
      <div className="flex-1 h-full flex items-center justify-center bg-background">
        <div className="text-muted-foreground text-center">
          <h2 className="text-xl font-semibold mb-2">LSTM - Symbolic Time Series Analysis</h2>
          <p>Main content area</p>
        </div>
      </div>
    </div>
  )
}
