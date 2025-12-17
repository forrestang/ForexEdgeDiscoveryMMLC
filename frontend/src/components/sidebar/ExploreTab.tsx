import { useInstruments, useInstrumentDates } from '@/hooks/useInstruments'
import { Button } from '@/components/ui/button'
import { Select } from '@/components/ui/select'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { SESSION_OPTIONS, TIMEFRAME_OPTIONS } from '@/lib/constants'
import { Loader2, Search } from 'lucide-react'
import type { ChartSettings } from '@/App'
import type { SessionType, TimeframeType } from '@/types'

interface ExploreTabProps {
  workingDirectory: string
  chartSettings: ChartSettings
  setChartSettings: React.Dispatch<React.SetStateAction<ChartSettings>>
}

export function ExploreTab({
  workingDirectory,
  chartSettings,
  setChartSettings,
}: ExploreTabProps) {
  const { data: instrumentsData, isLoading: loadingInstruments } = useInstruments(workingDirectory)
  const { data: datesData, isLoading: loadingDates } = useInstrumentDates(
    chartSettings.pair,
    workingDirectory
  )

  const instruments = instrumentsData?.instruments || []
  const dates = datesData?.dates || []

  const handleApply = () => {
    // The chart will automatically load based on chartSettings
    // This button is more for UX clarity
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Explore Chart</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Instrument Selection */}
          <div className="space-y-2">
            <label className="text-xs text-muted-foreground">Instrument</label>
            {loadingInstruments ? (
              <div className="flex items-center gap-2 h-9 px-3 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading...
              </div>
            ) : (
              <Select
                value={chartSettings.pair || ''}
                onChange={(e) =>
                  setChartSettings((prev) => ({
                    ...prev,
                    pair: e.target.value || null,
                    date: null, // Reset date when instrument changes
                  }))
                }
                options={[
                  { value: '', label: 'Select instrument...' },
                  ...instruments.map((i) => ({
                    value: i.pair,
                    label: i.pair,
                  })),
                ]}
              />
            )}
          </div>

          {/* Date Selection */}
          <div className="space-y-2">
            <label className="text-xs text-muted-foreground">Date</label>
            {loadingDates && chartSettings.pair ? (
              <div className="flex items-center gap-2 h-9 px-3 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading dates...
              </div>
            ) : (
              <Select
                value={chartSettings.date || ''}
                onChange={(e) =>
                  setChartSettings((prev) => ({
                    ...prev,
                    date: e.target.value || null,
                  }))
                }
                options={[
                  { value: '', label: 'Select date...' },
                  ...dates.map((d) => ({
                    value: d,
                    label: d,
                  })),
                ]}
                disabled={!chartSettings.pair}
              />
            )}
          </div>

          {/* Session Selection */}
          <div className="space-y-2">
            <label className="text-xs text-muted-foreground">Session</label>
            <Select
              value={chartSettings.session}
              onChange={(e) =>
                setChartSettings((prev) => ({
                  ...prev,
                  session: e.target.value as SessionType,
                }))
              }
              options={SESSION_OPTIONS.map((s) => ({
                value: s.value,
                label: s.label,
              }))}
            />
          </div>

          {/* Timeframe Selection */}
          <div className="space-y-2">
            <label className="text-xs text-muted-foreground">Timeframe</label>
            <Select
              value={chartSettings.timeframe}
              onChange={(e) =>
                setChartSettings((prev) => ({
                  ...prev,
                  timeframe: e.target.value as TimeframeType,
                }))
              }
              options={TIMEFRAME_OPTIONS.map((t) => ({
                value: t.value,
                label: t.label,
              }))}
            />
          </div>

          <Button
            onClick={handleApply}
            disabled={!chartSettings.pair || !chartSettings.date}
            className="w-full"
          >
            <Search className="mr-2 h-4 w-4" />
            Load Chart
          </Button>
        </CardContent>
      </Card>

      {/* Current Selection Summary */}
      {chartSettings.pair && chartSettings.date && (
        <Card>
          <CardContent className="p-3 text-xs">
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Pair:</span>
                <span className="font-medium">{chartSettings.pair}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Date:</span>
                <span className="font-medium">{chartSettings.date}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Session:</span>
                <span className="font-medium">
                  {SESSION_OPTIONS.find((s) => s.value === chartSettings.session)?.label.split(' ')[0]}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Timeframe:</span>
                <span className="font-medium">{chartSettings.timeframe}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
