import { Button } from '@/components/ui/button'
import { ChevronLeft, ChevronRight } from 'lucide-react'

interface BarNavigatorProps {
  selectedBarIndex: number | null
  totalBars: number
  onBarSelect: (index: number) => void
}

export function BarNavigator({
  selectedBarIndex,
  totalBars,
  onBarSelect,
}: BarNavigatorProps) {
  // Default to last bar if no selection
  const currentIndex = selectedBarIndex ?? totalBars - 1

  const handlePrevious = () => {
    if (currentIndex > 0) {
      onBarSelect(currentIndex - 1)
    }
  }

  const handleNext = () => {
    if (currentIndex < totalBars - 1) {
      onBarSelect(currentIndex + 1)
    }
  }

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onBarSelect(parseInt(e.target.value, 10))
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      handlePrevious()
    } else if (e.key === 'ArrowRight') {
      e.preventDefault()
      handleNext()
    }
  }

  return (
    <div
      className="flex items-center gap-3 px-4 py-2 bg-background border-t border-border"
      onKeyDown={handleKeyDown}
      tabIndex={0}
    >
      {/* Previous button */}
      <Button
        variant="outline"
        size="icon"
        className="h-7 w-7"
        onClick={handlePrevious}
        disabled={currentIndex <= 0}
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>

      {/* Slider */}
      <div className="flex-1 flex items-center gap-3">
        <input
          type="range"
          min={0}
          max={totalBars - 1}
          value={currentIndex}
          onChange={handleSliderChange}
          className="flex-1 h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
        />
      </div>

      {/* Next button */}
      <Button
        variant="outline"
        size="icon"
        className="h-7 w-7"
        onClick={handleNext}
        disabled={currentIndex >= totalBars - 1}
      >
        <ChevronRight className="h-4 w-4" />
      </Button>

      {/* Bar position label */}
      <div className="text-xs text-muted-foreground min-w-[80px] text-right">
        Bar <span className="text-foreground font-medium">{currentIndex + 1}</span> / {totalBars}
      </div>
    </div>
  )
}
