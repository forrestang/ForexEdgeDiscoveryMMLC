from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
from enum import Enum

from app.config import WAVE_COLORS


class Direction(Enum):
    UP = "up"
    DOWN = "down"

    def opposite(self) -> "Direction":
        return Direction.DOWN if self == Direction.UP else Direction.UP


@dataclass
class Wave:
    """Represents a single wave in the recursive waveform hierarchy."""

    id: int
    level: int  # L1, L2, L3, etc. (internal tracking, 1-indexed)
    direction: Direction
    start_time: datetime
    start_price: float
    end_time: datetime
    end_price: float
    parent_id: Optional[int] = None
    is_active: bool = True
    is_spline: bool = False  # True for intermediate developing leg lines

    @property
    def color(self) -> str:
        """Get wave color based on level (cycles through 5 colors)."""
        return WAVE_COLORS[(self.level - 1) % len(WAVE_COLORS)]

    def to_dict(self) -> dict:
        """Convert wave to dictionary for API response."""
        return {
            "id": self.id,
            "level": self.level,
            "direction": self.direction.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "start_price": self.start_price,
            "end_price": self.end_price,
            "color": self.color,
            "parent_id": self.parent_id,
            "is_spline": self.is_spline,
        }


@dataclass
class Candle:
    """Represents an OHLC candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def is_neutral(self) -> bool:
        return self.close == self.open

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2
