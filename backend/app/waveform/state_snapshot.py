"""
State Snapshot dataclasses for capturing waveform state at each bar.

These structures enable bar-by-bar state capture for the Edge Finder system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class WaveSnapshot:
    """
    Snapshot of a single wave's state at a point in time.

    This is a lightweight representation of a Wave, capturing only
    the features needed for matrix serialization.
    """
    level: int              # 1-5 (L1 through L5)
    direction: int          # +1 (UP) or -1 (DOWN)
    amplitude: float        # end_price - start_price (signed)
    duration_bars: int      # Number of bars since wave started
    start_bar_index: int    # Bar index when this wave started

    @property
    def is_up(self) -> bool:
        return self.direction == 1

    @property
    def is_down(self) -> bool:
        return self.direction == -1


@dataclass
class StackSnapshot:
    """
    Complete wave stack state captured at a single bar.

    Represents the instantaneous state of all active waves (L1-L5)
    at the moment after processing a specific bar.
    """
    bar_index: int                      # 0-indexed position in session
    timestamp: datetime                 # Bar's timestamp
    close_price: float                  # Bar's close price
    waves: list[WaveSnapshot] = field(default_factory=list)  # Active waves (L1 first, deepest last)

    # Counts for each level (used for leg_count feature)
    l1_count: int = 0  # Total L1 waves seen so far in session
    l2_count: int = 0  # Total L2 waves seen so far
    l3_count: int = 0
    l4_count: int = 0
    l5_count: int = 0

    def get_wave_at_level(self, level: int) -> Optional[WaveSnapshot]:
        """Get the wave snapshot at a specific level, or None if not active."""
        for wave in self.waves:
            if wave.level == level:
                return wave
        return None

    def get_leg_count(self, level: int) -> int:
        """Get the cumulative count of waves at a given level."""
        counts = {1: self.l1_count, 2: self.l2_count, 3: self.l3_count,
                  4: self.l4_count, 5: self.l5_count}
        return counts.get(level, 0)

    @property
    def max_active_level(self) -> int:
        """Return the deepest active level (highest level number)."""
        if not self.waves:
            return 0
        return max(w.level for w in self.waves)

    @property
    def num_active_levels(self) -> int:
        """Return count of currently active wave levels."""
        return len(self.waves)
