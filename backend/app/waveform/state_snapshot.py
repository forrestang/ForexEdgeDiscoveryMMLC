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


@dataclass
class MMLCSwingSnapshot:
    """
    Represents a single swing point (vertex) in the waveform.

    Raw values only - normalization happens downstream.
    Used for mmlcOut array to train Autoencoder.
    """
    bar: int                # Bar index where swing occurred (-1 for preswing)
    price: float            # Price at swing point
    direction: int          # +1 (HIGH/UP), -1 (LOW/DOWN), 0 (OPEN anchor)
    level: int              # Wave level (0=OPEN, 1=L1, 2=L2, etc.)


@dataclass
class MMLCLegSnapshot:
    """
    Represents a single leg (completed or developing) in the waveform.

    Raw values only - normalization happens downstream.
    Used for mmlcOut array to train Autoencoder.
    """
    start_bar: int          # Bar index where leg started
    start_price: float      # Price at leg start
    end_bar: int            # Bar index where leg ended (or current bar for developing)
    end_price: float        # Price at leg end (or current close for developing)
    level: int              # Wave level (1=L1, 2=L2, etc.)
    direction: int          # +1 (UP) or -1 (DOWN)
    is_developing: bool     # True if this is the active developing leg


@dataclass
class MMLCBarSnapshot:
    """
    Complete MMLC state captured at a single bar T.

    Contains session context (anchors) and accumulated swing/leg history.
    Used for mmlcOut array to train Autoencoder.
    """
    bar_index: int                          # Current bar (0-indexed)

    # Session Context (Anchors)
    session_open_price: float               # First bar's open (static)
    total_session_bars: int                 # Total bars in session (fixed)

    # Current bar info
    current_close: float                    # Close price at this bar

    # Waveform as Swings (vertex points) - PRIMARY OUTPUT
    # Index 0: Preswing (bar=-1, OPEN price)
    # Index 1 to N-1: Real swings from price action
    # Index N: Close swing (current bar, CLOSE price)
    swings: list[MMLCSwingSnapshot] = field(default_factory=list)

    # Waveform History (accumulated legs through bar T) - DEPRECATED
    legs: list[MMLCLegSnapshot] = field(default_factory=list)


@dataclass
class LSTMOutcome:
    """
    Forward-looking outcome data for supervised training.

    These values can only be calculated AFTER the session is complete,
    as they require knowledge of future price action.
    """
    next_bar_delta: float      # Close[T+1] - Close[T] (0 for last bar)
    session_close_delta: float # Close[SessionEnd] - Close[T]
    session_max_up: float      # Max(High[T...End]) - Close[T] -> MFE
    session_max_down: float    # Min(Low[T...End]) - Close[T] -> MAE (negative)


@dataclass
class LSTMBarPayload:
    """
    LSTM-focused output for each bar processed.

    Designed for sequential time-series model training with:
    - Monotonically increasing sequence_id
    - ISO timestamp for temporal ordering
    - Price vector features
    - Categorical event classification

    Event types:
    - EXTENSION: L1 makes new extreme in same direction as previous
    - RETRACEMENT: No L1 extreme (pullback bar, L2+ active)
    - REVERSAL: L1 makes new extreme in opposite direction, or outside bar
    """
    sequence_id: int           # Monotonically increasing (1, 2, 3, ...)
    timestamp: str             # ISO format: "2025-01-07T14:30:00"
    total_session_bars: int    # Total bars in session (fixed)

    # Vector features (raw values, normalization downstream)
    price_raw: float           # Current close price
    price_delta: float         # Close - previous close (or 0 for first bar)
    time_delta: int            # Bars since last event (always 1 for bar-by-bar)

    # State classification
    level: int                 # Deepest active level (1=L1, 2=L2, etc.)
    direction: str             # "UP" or "DOWN" - direction of the active level
    event: str                 # "EXTENSION", "RETRACEMENT", or "REVERSAL"

    # Forward-looking outcome (populated after session processing)
    outcome: Optional[LSTMOutcome] = None
