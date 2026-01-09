"""
MMLC Core - Frozen MMLC Logic for LSTM Bridge

This is a FROZEN copy of the core MMLC algorithm extracted from dev_engine.py.
It is INDEPENDENT from the sandbox development environment and will not be
affected by changes to the sandbox.

Purpose:
- Process bars one by one
- Track L1/L2/L3+ wave levels
- Determine level, direction, and event (SPAWN/EXTENSION/REVERSAL) per bar

Version: 1.0 (frozen from dev_engine.py on 2025-01-08)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


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
    def midpoint(self) -> float:
        return (self.high + self.low) / 2


@dataclass
class BarState:
    """MMLC state output for a single bar."""
    bar_idx: int
    level: int              # 1, 2, 3... (deepest active level)
    direction: str          # "UP" or "DOWN"
    event: str              # "SPAWN", "EXTENSION", or "REVERSAL"


@dataclass
class BarOutcome:
    """Forward-looking outcome for a bar."""
    next_bar_delta: float       # Close[T+1] - Close[T]
    session_close_delta: float  # Close[SessionEnd] - Close[T]
    session_max_up: float       # Highest[T...SessionEnd] - Close[T]
    session_max_down: float     # Lowest[T...SessionEnd] - Close[T]


@dataclass
class WaveLevel:
    """
    State for a single wave level (L1, L2, L3, etc.).

    Each level tracks:
    - Current extremes (high/low)
    - Origin point (where this level's tracking started)
    - Completed waves (historical retracements)
    """
    level: int                  # 1, 2, 3, etc.
    direction: int = 0          # -1=DOWN, 0=NEUTRAL, +1=UP
    high: float = 0.0           # Current highest price
    low: float = 0.0            # Current lowest price
    high_bar: int = 0           # Bar index where high occurred
    low_bar: int = 0            # Bar index where low occurred
    origin_bar: int = 0         # Start bar of current wave
    origin_price: float = 0.0   # Start price of current wave
    completed_waves: list = field(default_factory=list)

    # For L1 only: swing point arrays
    swing_x: list = field(default_factory=list)  # Bar indices
    swing_y: list = field(default_factory=list)  # Prices

    # Per-level counters
    current_leg: int = 0    # Overall swing number for this level (never resets)
    current_count: int = 0  # Extremes in current direction (resets on direction change)
    prev_direction: int = 0 # Track previous direction for count reset logic

    def reset(self, bar_idx: int, origin_price: float, direction: int) -> None:
        """Reset this level to a new origin point."""
        self.high = origin_price
        self.low = origin_price
        self.high_bar = bar_idx
        self.low_bar = bar_idx
        self.origin_bar = bar_idx
        self.origin_price = origin_price
        self.direction = direction

    def has_retracement(self) -> bool:
        """Check if this level has any retracement (high != low)."""
        return self.low < self.high

    def record_new_extreme(self, new_direction: int) -> None:
        """
        Call this when this level makes a new extreme (high or low).
        Updates current_leg (always increments) and current_count (resets on direction change).
        """
        self.current_leg += 1
        if new_direction != self.prev_direction:
            self.current_count = 1
        else:
            self.current_count += 1
        self.prev_direction = new_direction


class MMLCCore:
    """
    Frozen MMLC core engine for LSTM bridge processing.

    This is a stripped-down version containing only the essential
    bar-by-bar processing logic. All visualization and debug code removed.
    """

    def __init__(self):
        self._candles: list[Candle] = []

        # Stitch swings: (bar, price, direction) tuples
        self._stitch_swings: list[tuple[int, float, int]] = []
        self._stitch_swings_level: list[int] = []

        # L1 counters
        self._L1_count: int = 0
        self._prev_swing_direction: int = 0
        self._L1_leg: int = 0

        # Active level history: tracks which level each bar belongs to
        self._active_level_history: list[int] = []

        # Wave levels - dynamic list that can grow to L3, L4, etc.
        self.levels: list[WaveLevel] = [
            WaveLevel(level=1),  # L1
            WaveLevel(level=2),  # L2
        ]

    # ================================================================
    # L1 CONVENIENCE PROPERTIES (access levels[0] directly)
    # ================================================================

    @property
    def L1_Direction(self) -> int:
        return self.levels[0].direction

    @L1_Direction.setter
    def L1_Direction(self, value: int):
        self.levels[0].direction = value

    @property
    def L1_High(self) -> float:
        return self.levels[0].high

    @L1_High.setter
    def L1_High(self, value: float):
        self.levels[0].high = value

    @property
    def L1_Low(self) -> float:
        return self.levels[0].low

    @L1_Low.setter
    def L1_Low(self, value: float):
        self.levels[0].low = value

    @property
    def L1_High_bar(self) -> int:
        return self.levels[0].high_bar

    @L1_High_bar.setter
    def L1_High_bar(self, value: int):
        self.levels[0].high_bar = value

    @property
    def L1_Low_bar(self) -> int:
        return self.levels[0].low_bar

    @L1_Low_bar.setter
    def L1_Low_bar(self, value: int):
        self.levels[0].low_bar = value

    @property
    def L1_swing_x(self) -> list:
        return self.levels[0].swing_x

    @property
    def L1_swing_y(self) -> list:
        return self.levels[0].swing_y

    # L2 convenience properties
    @property
    def L2_High(self) -> float:
        return self.levels[1].high

    @L2_High.setter
    def L2_High(self, value: float):
        self.levels[1].high = value

    @property
    def L2_Low(self) -> float:
        return self.levels[1].low

    @L2_Low.setter
    def L2_Low(self, value: float):
        self.levels[1].low = value

    @property
    def L2_High_bar(self) -> int:
        return self.levels[1].high_bar

    @L2_High_bar.setter
    def L2_High_bar(self, value: int):
        self.levels[1].high_bar = value

    @property
    def L2_Low_bar(self) -> int:
        return self.levels[1].low_bar

    @L2_Low_bar.setter
    def L2_Low_bar(self, value: int):
        self.levels[1].low_bar = value

    @property
    def L2_origin_bar(self) -> int:
        return self.levels[1].origin_bar

    @L2_origin_bar.setter
    def L2_origin_bar(self, value: int):
        self.levels[1].origin_bar = value

    @property
    def L2_origin_price(self) -> float:
        return self.levels[1].origin_price

    @L2_origin_price.setter
    def L2_origin_price(self, value: float):
        self.levels[1].origin_price = value

    # ================================================================
    # CORE PROCESSING METHODS
    # ================================================================

    def _determine_bar_direction(self, candle: Candle) -> int:
        """
        Determine bar direction using the tiebreaker chain.

        Returns:
            +1 for bullish (LOW first, then HIGH)
            -1 for bearish (HIGH first, then LOW)
        """
        if candle.close > candle.open:
            return 1  # bullish
        elif candle.close < candle.open:
            return -1  # bearish
        else:
            # Doji - use midpoint
            midpoint = (candle.high + candle.low) / 2
            if candle.close > midpoint:
                return 1
            elif candle.close < midpoint:
                return -1
            else:
                return self.L1_Direction if self.L1_Direction != 0 else 1

    def _get_level(self, level_num: int) -> Optional[WaveLevel]:
        """Get a level by number (1-indexed). Returns None if not exists."""
        idx = level_num - 1
        if 0 <= idx < len(self.levels):
            return self.levels[idx]
        return None

    def _ensure_level_exists(self, level_num: int) -> WaveLevel:
        """Ensure a level exists, creating it if necessary."""
        idx = level_num - 1
        while len(self.levels) <= idx:
            new_level_num = len(self.levels) + 1
            self.levels.append(WaveLevel(level=new_level_num))
        return self.levels[idx]

    def _push_stitch_swing(self, bar: int, price: float, direction: int) -> None:
        """Push a swing point to the stitch_swings array."""
        self._L1_leg += 1
        if direction != self._prev_swing_direction:
            self._L1_count = 1
        else:
            self._L1_count += 1
        self._prev_swing_direction = direction
        self.levels[0].record_new_extreme(direction)
        self._stitch_swings.append((bar, price, direction))
        self._stitch_swings_level.append(1)

    def _pop_stitch_swing(self) -> Optional[tuple]:
        """Pop the last swing from stitch_swings array."""
        if self._stitch_swings:
            popped = self._stitch_swings.pop()
            if self._stitch_swings_level:
                self._stitch_swings_level.pop()
            return popped
        return None

    def _get_last_stitch_bar(self) -> int:
        """Get the bar number of the most recent swing."""
        if self._stitch_swings:
            return self._stitch_swings[-1][0]
        return -1

    def _find_child_swing_at_bar(self, bar_idx: int) -> Optional[tuple]:
        """Find a child level (L2+) swing at the current bar."""
        for level in self.levels[1:]:
            if level.high_bar == bar_idx:
                return (level.level, level.high, +1)
            elif level.low_bar == bar_idx:
                return (level.level, level.low, -1)
        return None

    def _push_stitch_swing_child(self, bar: int, price: float, direction: int, level: int = 2) -> None:
        """Push a child-level swing point (doesn't increment L1 counters)."""
        self._stitch_swings.append((bar, price, direction))
        self._stitch_swings_level.append(level)

    def _complete_and_reset_level(
        self,
        level_num: int,
        bar_idx: int,
        new_origin_price: float,
        parent_direction: int
    ) -> None:
        """Complete current wave for level N and reset to new origin."""
        level = self._ensure_level_exists(level_num)

        if level.has_retracement():
            if parent_direction == 1:
                level.completed_waves.append((
                    level.origin_bar, level.origin_price,
                    level.low_bar, level.low
                ))
            else:
                level.completed_waves.append((
                    level.origin_bar, level.origin_price,
                    level.high_bar, level.high
                ))

        child_direction = -parent_direction
        level.reset(bar_idx, new_origin_price, child_direction)

        next_level = self._get_level(level_num + 1)
        if next_level is not None and next_level.has_retracement():
            self._complete_and_reset_level(
                level_num + 1,
                bar_idx,
                new_origin_price,
                child_direction
            )

    def _update_level_pullback(
        self,
        level_num: int,
        candle: Candle,
        bar_idx: int,
        parent_direction: int
    ) -> bool:
        """Update level N during parent's pullback. Returns True if new extreme."""
        level = self._ensure_level_exists(level_num)
        made_new_extreme = False

        if parent_direction == 1:
            # Parent UP -> this level retraces DOWN -> track lower lows
            if candle.high <= level.high and candle.low < level.low:
                level.low = candle.low
                level.low_bar = bar_idx
                made_new_extreme = True
                level.record_new_extreme(-1)
        else:
            # Parent DOWN -> this level retraces UP -> track higher highs
            if candle.low >= level.low and candle.high > level.high:
                level.high = candle.high
                level.high_bar = bar_idx
                made_new_extreme = True
                level.record_new_extreme(+1)

        if made_new_extreme:
            self._ensure_level_exists(level_num + 1)
            child_parent_dir = -parent_direction
            self._complete_and_reset_level(
                level_num + 1,
                bar_idx,
                candle.low if parent_direction == 1 else candle.high,
                child_parent_dir
            )
        else:
            child_level = self._get_level(level_num + 1)
            if child_level is not None:
                child_parent_dir = -parent_direction
                self._update_level_pullback(
                    level_num + 1,
                    candle,
                    bar_idx,
                    child_parent_dir
                )

        return made_new_extreme

    def _initialize_first_bar(self, candle: Candle, bar_idx: int) -> None:
        """Initialize L1 state on the first bar."""
        bar_direction = self._determine_bar_direction(candle)
        self.L1_Direction = bar_direction

        # Add preswing at index -1 (OPEN price anchor)
        self.L1_swing_x.append(-1)
        self.L1_swing_y.append(candle.open)

        # First swing based on direction
        if bar_direction == 1:
            self.L1_swing_x.append(bar_idx)
            self.L1_swing_y.append(candle.low)
        else:
            self.L1_swing_x.append(bar_idx)
            self.L1_swing_y.append(candle.high)

        self.L1_High = candle.high
        self.L1_Low = candle.low
        self.L1_High_bar = bar_idx
        self.L1_Low_bar = bar_idx

        # Add OPEN anchor
        self._stitch_swings.append((-1, candle.open, 0))
        self._stitch_swings_level.append(0)

        if bar_direction == 1:
            self._push_stitch_swing(bar_idx, candle.low, -1)
            self._push_stitch_swing(bar_idx, candle.high, +1)
        else:
            self._push_stitch_swing(bar_idx, candle.high, +1)
            self._push_stitch_swing(bar_idx, candle.low, -1)

        # Initialize L2
        if self.L1_Direction == 1:
            self.L2_High = candle.high
            self.L2_Low = candle.high
            self.L2_High_bar = bar_idx
            self.L2_Low_bar = bar_idx
            self.L2_origin_bar = bar_idx
            self.L2_origin_price = candle.high
            self.levels[1].direction = -1
        else:
            self.L2_High = candle.low
            self.L2_Low = candle.low
            self.L2_High_bar = bar_idx
            self.L2_Low_bar = bar_idx
            self.L2_origin_bar = bar_idx
            self.L2_origin_price = candle.low
            self.levels[1].direction = 1

    def _complete_and_reset_L2(self, bar_idx: int, new_l2_origin_price: float, new_parent_direction: int = None) -> None:
        """Complete current L2 wave and reset to new origin."""
        old_parent_direction = self.L1_Direction
        if new_parent_direction is None:
            new_parent_direction = self.L1_Direction

        level = self._ensure_level_exists(2)
        if level.has_retracement():
            if old_parent_direction == 1:
                level.completed_waves.append((
                    level.origin_bar, level.origin_price,
                    level.low_bar, level.low
                ))
            else:
                level.completed_waves.append((
                    level.origin_bar, level.origin_price,
                    level.high_bar, level.high
                ))

        child_direction = -new_parent_direction
        level.reset(bar_idx, new_l2_origin_price, child_direction)

        next_level = self._get_level(3)
        if next_level is not None and next_level.has_retracement():
            self._complete_and_reset_level(
                3,
                bar_idx,
                new_l2_origin_price,
                child_direction
            )

    def _update_L2_pullback(self, candle: Candle, bar_idx: int) -> None:
        """Update L2 during pullback."""
        self._update_level_pullback(
            level_num=2,
            candle=candle,
            bar_idx=bar_idx,
            parent_direction=self.L1_Direction
        )

    def _process_bar(self, candle: Candle, bar_idx: int) -> None:
        """Process subsequent bars (after bar 0)."""
        if self.L1_Direction == 1:
            # UP direction rules
            if candle.high > self.L1_High and candle.low >= self.L1_Low:
                # Case 1: New high, no new low - continuing UP
                self._complete_and_reset_L2(bar_idx, candle.high)
                self.L1_High = candle.high
                self.L1_High_bar = bar_idx
                if self._stitch_swings and self._stitch_swings[-1][2] == +1:
                    self._pop_stitch_swing()
                self._push_stitch_swing(bar_idx, candle.high, +1)

            elif candle.low < self.L1_Low and candle.high <= self.L1_High:
                # Case 2: New low, no new high - reversal
                self.L1_swing_x.append(self.L1_High_bar)
                self.L1_swing_y.append(self.L1_High)
                self._complete_and_reset_L2(bar_idx, candle.low, new_parent_direction=-1)
                self.L1_Low = candle.low
                self.L1_Low_bar = bar_idx
                if self._stitch_swings and self._stitch_swings[-1][2] == -1:
                    self._pop_stitch_swing()
                self._push_stitch_swing(bar_idx, candle.low, -1)
                self.L1_Direction = -1

            elif candle.high > self.L1_High and candle.low < self.L1_Low:
                # Case 3: Outside bar
                bar_direction = self._determine_bar_direction(candle)
                if bar_direction == 1:
                    # Bullish outside bar - continuation
                    self.L1_swing_x.append(self.L1_High_bar)
                    self.L1_swing_y.append(self.L1_High)
                    self.L1_swing_x.append(bar_idx)
                    self.L1_swing_y.append(candle.low)
                    self._complete_and_reset_L2(bar_idx, candle.high)
                    self.L1_Low = candle.low
                    self.L1_Low_bar = bar_idx
                    self.L1_High = candle.high
                    self.L1_High_bar = bar_idx
                    self._push_stitch_swing(bar_idx, candle.low, -1)
                    self._push_stitch_swing(bar_idx, candle.high, +1)
                else:
                    # Bearish outside bar - reversal
                    if self._stitch_swings and self._stitch_swings[-1][0] == self.L1_High_bar:
                        self._pop_stitch_swing()
                    if self.L1_swing_x and self.L1_swing_x[-1] == self.L1_High_bar:
                        self.L1_swing_x.pop()
                        self.L1_swing_y.pop()
                    self.L1_swing_x.append(bar_idx)
                    self.L1_swing_y.append(candle.high)
                    self._complete_and_reset_L2(bar_idx, candle.low, new_parent_direction=-1)
                    self.L1_High = candle.high
                    self.L1_High_bar = bar_idx
                    self.L1_Low = candle.low
                    self.L1_Low_bar = bar_idx
                    self._push_stitch_swing(bar_idx, candle.high, +1)
                    self._push_stitch_swing(bar_idx, candle.low, -1)
                    self.L1_Direction = -1

            else:
                # Pullback bar
                self._update_L2_pullback(candle, bar_idx)
                last_stitch_bar = self._get_last_stitch_bar()
                if last_stitch_bar < bar_idx:
                    child_swing = self._find_child_swing_at_bar(bar_idx)
                    if child_swing:
                        level_num, price, direction = child_swing
                        if self._stitch_swings and self._stitch_swings[-1][2] == direction:
                            self._pop_stitch_swing()
                        self._push_stitch_swing_child(bar_idx, price, direction, level_num)

        elif self.L1_Direction == -1:
            # DOWN direction rules (mirrored)
            if candle.low < self.L1_Low and candle.high <= self.L1_High:
                # Case 1: New low, no new high - continuing DOWN
                self._complete_and_reset_L2(bar_idx, candle.low)
                self.L1_Low = candle.low
                self.L1_Low_bar = bar_idx
                if self._stitch_swings and self._stitch_swings[-1][2] == -1:
                    self._pop_stitch_swing()
                self._push_stitch_swing(bar_idx, candle.low, -1)

            elif candle.high > self.L1_High and candle.low >= self.L1_Low:
                # Case 2: New high, no new low - reversal
                self.L1_swing_x.append(self.L1_Low_bar)
                self.L1_swing_y.append(self.L1_Low)
                self._complete_and_reset_L2(bar_idx, candle.high, new_parent_direction=+1)
                self.L1_High = candle.high
                self.L1_High_bar = bar_idx
                if self._stitch_swings and self._stitch_swings[-1][2] == +1:
                    self._pop_stitch_swing()
                self._push_stitch_swing(bar_idx, candle.high, +1)
                self.L1_Direction = 1

            elif candle.high > self.L1_High and candle.low < self.L1_Low:
                # Case 3: Outside bar
                bar_direction = self._determine_bar_direction(candle)
                if bar_direction == 1:
                    # Bullish outside bar - reversal
                    if self._stitch_swings and self._stitch_swings[-1][0] == self.L1_Low_bar:
                        self._pop_stitch_swing()
                    self.L1_swing_x.append(self.L1_Low_bar)
                    self.L1_swing_y.append(self.L1_Low)
                    self.L1_swing_x.append(bar_idx)
                    self.L1_swing_y.append(candle.low)
                    self._complete_and_reset_L2(bar_idx, candle.high, new_parent_direction=+1)
                    self.L1_Low = candle.low
                    self.L1_Low_bar = bar_idx
                    self.L1_High = candle.high
                    self.L1_High_bar = bar_idx
                    self._push_stitch_swing(bar_idx, candle.low, -1)
                    self._push_stitch_swing(bar_idx, candle.high, +1)
                    self.L1_Direction = 1
                else:
                    # Bearish outside bar - continuation
                    self.L1_swing_x.append(self.L1_Low_bar)
                    self.L1_swing_y.append(self.L1_Low)
                    self.L1_swing_x.append(bar_idx)
                    self.L1_swing_y.append(candle.high)
                    self._complete_and_reset_L2(bar_idx, candle.low)
                    self.L1_High = candle.high
                    self.L1_High_bar = bar_idx
                    self.L1_Low = candle.low
                    self.L1_Low_bar = bar_idx
                    self._push_stitch_swing(bar_idx, candle.high, +1)
                    self._push_stitch_swing(bar_idx, candle.low, -1)

            else:
                # Pullback bar
                self._update_L2_pullback(candle, bar_idx)
                last_stitch_bar = self._get_last_stitch_bar()
                if last_stitch_bar < bar_idx:
                    child_swing = self._find_child_swing_at_bar(bar_idx)
                    if child_swing:
                        level_num, price, direction = child_swing
                        if self._stitch_swings and self._stitch_swings[-1][2] == direction:
                            self._pop_stitch_swing()
                        self._push_stitch_swing_child(bar_idx, price, direction, level_num)

    def _get_bar_level(self, bar_idx: int) -> int:
        """Determine which level this bar belongs to."""
        if self.L1_Direction == 1 and self.L1_High_bar == bar_idx:
            return 1
        if self.L1_Direction == -1 and self.L1_Low_bar == bar_idx:
            return 1

        last_level_on_bar = None
        for i, (swing_bar, _, _) in enumerate(self._stitch_swings):
            if swing_bar == bar_idx:
                last_level_on_bar = self._stitch_swings_level[i] if i < len(self._stitch_swings_level) else None

        if last_level_on_bar is not None and last_level_on_bar > 0:
            return last_level_on_bar

        for i in range(len(self.levels) - 1, 0, -1):
            lvl = self.levels[i]
            if lvl.current_count > 0:
                return lvl.level

        return 1

    def _determine_lstm_event(self, bar_idx: int, pre_l1_direction: int) -> str:
        """Determine the LSTM event classification for this bar."""
        # REVERSAL: L1 direction changed
        if pre_l1_direction != 0 and self.L1_Direction != pre_l1_direction:
            return "REVERSAL"

        current_level = self._get_bar_level(bar_idx)
        prev_level = self._active_level_history[-1] if self._active_level_history else 1

        # SPAWN: Level increased
        if current_level > prev_level:
            return "SPAWN"

        return "EXTENSION"

    # ================================================================
    # PUBLIC API
    # ================================================================

    def process_session(self, candles: list[Candle]) -> list[BarState]:
        """
        Process all bars in a session and return per-bar MMLC state.

        Args:
            candles: List of OHLC candles for the session

        Returns:
            List of BarState objects, one per bar
        """
        self._candles = candles
        self._stitch_swings = []
        self._stitch_swings_level = []
        self._L1_count = 0
        self._prev_swing_direction = 0
        self._L1_leg = 0
        self._active_level_history = []

        # Reset wave levels
        self.levels = [
            WaveLevel(level=1),
            WaveLevel(level=2),
        ]

        if not candles:
            return []

        results: list[BarState] = []

        for bar_idx, candle in enumerate(candles):
            if bar_idx == 0:
                self._initialize_first_bar(candle, bar_idx)
                event = "EXTENSION"
            else:
                pre_l1_direction = self.L1_Direction
                self._process_bar(candle, bar_idx)
                event = self._determine_lstm_event(bar_idx, pre_l1_direction)

            # Track level history
            bar_level = self._get_bar_level(bar_idx)
            self._active_level_history.append(bar_level)

            # Determine direction from the active level
            level_direction = self.levels[bar_level - 1].direction
            direction_str = "UP" if level_direction == 1 else "DOWN"

            results.append(BarState(
                bar_idx=bar_idx,
                level=bar_level,
                direction=direction_str,
                event=event
            ))

        return results

    def calculate_outcomes(self, candles: list[Candle]) -> list[BarOutcome]:
        """
        Calculate forward-looking outcomes for all bars.

        Must be called AFTER process_session() with the same candles.

        Args:
            candles: List of OHLC candles (same as passed to process_session)

        Returns:
            List of BarOutcome objects, one per bar
        """
        if not candles:
            return []

        total_bars = len(candles)
        last_bar_idx = total_bars - 1
        session_close = candles[last_bar_idx].close

        outcomes: list[BarOutcome] = []

        for bar_idx, candle in enumerate(candles):
            current_close = candle.close

            # next_bar_delta
            if bar_idx < last_bar_idx:
                next_close = candles[bar_idx + 1].close
                next_bar_delta = next_close - current_close
            else:
                next_bar_delta = 0.0

            # session_close_delta
            session_close_delta = session_close - current_close

            # max up/down
            max_high = current_close
            min_low = current_close
            for j in range(bar_idx, total_bars):
                c = candles[j]
                if c.high > max_high:
                    max_high = c.high
                if c.low < min_low:
                    min_low = c.low

            outcomes.append(BarOutcome(
                next_bar_delta=next_bar_delta,
                session_close_delta=session_close_delta,
                session_max_up=max_high - current_close,
                session_max_down=min_low - current_close
            ))

        return outcomes
