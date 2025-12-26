"""
MMLC Development Engine

This is a blank-slate engine for iteratively developing MMLC waveform logic.
Rules are added one at a time based on user feedback during development.

Usage:
    1. Load a session in the /mmlc-dev page
    2. Click "Run" to process bars
    3. Review the waveform output
    4. Describe corrections needed
    5. Rules are added/modified here
    6. Repeat until logic is correct
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.waveform.wave import Wave, Candle, Direction
from app.config import WAVE_COLORS


@dataclass
class DevWave:
    """Development wave - simpler structure for iterating on logic."""
    id: int
    level: int
    direction: Direction
    start_time: datetime
    start_price: float
    end_time: datetime
    end_price: float
    start_bar_index: int
    end_bar_index: int
    parent_id: Optional[int] = None

    @property
    def color(self) -> str:
        """Get wave color based on level."""
        return WAVE_COLORS[(self.level - 1) % len(WAVE_COLORS)]

    def to_wave(self) -> Wave:
        """Convert to standard Wave for API response."""
        return Wave(
            id=self.id,
            level=self.level,
            direction=self.direction,
            start_time=self.start_time,
            start_price=self.start_price,
            end_time=self.end_time,
            end_price=self.end_price,
            parent_id=self.parent_id,
            is_active=True,
        )


@dataclass
class WaveLevel:
    """
    State for a single wave level (L1, L2, L3, etc.).

    Each level tracks:
    - Current extremes (high/low)
    - Origin point (where this level's tracking started)
    - Completed waves (historical retracements)
    - Spline segments (for intermediate visualization)
    """
    level: int                  # 1, 2, 3, etc.
    direction: int = 0          # -1=DOWN, 0=NEUTRAL, +1=UP
    high: float = 0.0           # Current highest price
    low: float = 0.0            # Current lowest price
    high_bar: int = 0           # Bar index where high occurred
    low_bar: int = 0            # Bar index where low occurred
    origin_bar: int = 0         # Start bar of current wave
    origin_price: float = 0.0   # Start price of current wave
    # Completed waves: [(origin_bar, origin_price, end_bar, end_price), ...]
    completed_waves: list = field(default_factory=list)
    # Spline segments for intermediate visualization
    spline_segments: list = field(default_factory=list)
    # For L1 only: swing point arrays
    swing_x: list = field(default_factory=list)  # Bar indices
    swing_y: list = field(default_factory=list)  # Prices

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


class MMLCDevEngine:
    """
    Development engine for iterating on MMLC logic.

    This engine starts empty. Rules are added iteratively
    based on user feedback during development.
    """

    def __init__(self):
        self._waves: list[DevWave] = []
        self._wave_id_counter: int = 0
        self._candles: list[Candle] = []

        # Wave levels - dynamic list that can grow to L3, L4, etc.
        # Index 0 = L1, Index 1 = L2, etc.
        self.levels: list[WaveLevel] = [
            WaveLevel(level=1),  # L1
            WaveLevel(level=2),  # L2
        ]

        # ================================================================
        # BACKWARD COMPATIBILITY PROPERTIES
        # These map old variable names to the new levels structure.
        # Will be removed once all code is refactored.
        # ================================================================

    # L1 backward compatibility properties
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

    @L1_swing_x.setter
    def L1_swing_x(self, value: list):
        self.levels[0].swing_x = value

    @property
    def L1_swing_y(self) -> list:
        return self.levels[0].swing_y

    @L1_swing_y.setter
    def L1_swing_y(self, value: list):
        self.levels[0].swing_y = value

    @property
    def L1_spline_segments(self) -> list:
        return self.levels[0].spline_segments

    @L1_spline_segments.setter
    def L1_spline_segments(self, value: list):
        self.levels[0].spline_segments = value

    # L2 backward compatibility properties
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

    @property
    def L2_completed_waves(self) -> list:
        return self.levels[1].completed_waves

    @L2_completed_waves.setter
    def L2_completed_waves(self, value: list):
        self.levels[1].completed_waves = value

    @property
    def L2_spline_segments(self) -> list:
        return self.levels[1].spline_segments

    @L2_spline_segments.setter
    def L2_spline_segments(self, value: list):
        self.levels[1].spline_segments = value

    def _next_wave_id(self) -> int:
        """Get next unique wave ID."""
        self._wave_id_counter += 1
        return self._wave_id_counter

    # ================================================================
    # GENERIC LEVEL METHODS
    # These work with any level (L1, L2, L3, etc.)
    # ================================================================

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

    def _complete_and_reset_level(
        self,
        level_num: int,
        bar_idx: int,
        new_origin_price: float,
        parent_direction: int
    ) -> None:
        """
        Complete current wave for level N and reset to new origin.
        Called when parent level makes a new extreme.

        Args:
            level_num: The level to reset (2 for L2, 3 for L3, etc.)
            bar_idx: Current bar index
            new_origin_price: Price at the new origin (parent's new extreme)
            parent_direction: Parent level's direction (+1=UP, -1=DOWN)
        """
        level = self._ensure_level_exists(level_num)

        # Check if there was retracement in the current wave
        if level.has_retracement():
            # Store completed wave: origin → retracement endpoint
            # Direction is opposite to parent
            if parent_direction == 1:
                # Parent UP, this level retraces DOWN → track lows
                level.completed_waves.append((
                    level.origin_bar, level.origin_price,
                    level.low_bar, level.low
                ))
            else:
                # Parent DOWN, this level retraces UP → track highs
                level.completed_waves.append((
                    level.origin_bar, level.origin_price,
                    level.high_bar, level.high
                ))

        # Reset level to new origin
        # Child direction is opposite to parent
        child_direction = -parent_direction
        level.reset(bar_idx, new_origin_price, child_direction)

        # Cascade: also reset the next level (L3 when L2 resets, etc.)
        # But only if next level exists and had tracking
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
        """
        Update level N during parent's pullback (when parent not making new extreme).
        Returns True if this level made a new extreme (triggering child cascade).

        Args:
            level_num: The level to update (2 for L2, 3 for L3, etc.)
            candle: Current candle
            bar_idx: Current bar index
            parent_direction: Parent level's direction (+1=UP, -1=DOWN)

        Returns:
            True if this level made a new extreme
        """
        level = self._ensure_level_exists(level_num)
        made_new_extreme = False

        if parent_direction == 1:
            # Parent UP → this level retraces DOWN → track lower lows
            if candle.high <= level.high and candle.low < level.low:
                level.low = candle.low
                level.low_bar = bar_idx
                made_new_extreme = True

                # Track spline segment for visualization
                if self._mode == "spline":
                    level.spline_segments.append((
                        level.origin_bar, level.origin_price,
                        bar_idx, candle.low
                    ))
        else:
            # Parent DOWN → this level retraces UP → track higher highs
            if candle.low >= level.low and candle.high > level.high:
                level.high = candle.high
                level.high_bar = bar_idx
                made_new_extreme = True

                # Track spline segment for visualization
                if self._mode == "spline":
                    level.spline_segments.append((
                        level.origin_bar, level.origin_price,
                        bar_idx, candle.high
                    ))

        # If this level made a new extreme, reset child level
        if made_new_extreme:
            # Ensure next level exists (auto-creates L3 when L2 makes extreme, etc.)
            self._ensure_level_exists(level_num + 1)
            # Child direction is opposite to this level's retracement direction
            child_parent_dir = -parent_direction  # This level's direction
            self._complete_and_reset_level(
                level_num + 1,
                bar_idx,
                candle.low if parent_direction == 1 else candle.high,
                child_parent_dir
            )
        else:
            # This level didn't make new extreme = this level is pulling back
            # Let child level track its own retracement (recursive cascade)
            child_level = self._get_level(level_num + 1)
            if child_level is not None:
                child_parent_dir = -parent_direction  # This level's direction
                self._update_level_pullback(
                    level_num + 1,
                    candle,
                    bar_idx,
                    child_parent_dir
                )

        return made_new_extreme

    def process_session(
        self,
        candles: list[Candle],
        start_bar: int = 0,
        end_bar: Optional[int] = None,
        mode: str = "complete"
    ) -> list[Wave]:
        """
        Process bars from start_bar to end_bar.

        Args:
            candles: List of OHLC candles for the session
            start_bar: First bar to process (0-indexed)
            end_bar: Last bar to process (inclusive, defaults to last bar)
            mode: "complete" (final waveform only) or "spline" (all intermediate lines)

        Returns:
            List of Wave objects representing the waveform state
        """
        self._candles = candles
        self._waves = []
        self._wave_id_counter = 0
        self._mode = mode

        # Reset all wave levels
        self.levels = [
            WaveLevel(level=1),  # L1
            WaveLevel(level=2),  # L2
        ]

        if end_bar is None:
            end_bar = len(candles) - 1

        # Clamp to valid range
        start_bar = max(0, start_bar)
        end_bar = min(len(candles) - 1, end_bar)

        if start_bar > end_bar:
            return []

        # ============================================================
        # MMLC RULES
        # ============================================================

        for bar_idx in range(start_bar, end_bar + 1):
            candle = candles[bar_idx]

            if bar_idx == start_bar:
                # Rule 1: Initialize on first bar
                self._initialize_first_bar(candle, bar_idx)
            else:
                # Subsequent bars: Update L1_High/L1_Low
                self._process_bar(candle, bar_idx)

        # Add developing leg: from last swing point to current extreme
        self._add_developing_leg(end_bar)

        # Track historical counts per level BEFORE adding developing legs
        historical_counts = {level.level: len(level.completed_waves) for level in self.levels}

        # Add developing legs for all levels >= 2
        # Parent direction alternates: L1_Dir for L2, -L1_Dir for L3, L1_Dir for L4, etc.
        parent_dir = self.L1_Direction
        for level in self.levels[1:]:  # Skip L1
            self._add_developing_leg_level(level.level, parent_dir)
            parent_dir = -parent_dir  # Flip for next level

        # Convert L1 swing points to Wave objects for display
        waves = self._build_waves()

        # Add waves for all levels >= 2
        for level in self.levels[1:]:
            level_waves = self._build_level_waves(
                level.level,
                historical_counts.get(level.level, 0)
            )
            waves.extend(level_waves)

        # In spline mode, add spline waves for all levels
        if self._mode == "spline":
            spline_waves = self._build_spline_waves()  # L1 splines
            waves.extend(spline_waves)
            # Add splines for all levels >= 2
            for level in self.levels[1:]:
                level_spline_waves = self._build_level_spline_waves(level.level)
                waves.extend(level_spline_waves)

        # Add close leg (final segment from deepest level's extreme to close price)
        close_leg = self._build_close_leg(end_bar)
        if close_leg:
            waves.append(close_leg)

        return waves

    def _initialize_first_bar(self, candle: Candle, bar_idx: int) -> None:
        """
        Rule 1: Initialize L1 state on the first bar.

        Step 1: Determine direction from close/open relationship
        Step 2: Check if preswing is needed (skip if open equals the natural starting extreme)
        Step 3: Push swing point (low if UP, high if DOWN)
        """
        # Step 1: Determine L1_Direction from close/open relationship
        if candle.close > candle.open:
            self.L1_Direction = 1  # UP
        elif candle.close < candle.open:
            self.L1_Direction = -1  # DOWN
        else:
            # Doji: close == open, use midpoint to decide
            midpoint = (candle.high + candle.low) / 2
            if candle.close < midpoint:
                self.L1_Direction = -1  # DOWN
            elif candle.close > midpoint:
                self.L1_Direction = 1  # UP
            else:
                # Dead center - default to UP for first bar
                self.L1_Direction = 1

        # Step 2: Check if we need the artificial preswing at index -1
        # Skip preswing if:
        # - UP bar and open == low (bar naturally starts at its low)
        # - DOWN bar and open == high (bar naturally starts at its high)
        needs_preswing = True
        if self.L1_Direction == 1 and candle.open == candle.low:
            needs_preswing = False
        elif self.L1_Direction == -1 and candle.open == candle.high:
            needs_preswing = False

        if needs_preswing:
            # Add anchor point at index -1, price = open
            self.L1_swing_x.append(-1)
            self.L1_swing_y.append(candle.open)

        # Step 3: Push swing point based on direction
        if self.L1_Direction == 1:
            # UP direction: swing LOW at bar 0
            self.L1_swing_x.append(bar_idx)
            self.L1_swing_y.append(candle.low)
        else:
            # DOWN direction: swing HIGH at bar 0
            self.L1_swing_x.append(bar_idx)
            self.L1_swing_y.append(candle.high)

        # Update L1_High and L1_Low to bar's extremes
        self.L1_High = candle.high
        self.L1_Low = candle.low
        self.L1_High_bar = bar_idx
        self.L1_Low_bar = bar_idx

        # Initialize L2: starts from the L1 extreme in the current direction
        if self.L1_Direction == 1:
            # UP direction: L2 tracks retracement from the high
            self.L2_High = candle.high
            self.L2_Low = candle.high  # Same initially (no retracement yet)
            self.L2_High_bar = bar_idx
            self.L2_Low_bar = bar_idx
            self.L2_origin_bar = bar_idx
            self.L2_origin_price = candle.high
        else:
            # DOWN direction: L2 tracks retracement from the low
            self.L2_High = candle.low  # Same initially
            self.L2_Low = candle.low
            self.L2_High_bar = bar_idx
            self.L2_Low_bar = bar_idx
            self.L2_origin_bar = bar_idx
            self.L2_origin_price = candle.low

    def _process_bar(self, candle: Candle, bar_idx: int) -> None:
        """
        Process subsequent bars (after bar 0).
        """
        # Store old direction to detect L1 reversals
        old_L1_direction = self.L1_Direction

        if self.L1_Direction == 1:
            # UP direction rules
            if candle.high > self.L1_High and candle.low >= self.L1_Low:
                # Case 1: New high, no new low - continuing UP
                # Track spline segment (from last swing to new high) for spline mode
                if self._mode == "spline" and len(self.L1_swing_x) > 0:
                    origin_bar = self.L1_swing_x[-1]
                    origin_price = self.L1_swing_y[-1]
                    self.L1_spline_segments.append((origin_bar, origin_price, bar_idx, candle.high))

                # L2: Complete current L2 wave and reset to new L1 extreme
                self._complete_and_reset_L2(bar_idx, candle.high)

                # Update L1_High
                self.L1_High = candle.high
                self.L1_High_bar = bar_idx

            elif candle.low < self.L1_Low and candle.high <= self.L1_High:
                # Case 2: New low, no new high - reversal
                # Push swing high to array (confirms the high)
                self.L1_swing_x.append(self.L1_High_bar)
                self.L1_swing_y.append(self.L1_High)
                # Track spline segment for the initial developing leg of new direction
                if self._mode == "spline":
                    self.L1_spline_segments.append((self.L1_High_bar, self.L1_High, bar_idx, candle.low))

                # L2: Complete current L2 wave and reset to new direction's extreme
                self._complete_and_reset_L2(bar_idx, candle.low)

                # Update L1_Low to new low
                self.L1_Low = candle.low
                self.L1_Low_bar = bar_idx
                # Direction changes to DOWN
                self.L1_Direction = -1

            elif candle.high > self.L1_High and candle.low < self.L1_Low:
                # Case 3: Outside bar - both new high and new low
                if candle.close > candle.open:
                    # Bullish bar: low happened first
                    # Push swing low to array
                    self.L1_swing_x.append(bar_idx)
                    self.L1_swing_y.append(candle.low)

                    # L2: Reset to the new high (final extreme after outside bar)
                    self._complete_and_reset_L2(bar_idx, candle.high)

                    # Update both extremes
                    self.L1_Low = candle.low
                    self.L1_Low_bar = bar_idx
                    self.L1_High = candle.high
                    self.L1_High_bar = bar_idx

                elif candle.close < candle.open:
                    # Bearish bar: high happened first
                    # Push swing high to array
                    self.L1_swing_x.append(bar_idx)
                    self.L1_swing_y.append(candle.high)

                    # L2: Reset to the new low (final extreme after outside bar)
                    self._complete_and_reset_L2(bar_idx, candle.low)

                    # Update both extremes
                    self.L1_High = candle.high
                    self.L1_High_bar = bar_idx
                    self.L1_Low = candle.low
                    self.L1_Low_bar = bar_idx
                    # Direction changes to DOWN
                    self.L1_Direction = -1

                else:
                    # Doji: use midpoint to determine which happened first
                    midpoint = (candle.high + candle.low) / 2
                    if candle.close >= midpoint:
                        # Above midpoint: treat as bullish (low first)
                        self.L1_swing_x.append(bar_idx)
                        self.L1_swing_y.append(candle.low)

                        # L2: Reset to the new high
                        self._complete_and_reset_L2(bar_idx, candle.high)

                        self.L1_Low = candle.low
                        self.L1_Low_bar = bar_idx
                        self.L1_High = candle.high
                        self.L1_High_bar = bar_idx
                    else:
                        # Below midpoint: treat as bearish (high first)
                        self.L1_swing_x.append(bar_idx)
                        self.L1_swing_y.append(candle.high)

                        # L2: Reset to the new low
                        self._complete_and_reset_L2(bar_idx, candle.low)

                        self.L1_High = candle.high
                        self.L1_High_bar = bar_idx
                        self.L1_Low = candle.low
                        self.L1_Low_bar = bar_idx
                        # Direction changes to DOWN
                        self.L1_Direction = -1

            else:
                # No new L1 extreme - this is a pullback bar
                # L2: Track retracement (L1 is UP, track lower lows)
                self._update_L2_pullback(candle, bar_idx)

        elif self.L1_Direction == -1:
            # DOWN direction rules (mirrored from UP)
            if candle.low < self.L1_Low and candle.high <= self.L1_High:
                # Case 1: New low, no new high - continuing DOWN
                # Track spline segment (from last swing to new low) for spline mode
                if self._mode == "spline" and len(self.L1_swing_x) > 0:
                    origin_bar = self.L1_swing_x[-1]
                    origin_price = self.L1_swing_y[-1]
                    self.L1_spline_segments.append((origin_bar, origin_price, bar_idx, candle.low))

                # L2: Complete current L2 wave and reset to new L1 extreme
                self._complete_and_reset_L2(bar_idx, candle.low)

                # Update L1_Low
                self.L1_Low = candle.low
                self.L1_Low_bar = bar_idx

            elif candle.high > self.L1_High and candle.low >= self.L1_Low:
                # Case 2: New high, no new low - reversal
                # Push swing low to array (confirms the low)
                self.L1_swing_x.append(self.L1_Low_bar)
                self.L1_swing_y.append(self.L1_Low)
                # Track spline segment for the initial developing leg of new direction
                if self._mode == "spline":
                    self.L1_spline_segments.append((self.L1_Low_bar, self.L1_Low, bar_idx, candle.high))

                # L2: Complete current L2 wave and reset to new direction's extreme
                self._complete_and_reset_L2(bar_idx, candle.high)

                # Update L1_High to new high
                self.L1_High = candle.high
                self.L1_High_bar = bar_idx
                # Direction changes to UP
                self.L1_Direction = 1

            elif candle.high > self.L1_High and candle.low < self.L1_Low:
                # Case 3: Outside bar - both new high and new low
                if candle.close > candle.open:
                    # Bullish bar: low happened first
                    # Push swing low to array
                    self.L1_swing_x.append(bar_idx)
                    self.L1_swing_y.append(candle.low)

                    # L2: Reset to the new high (final extreme after outside bar)
                    self._complete_and_reset_L2(bar_idx, candle.high)

                    # Update both extremes
                    self.L1_Low = candle.low
                    self.L1_Low_bar = bar_idx
                    self.L1_High = candle.high
                    self.L1_High_bar = bar_idx
                    # Direction changes to UP
                    self.L1_Direction = 1

                elif candle.close < candle.open:
                    # Bearish bar: high happened first
                    # Push swing high to array
                    self.L1_swing_x.append(bar_idx)
                    self.L1_swing_y.append(candle.high)

                    # L2: Reset to the new low (final extreme after outside bar)
                    self._complete_and_reset_L2(bar_idx, candle.low)

                    # Update both extremes
                    self.L1_High = candle.high
                    self.L1_High_bar = bar_idx
                    self.L1_Low = candle.low
                    self.L1_Low_bar = bar_idx
                    # Stay DOWN

                else:
                    # Doji: use midpoint to determine which happened first
                    midpoint = (candle.high + candle.low) / 2
                    if candle.close >= midpoint:
                        # Above midpoint: treat as bullish (low first)
                        self.L1_swing_x.append(bar_idx)
                        self.L1_swing_y.append(candle.low)

                        # L2: Reset to the new high
                        self._complete_and_reset_L2(bar_idx, candle.high)

                        self.L1_Low = candle.low
                        self.L1_Low_bar = bar_idx
                        self.L1_High = candle.high
                        self.L1_High_bar = bar_idx
                        # Direction changes to UP
                        self.L1_Direction = 1
                    else:
                        # Below midpoint: treat as bearish (high first)
                        self.L1_swing_x.append(bar_idx)
                        self.L1_swing_y.append(candle.high)

                        # L2: Reset to the new low
                        self._complete_and_reset_L2(bar_idx, candle.low)

                        self.L1_High = candle.high
                        self.L1_High_bar = bar_idx
                        self.L1_Low = candle.low
                        self.L1_Low_bar = bar_idx
                        # Stay DOWN

            else:
                # No new L1 extreme - this is a pullback bar
                # L2: Track retracement (L1 is DOWN, track higher highs)
                self._update_L2_pullback(candle, bar_idx)

    def _complete_and_reset_L2(self, bar_idx: int, new_l2_origin_price: float) -> None:
        """
        Complete current L2 wave (if there was retracement) and reset L2 to new origin.

        Called when L1 makes a new extreme (continuation, reversal, or outside bar).
        Now delegates to the generic _complete_and_reset_level method.
        """
        # Delegate to generic method, which handles L2 and cascades to L3+
        self._complete_and_reset_level(
            level_num=2,
            bar_idx=bar_idx,
            new_origin_price=new_l2_origin_price,
            parent_direction=self.L1_Direction
        )

    def _update_L2_pullback(self, candle: Candle, bar_idx: int) -> None:
        """
        Update L2 during pullback (when L1 is not making a new extreme).
        Now delegates to the generic _update_level_pullback method.
        """
        # Delegate to generic method, which handles L2 and cascades to L3+
        self._update_level_pullback(
            level_num=2,
            candle=candle,
            bar_idx=bar_idx,
            parent_direction=self.L1_Direction
        )

    def _add_developing_leg(self, current_bar: int) -> None:
        """
        Add developing leg from last swing point to current extreme.

        If L1_Direction is UP: draw to L1_High
        If L1_Direction is DOWN: draw to L1_Low
        """
        if len(self.L1_swing_x) == 0:
            return

        if self.L1_Direction == 1:
            # UP direction: developing leg goes to current high
            self.L1_swing_x.append(self.L1_High_bar)
            self.L1_swing_y.append(self.L1_High)
        elif self.L1_Direction == -1:
            # DOWN direction: developing leg goes to current low
            self.L1_swing_x.append(self.L1_Low_bar)
            self.L1_swing_y.append(self.L1_Low)

    def _add_developing_leg_level(self, level_num: int, parent_direction: int) -> None:
        """
        Add developing leg for any level (L2, L3, L4, etc.).

        Args:
            level_num: The level to add developing leg for (2 for L2, etc.)
            parent_direction: Direction of the parent level (+1=UP, -1=DOWN)
        """
        level = self._get_level(level_num)
        if not level:
            return

        # Only add developing leg if there was retracement
        if level.has_retracement():
            if parent_direction == 1:
                # Parent UP: this level retraces DOWN, track lows
                level.completed_waves.append((
                    level.origin_bar, level.origin_price,
                    level.low_bar, level.low
                ))
            else:
                # Parent DOWN: this level retraces UP, track highs
                level.completed_waves.append((
                    level.origin_bar, level.origin_price,
                    level.high_bar, level.high
                ))

    def _add_developing_leg_L2(self, current_bar: int) -> None:
        """Backward compatibility wrapper - delegates to generic method."""
        self._add_developing_leg_level(2, self.L1_Direction)

    def _build_waves(self) -> list[Wave]:
        """
        Convert L1 swing points to Wave objects for display.

        Swing points define the vertices of the waveform.
        Each consecutive pair of points becomes a wave segment.
        """
        from datetime import timedelta

        waves = []

        # Need at least 2 points to draw a wave
        if len(self.L1_swing_x) < 2:
            return waves

        for i in range(len(self.L1_swing_x) - 1):
            x1, y1 = self.L1_swing_x[i], self.L1_swing_y[i]
            x2, y2 = self.L1_swing_x[i + 1], self.L1_swing_y[i + 1]

            # Determine direction from price movement
            if y2 > y1:
                direction = Direction.UP
            else:
                direction = Direction.DOWN

            # Get timestamps for the bar indices
            # Handle index -1 (before first bar) by subtracting one bar interval
            if x1 < 0:
                # Calculate interval between bars (assume uniform spacing)
                if len(self._candles) >= 2:
                    interval = self._candles[1].timestamp - self._candles[0].timestamp
                else:
                    interval = timedelta(minutes=10)  # Default fallback
                t1 = self._candles[0].timestamp - interval
            else:
                t1 = self._candles[min(x1, len(self._candles) - 1)].timestamp

            if x2 < 0:
                if len(self._candles) >= 2:
                    interval = self._candles[1].timestamp - self._candles[0].timestamp
                else:
                    interval = timedelta(minutes=10)
                t2 = self._candles[0].timestamp - interval
            else:
                t2 = self._candles[min(x2, len(self._candles) - 1)].timestamp

            wave = Wave(
                id=i + 1,
                level=1,
                direction=direction,
                start_time=t1,
                start_price=y1,
                end_time=t2,
                end_price=y2,
                parent_id=None,
                is_active=True,
            )
            waves.append(wave)

        return waves

    def _build_level_waves(self, level_num: int, historical_count: int) -> list[Wave]:
        """
        Convert completed waves for any level (L2, L3, L4, etc.) to Wave objects.

        Args:
            level_num: The level to build waves for (2 for L2, 3 for L3, etc.)
            historical_count: Number of waves completed before adding developing leg.
                             Entries after this index are "developing" waves.

        Mode behavior:
        - Complete mode: Only show the developing wave (if exists). Never show historical.
        - Spline mode: Show ALL historical as dotted + developing as solid.
        """
        from datetime import timedelta

        level = self._get_level(level_num)
        if not level:
            return []

        waves = []
        completed = level.completed_waves

        if len(completed) == 0:
            return waves

        # Separate historical from developing
        historical = completed[:historical_count]
        developing = completed[historical_count:]

        # ID offset: L2=1000+, L3=1500+, L4=2000+, etc.
        id_offset = level_num * 500

        if self._mode == "complete":
            # Complete mode: Only show developing wave (solid), no historical
            waves_to_process = [(i + historical_count, entry, False)
                                for i, entry in enumerate(developing)]
        else:
            # Spline mode: All historical as dotted + developing as solid
            waves_to_process = [(i, entry, True) for i, entry in enumerate(historical)]
            waves_to_process += [(i + historical_count, entry, False)
                                 for i, entry in enumerate(developing)]

        for actual_idx, (x1, y1, x2, y2), is_spline in waves_to_process:
            # Determine direction from price movement
            if y2 > y1:
                direction = Direction.UP
            else:
                direction = Direction.DOWN

            # Get timestamps for the bar indices
            if x1 < 0:
                if len(self._candles) >= 2:
                    interval = self._candles[1].timestamp - self._candles[0].timestamp
                else:
                    interval = timedelta(minutes=10)
                t1 = self._candles[0].timestamp - interval
            else:
                t1 = self._candles[min(x1, len(self._candles) - 1)].timestamp

            if x2 < 0:
                if len(self._candles) >= 2:
                    interval = self._candles[1].timestamp - self._candles[0].timestamp
                else:
                    interval = timedelta(minutes=10)
                t2 = self._candles[0].timestamp - interval
            else:
                t2 = self._candles[min(x2, len(self._candles) - 1)].timestamp

            wave = Wave(
                id=id_offset + actual_idx,  # Level-based offset to avoid conflicts
                level=level_num,  # Use actual level number
                direction=direction,
                start_time=t1,
                start_price=y1,
                end_time=t2,
                end_price=y2,
                parent_id=None,
                is_active=True,
                is_spline=is_spline,
            )
            waves.append(wave)

        return waves

    def _build_L2_waves(self, historical_l2_count: int) -> list[Wave]:
        """Backward compatibility wrapper - delegates to generic method."""
        return self._build_level_waves(2, historical_l2_count)

    def _build_spline_waves(self) -> list[Wave]:
        """
        Build spline waves - lines from origin swing to each intermediate extreme.

        These show the progression of the developing leg as L1_High/L1_Low updates.
        Each segment stores its own origin point so splines persist across reversals.
        """
        from datetime import timedelta

        waves = []

        if len(self.L1_spline_segments) == 0:
            return waves

        # Create a wave for each spline segment
        for i, (origin_bar, origin_price, end_bar, end_price) in enumerate(self.L1_spline_segments):
            # Get timestamp for origin point
            if origin_bar < 0:
                if len(self._candles) >= 2:
                    interval = self._candles[1].timestamp - self._candles[0].timestamp
                else:
                    interval = timedelta(minutes=10)
                origin_time = self._candles[0].timestamp - interval
            else:
                origin_time = self._candles[min(origin_bar, len(self._candles) - 1)].timestamp

            # Get timestamp for end point
            end_time = self._candles[min(end_bar, len(self._candles) - 1)].timestamp

            if end_price > origin_price:
                direction = Direction.UP
            else:
                direction = Direction.DOWN

            wave = Wave(
                id=750 + i,  # L1 spline offset (1*500+250 = 750) to avoid conflicts with L2 waves (1000+)
                level=1,  # Use level 1 (yellow) for spline waves
                direction=direction,
                start_time=origin_time,
                start_price=origin_price,
                end_time=end_time,
                end_price=end_price,
                parent_id=None,
                is_active=True,
                is_spline=True,  # Mark as spline for dotted line rendering
            )
            waves.append(wave)

        return waves

    def _build_level_spline_waves(self, level_num: int) -> list[Wave]:
        """
        Build spline waves for any level (L2, L3, L4, etc.).

        These show the progression of the retracement as the level's extreme updates.

        Args:
            level_num: The level to build splines for (2 for L2, 3 for L3, etc.)
        """
        from datetime import timedelta

        level = self._get_level(level_num)
        if not level:
            return []

        waves = []

        if len(level.spline_segments) == 0:
            return waves

        # Spline ID offset: L2=1250+, L3=1750+, L4=2250+, etc. (midpoint between level ranges)
        spline_id_offset = level_num * 500 + 250

        # Create a wave for each spline segment
        for i, (origin_bar, origin_price, end_bar, end_price) in enumerate(level.spline_segments):
            # Get timestamp for origin point
            if origin_bar < 0:
                if len(self._candles) >= 2:
                    interval = self._candles[1].timestamp - self._candles[0].timestamp
                else:
                    interval = timedelta(minutes=10)
                origin_time = self._candles[0].timestamp - interval
            else:
                origin_time = self._candles[min(origin_bar, len(self._candles) - 1)].timestamp

            # Get timestamp for end point
            end_time = self._candles[min(end_bar, len(self._candles) - 1)].timestamp

            if end_price > origin_price:
                direction = Direction.UP
            else:
                direction = Direction.DOWN

            wave = Wave(
                id=spline_id_offset + i,  # Level-based spline offset
                level=level_num,  # Use actual level number
                direction=direction,
                start_time=origin_time,
                start_price=origin_price,
                end_time=end_time,
                end_price=end_price,
                parent_id=None,
                is_active=True,
                is_spline=True,  # Mark as spline for dotted line rendering
            )
            waves.append(wave)

        return waves

    def _build_L2_spline_waves(self) -> list[Wave]:
        """Backward compatibility wrapper - delegates to generic method."""
        return self._build_level_spline_waves(2)

    def _build_close_leg(self, end_bar: int) -> Optional[Wave]:
        """
        Build a close leg from the deepest level's extreme to the close price.

        This completes the waveform by showing the final retracement to close.
        Returns None if close is at high or low (no retracement to show).
        """
        from datetime import timedelta

        if end_bar < 0 or end_bar >= len(self._candles):
            print(f"[CLOSE LEG] end_bar out of range: {end_bar}")
            return None

        candle = self._candles[end_bar]
        close = candle.close

        import sys
        print(f"[CLOSE LEG] Bar {end_bar}: close={close}, high={candle.high}, low={candle.low}", file=sys.stderr)
        print(f"[CLOSE LEG] Levels: {[f'L{l.level}(h={l.high:.5f},l={l.low:.5f},ret={l.has_retracement()})' for l in self.levels]}", file=sys.stderr)

        # Skip if close is at high or low (no retracement to show)
        if close == candle.high or close == candle.low:
            print(f"[CLOSE LEG] Skipping - close equals high or low", flush=True)
            return None

        # Find the deepest level with retracement
        deepest_level = None
        for level in reversed(self.levels):
            if level.has_retracement():
                deepest_level = level
                print(f"[CLOSE LEG] Found deepest level: L{level.level}", flush=True)
                break

        # Fall back to L1 if no level has retracement
        if deepest_level is None:
            deepest_level = self.levels[0]
            print(f"[CLOSE LEG] Fell back to L1", flush=True)

        # Determine the direction of the deepest level
        # Direction alternates: L1=L1_Dir, L2=-L1_Dir, L3=L1_Dir, L4=-L1_Dir, etc.
        # For level N: direction = L1_Direction if N is odd, -L1_Direction if N is even
        if deepest_level.level % 2 == 1:
            deepest_direction = self.L1_Direction
        else:
            deepest_direction = -self.L1_Direction

        # Determine origin point based on deepest level's tracking direction
        # If deepest is tracking UP (parent DOWN): origin = level's high
        # If deepest is tracking DOWN (parent UP): origin = level's low
        if deepest_direction == 1:
            # Deepest level is UP, so it's tracking highs - origin is the high
            origin_bar = deepest_level.high_bar
            origin_price = deepest_level.high
        else:
            # Deepest level is DOWN, so it's tracking lows - origin is the low
            origin_bar = deepest_level.low_bar
            origin_price = deepest_level.low

        # The close leg is the NEXT level after deepest
        close_leg_level = deepest_level.level + 1

        # Calculate timestamps
        origin_time = self._candles[min(origin_bar, len(self._candles) - 1)].timestamp

        # End time is slightly after the last bar (like the anchor swing at bar -1)
        if len(self._candles) >= 2:
            interval = self._candles[1].timestamp - self._candles[0].timestamp
        else:
            interval = timedelta(minutes=10)
        end_time = candle.timestamp + interval

        # Determine direction from price movement
        if close > origin_price:
            direction = Direction.UP
        else:
            direction = Direction.DOWN

        print(f"[CLOSE LEG] Creating: L{close_leg_level} from {origin_price:.5f} to {close:.5f}", flush=True)

        return Wave(
            id=9000,  # Unique ID for close leg
            level=close_leg_level,
            direction=direction,
            start_time=origin_time,
            start_price=origin_price,
            end_time=end_time,
            end_price=close,
            parent_id=None,
            is_active=True,
            is_spline=False,
        )

    # ================================================================
    # HELPER METHODS
    # These will be added as needed when implementing rules
    # ================================================================

    def _create_wave(
        self,
        level: int,
        direction: Direction,
        start_time: datetime,
        start_price: float,
        end_time: datetime,
        end_price: float,
        start_bar_index: int,
        end_bar_index: int,
        parent_id: Optional[int] = None,
    ) -> DevWave:
        """Create a new wave and add it to the wave list."""
        wave = DevWave(
            id=self._next_wave_id(),
            level=level,
            direction=direction,
            start_time=start_time,
            start_price=start_price,
            end_time=end_time,
            end_price=end_price,
            start_bar_index=start_bar_index,
            end_bar_index=end_bar_index,
            parent_id=parent_id,
        )
        self._waves.append(wave)
        return wave

    def _get_active_wave_at_level(self, level: int) -> Optional[DevWave]:
        """Get the most recent wave at a given level."""
        for wave in reversed(self._waves):
            if wave.level == level:
                return wave
        return None
