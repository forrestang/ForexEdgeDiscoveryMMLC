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
class SwingLabel:
    """Label info for a swing point."""
    bar: int
    timestamp: datetime
    price: float
    is_high: bool
    child_level: Optional[int] = None  # e.g., 2 for L2
    child_price: Optional[float] = None  # Price of the child swing
    bars_ago: Optional[int] = None  # Bars from child swing to this swing


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
        self._stitch_annotations: list[dict] = []  # Debug annotations for stitch mode
        self._swing_labels: list[SwingLabel] = []  # Labels for swing points

        # Stitch mode: permanent legs recorded on L1 direction change
        # Each entry: (start_bar, start_price, end_bar, end_price, direction)
        self._stitch_permanent_legs: list[tuple[int, float, int, float, int]] = []
        self._prev_L1_Direction: int = 0  # Track previous direction for change detection

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

    @property
    def stitch_annotations(self) -> list[dict]:
        """Get debug annotations from last stitch build."""
        return self._stitch_annotations

    @property
    def swing_labels(self) -> list[SwingLabel]:
        """Get swing labels from last run."""
        return self._swing_labels

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

    def _record_stitch_direction_change(self, new_direction: int) -> None:
        """
        Record a permanent leg when L1 direction changes (for stitch mode).
        Called at the moment of direction reversal.
        """
        if self._mode != "stitch":
            return

        old_direction = self._prev_L1_Direction

        # Only record if direction is actually changing (not initial setup)
        if old_direction != 0 and old_direction != new_direction:
            if old_direction == 1:
                # Was UP, now DOWN - record the completed UP leg
                # From last swing low to the high that just ended
                if len(self.L1_swing_x) >= 2:
                    start_bar = self.L1_swing_x[-2] if len(self.L1_swing_x) >= 2 else 0
                    start_price = self.L1_swing_y[-2] if len(self.L1_swing_y) >= 2 else self.L1_Low
                else:
                    start_bar = self.L1_Low_bar
                    start_price = self.L1_Low
                end_bar = self.L1_High_bar
                end_price = self.L1_High
                self._stitch_permanent_legs.append((start_bar, start_price, end_bar, end_price, 1))
            else:
                # Was DOWN, now UP - record the completed DOWN leg
                # From last swing high to the low that just ended
                if len(self.L1_swing_x) >= 2:
                    start_bar = self.L1_swing_x[-2] if len(self.L1_swing_x) >= 2 else 0
                    start_price = self.L1_swing_y[-2] if len(self.L1_swing_y) >= 2 else self.L1_High
                else:
                    start_bar = self.L1_High_bar
                    start_price = self.L1_High
                end_bar = self.L1_Low_bar
                end_price = self.L1_Low
                self._stitch_permanent_legs.append((start_bar, start_price, end_bar, end_price, -1))

        # Update previous direction tracker
        self._prev_L1_Direction = new_direction

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
                if self._mode in ("spline", "stitch"):
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
                if self._mode in ("spline", "stitch"):
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
            mode: "complete" (final waveform only), "spline" (all intermediate lines),
                  or "stitch" (continuous waveform stitching all levels)

        Returns:
            List of Wave objects representing the waveform state
        """
        self._candles = candles
        self._waves = []
        self._wave_id_counter = 0
        self._mode = mode

        # Reset stitch mode state
        self._stitch_permanent_legs = []
        self._prev_L1_Direction = 0

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

        # Build waves based on mode
        if self._mode == "stitch":
            # Stitch mode: Custom display based on current state
            waves = self._build_stitch_display_waves(end_bar)

        else:
            # Complete or spline mode
            waves = self._build_waves()

            for level in self.levels[1:]:
                level_waves = self._build_level_waves(
                    level.level,
                    historical_counts.get(level.level, 0)
                )
                waves.extend(level_waves)

            if self._mode == "spline":
                spline_waves = self._build_spline_waves()
                waves.extend(spline_waves)
                for level in self.levels[1:]:
                    level_spline_waves = self._build_level_spline_waves(level.level)
                    waves.extend(level_spline_waves)

            close_leg = self._build_close_leg(end_bar)
            if close_leg:
                waves.append(close_leg)

        # Build swing labels for all modes
        self._build_swing_labels(end_bar)

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

        # Initialize previous direction tracker for stitch mode
        self._prev_L1_Direction = self.L1_Direction

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
                if self._mode in ("spline", "stitch") and len(self.L1_swing_x) > 0:
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
                if self._mode in ("spline", "stitch"):
                    self.L1_spline_segments.append((self.L1_High_bar, self.L1_High, bar_idx, candle.low))

                # L2: Complete current L2 wave and reset to new direction's extreme
                self._complete_and_reset_L2(bar_idx, candle.low)

                # Update L1_Low to new low
                self.L1_Low = candle.low
                self.L1_Low_bar = bar_idx
                # Direction changes to DOWN
                self._record_stitch_direction_change(-1)
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
                    self._record_stitch_direction_change(-1)
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
                        self._record_stitch_direction_change(-1)
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
                if self._mode in ("spline", "stitch") and len(self.L1_swing_x) > 0:
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
                if self._mode in ("spline", "stitch"):
                    self.L1_spline_segments.append((self.L1_Low_bar, self.L1_Low, bar_idx, candle.high))

                # L2: Complete current L2 wave and reset to new direction's extreme
                self._complete_and_reset_L2(bar_idx, candle.high)

                # Update L1_High to new high
                self.L1_High = candle.high
                self.L1_High_bar = bar_idx
                # Direction changes to UP
                self._record_stitch_direction_change(1)
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
                    self._record_stitch_direction_change(1)
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
                        self._record_stitch_direction_change(1)
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

    def _is_swing_high(self, index: int) -> bool:
        """
        Determine if the L1 swing at index is a HIGH or LOW.

        Returns True if HIGH, False if LOW.
        """
        L1 = self.levels[0]
        if index == 0:
            # First swing: compare with next to determine type
            if len(L1.swing_y) > 1:
                return L1.swing_y[0] > L1.swing_y[1]
            # Only one swing - use L1 direction (if DOWN, first is HIGH)
            return L1.direction == -1
        else:
            # Compare with previous: if higher than previous, it's HIGH
            return L1.swing_y[index] > L1.swing_y[index - 1]

    def _build_swing_labels(self, end_bar: int) -> None:
        """
        Build swing labels for ALL swing points (L1, L2, L3, etc.) - stitch mode only.

        For each swing at any level, find the most recent child swing (opposite type)
        that preceded it. This shows what deeper level swing led to this swing.
        Label format: Line1 = child swing price, Line2 = bars ago.
        Only create label if there IS a child swing.
        """
        self._swing_labels = []

        # Only build labels in stitch mode
        if self._mode != "stitch":
            return

        # Collect all swing points from all levels
        # Each entry: (bar, price, level, is_high)
        all_swings = []

        # L1 swings from swing_x/swing_y
        L1 = self.levels[0]
        for i in range(len(L1.swing_x)):
            bar = L1.swing_x[i]
            price = L1.swing_y[i]
            if bar >= 0 and bar <= end_bar:
                is_high = self._is_swing_high(i)
                all_swings.append((bar, price, 1, is_high))

        # L2+ swings from completed_waves
        for level in self.levels[1:]:
            for (orig_bar, orig_price, end_bar_wave, end_price) in level.completed_waves:
                if end_bar_wave >= 0 and end_bar_wave <= end_bar:
                    # Determine if this is a HIGH or LOW based on price movement
                    is_high = end_price > orig_price
                    all_swings.append((end_bar_wave, end_price, level.level, is_high))

        # Sort by bar index
        all_swings.sort(key=lambda x: x[0])

        # For each swing, find the most recent child swing of opposite type
        for i, (bar, price, level, is_high) in enumerate(all_swings):
            if bar < 0 or bar > end_bar:
                continue

            # Find deepest child swing leading to this swing
            # Search ALL preceding bars (not just since last swing of any level)
            child_level = None
            child_bar = None
            child_price = None

            for child_level_num in range(level + 1, len(self.levels) + 1):
                # Child swing is opposite type: if parent HIGH, look for child LOW
                looking_for_low = is_high
                # Search from start (bar 0) to just before this swing
                child_swing = self._find_most_recent_child_swing(
                    child_level_num,
                    bar,
                    looking_for_low
                )
                if child_swing:
                    child_level = child_level_num
                    child_bar = child_swing[0]
                    child_price = child_swing[1]
                    # Keep searching for deeper levels

            # Only create label if there IS a child swing
            if child_level is not None and child_bar is not None and child_price is not None:
                bars_ago = bar - child_bar
                timestamp = self._candles[bar].timestamp

                self._swing_labels.append(SwingLabel(
                    bar=bar,
                    timestamp=timestamp,
                    price=price,
                    is_high=is_high,
                    child_level=child_level,
                    child_price=child_price,
                    bars_ago=bars_ago,
                ))

    def _find_most_recent_child_swing(
        self,
        level_num: int,
        before_bar: int,
        looking_for_low: bool
    ) -> Optional[tuple]:
        """
        Find the most recent child swing of the specified type before the given bar.

        Returns (bar, price) of the most recent matching swing, or None.
        """
        if level_num > len(self.levels):
            return None

        level = self.levels[level_num - 1]
        best_match = None

        for (orig_bar, orig_price, end_bar, end_price) in level.completed_waves:
            if end_bar >= before_bar:
                continue  # Must be before the parent swing

            # Check if this is the right type (HIGH or LOW)
            is_high = end_price > orig_price
            is_low = not is_high

            if looking_for_low and is_low:
                if best_match is None or end_bar > best_match[0]:
                    best_match = (end_bar, end_price)
            elif not looking_for_low and is_high:
                if best_match is None or end_bar > best_match[0]:
                    best_match = (end_bar, end_price)

        return best_match

    def _find_preceding_child_swing(
        self,
        level_num: int,
        after_bar: int,
        before_bar: int,
        looking_for_low: bool
    ) -> Optional[tuple]:
        """
        Find child level swing that occurred between after_bar and before_bar.

        Args:
            level_num: The child level to search (2, 3, 4, etc.)
            after_bar: Must be after this bar
            before_bar: Must be before or at this bar (parent extreme bar)
            looking_for_low: True if looking for child LOW, False for child HIGH

        Returns:
            Tuple (end_bar, end_price) if found, else None.
            If multiple found, returns the latest one (closest to before_bar).
        """
        level = self._get_level(level_num)
        if not level:
            return None

        best_match = None
        for (orig_bar, orig_price, end_bar, end_price) in level.completed_waves:
            # Check if this wave's endpoint falls in the range
            if end_bar > after_bar and end_bar <= before_bar:
                # Determine if this wave is a LOW or HIGH
                # Wave is a LOW if end_price < orig_price
                is_low = end_price < orig_price
                if is_low == looking_for_low:
                    # Take the latest one (closest to parent extreme)
                    if best_match is None or end_bar > best_match[0]:
                        best_match = (end_bar, end_price)
        return best_match

    def _get_recursive_child_points(
        self,
        parent_level: int,
        parent_bar: int,
        parent_is_high: bool,
        prev_bar: int
    ) -> list:
        """
        Recursively find child swing points leading up to a parent extreme.

        For a parent HIGH at bar X, finds L2 LOW, then L3 HIGH before that, etc.
        Returns points in order from deepest to shallowest (e.g., L4→L3→L2).

        Args:
            parent_level: The parent level (1, 2, 3, etc.)
            parent_bar: Bar index of parent extreme
            parent_is_high: True if parent made a HIGH, False if LOW
            prev_bar: Previous point's bar (search starts after this)

        Returns:
            List of (bar, price, level) tuples, deepest first.
        """
        child_level = parent_level + 1
        # Child swing is opposite type: if parent HIGH, look for child LOW
        looking_for_low = parent_is_high

        child_swing = self._find_preceding_child_swing(
            child_level,
            prev_bar,
            parent_bar,
            looking_for_low
        )

        if not child_swing:
            return []

        child_bar, child_price = child_swing
        child_is_high = not looking_for_low  # Opposite of what we searched for

        # Recursively get grandchild points
        grandchild_points = self._get_recursive_child_points(
            child_level,
            child_bar,
            child_is_high,
            prev_bar
        )

        # Return grandchildren first (deepest), then this child
        return grandchild_points + [(child_bar, child_price, child_level)]

    def _build_stitch_waves(self, end_bar: int) -> list[Wave]:
        """
        Build stitch waveform using recursive parent-child relationship.

        For each L1 swing:
        - Recursively find child swings (L2, L3, L4...) leading up to it
        - Insert all child points (deepest first), then L1 point

        Example: L1 HIGH at bar 50
          → finds L2 LOW at bar 45
            → finds L3 HIGH at bar 43
              → finds L4 LOW at bar 42
          Result: L4_low → L3_high → L2_low → L1_high

        Segments are colored by DESTINATION point's level.
        """
        from datetime import timedelta

        L1 = self.levels[0]

        # Build list of stitch points as (bar, price, level)
        stitch_points = []

        # Clear and populate annotations for debugging
        self._stitch_annotations = []

        for i in range(len(L1.swing_x)):
            bar = L1.swing_x[i]
            price = L1.swing_y[i]

            if bar > end_bar:
                continue

            # Determine if this L1 swing is HIGH or LOW
            is_high = self._is_swing_high(i)

            # Get previous bar for search range
            prev_bar = L1.swing_x[i - 1] if i > 0 else -1

            # Recursively find all child points leading to this L1 swing
            # Returns deepest first: [L4, L3, L2] for example
            child_points = self._get_recursive_child_points(
                parent_level=1,
                parent_bar=bar,
                parent_is_high=is_high,
                prev_bar=prev_bar
            )

            # Generate annotation for this L1 swing
            if bar >= 0 and bar < len(self._candles):
                timestamp = self._candles[bar].timestamp
                if child_points:
                    # Show immediate child (L2) info
                    # child_points is deepest first, so L2 is last
                    l2_point = child_points[-1]  # (bar, price, level)
                    l2_bar, l2_price, l2_level = l2_point
                    bars_ago = bar - l2_bar
                    child_type = "low" if is_high else "high"
                    text = f"[L{l2_level} {child_type}: {l2_price:.5f} - {bars_ago} bars]"
                else:
                    child_type = "low" if is_high else "high"
                    text = f"[No L2 {child_type} found]"

                self._stitch_annotations.append({
                    'bar': bar,
                    'timestamp': timestamp.isoformat(),
                    'price': price,
                    'level': 1,
                    'is_high': is_high,
                    'text': text,
                    'child_points': [(cp[0], cp[1], cp[2]) for cp in child_points],
                })

            # Insert child points (deepest first)
            stitch_points.extend(child_points)

            # Insert L1 point
            stitch_points.append((bar, price, 1))

        # Build waves from consecutive points
        waves = []
        for i in range(len(stitch_points) - 1):
            bar1, price1, level1 = stitch_points[i]
            bar2, price2, level2 = stitch_points[i + 1]

            # Direction from price movement
            direction = Direction.UP if price2 > price1 else Direction.DOWN

            # Get timestamps for bar indices
            if bar1 < 0:
                if len(self._candles) >= 2:
                    interval = self._candles[1].timestamp - self._candles[0].timestamp
                else:
                    interval = timedelta(minutes=10)
                t1 = self._candles[0].timestamp - interval
            else:
                t1 = self._candles[min(bar1, len(self._candles) - 1)].timestamp

            if bar2 < 0:
                if len(self._candles) >= 2:
                    interval = self._candles[1].timestamp - self._candles[0].timestamp
                else:
                    interval = timedelta(minutes=10)
                t2 = self._candles[0].timestamp - interval
            else:
                t2 = self._candles[min(bar2, len(self._candles) - 1)].timestamp

            # Color by DESTINATION level
            segment_level = level2

            wave = Wave(
                id=3000 + i,
                level=segment_level,
                direction=direction,
                start_time=t1,
                start_price=price1,
                end_time=t2,
                end_price=price2,
                parent_id=None,
                is_active=True,
                is_spline=False,
            )
            waves.append(wave)

        # Add close leg if needed
        if stitch_points and end_bar >= 0 and end_bar < len(self._candles):
            candle = self._candles[end_bar]
            last_bar, last_price, last_level = stitch_points[-1]

            # Only add close leg if close differs from last point
            if abs(candle.close - last_price) > 1e-10:
                if last_bar < 0:
                    if len(self._candles) >= 2:
                        interval = self._candles[1].timestamp - self._candles[0].timestamp
                    else:
                        interval = timedelta(minutes=10)
                    t1 = self._candles[0].timestamp - interval
                else:
                    t1 = self._candles[min(last_bar, len(self._candles) - 1)].timestamp

                if len(self._candles) >= 2:
                    interval = self._candles[1].timestamp - self._candles[0].timestamp
                else:
                    interval = timedelta(minutes=10)
                t2 = candle.timestamp + interval

                direction = Direction.UP if candle.close > last_price else Direction.DOWN

                close_wave = Wave(
                    id=3000 + len(stitch_points),
                    level=last_level + 1,
                    direction=direction,
                    start_time=t1,
                    start_price=last_price,
                    end_time=t2,
                    end_price=candle.close,
                    parent_id=None,
                    is_active=True,
                    is_spline=False,
                )
                waves.append(close_wave)

        return waves

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
    # STITCH MODE DISPLAY
    # Custom wave display logic for stitch mode
    # ================================================================

    def _build_stitch_display_waves(self, end_bar: int) -> list[Wave]:
        """
        Build custom display waves for stitch mode.

        Only shows permanent L1 legs - recorded at the moment of direction change.
        """
        from datetime import timedelta
        import sys

        waves: list[Wave] = []

        # Debug output
        print(f"\n[STITCH] Bar {end_bar}: {len(self._stitch_permanent_legs)} permanent legs", file=sys.stderr)

        # Only show permanent legs (recorded on direction change)
        for i, (start_bar, start_price, end_bar_leg, end_price, direction) in enumerate(self._stitch_permanent_legs):
            # Get timestamp for start point
            if start_bar < 0:
                if len(self._candles) >= 2:
                    interval = self._candles[1].timestamp - self._candles[0].timestamp
                else:
                    interval = timedelta(minutes=10)
                start_time = self._candles[0].timestamp - interval
            else:
                start_time = self._candles[min(start_bar, len(self._candles) - 1)].timestamp

            # Get timestamp for end point
            end_time = self._candles[min(end_bar_leg, len(self._candles) - 1)].timestamp

            wave_direction = Direction.UP if direction == 1 else Direction.DOWN

            wave = Wave(
                id=6000 + i,  # Stitch permanent leg IDs
                level=1,  # Yellow (L1)
                direction=wave_direction,
                start_time=start_time,
                start_price=start_price,
                end_time=end_time,
                end_price=end_price,
                parent_id=None,
                is_active=True,
                is_spline=False,  # SOLID line
            )
            waves.append(wave)
            print(f"  Leg {i}: bar {start_bar} ({start_price:.5f}) -> bar {end_bar_leg} ({end_price:.5f}) [{'UP' if direction == 1 else 'DOWN'}]", file=sys.stderr)

        return waves

    def _print_wave_state_debug(self, end_bar: int) -> None:
        """Print comprehensive wave state for debugging."""
        import sys

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[WAVE STATE] Bar {end_bar}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        # L1 State
        dir_str = "UP" if self.L1_Direction == 1 else "DOWN"
        print(f"L1_Direction: {dir_str}", file=sys.stderr)
        print(f"L1_High: {self.L1_High:.5f} @ bar {self.L1_High_bar}", file=sys.stderr)
        print(f"L1_Low:  {self.L1_Low:.5f} @ bar {self.L1_Low_bar}", file=sys.stderr)

        # L1 Swing Points
        print(f"\nL1 Swing Points ({len(self.L1_swing_x)} points):", file=sys.stderr)
        for i, (bar, price) in enumerate(zip(self.L1_swing_x, self.L1_swing_y)):
            label = "developing" if i == len(self.L1_swing_x) - 1 else f"swing {i}"
            print(f"  [{i}] bar={bar}, price={price:.5f} ({label})", file=sys.stderr)

        # L1 Spline Segments
        print(f"\nL1 Spline Segments ({len(self.L1_spline_segments)} segments):", file=sys.stderr)
        for i, (ob, op, eb, ep) in enumerate(self.L1_spline_segments):
            dir_s = "UP" if ep > op else "DOWN"
            print(f"  [{i}] bar {ob} ({op:.5f}) -> bar {eb} ({ep:.5f}) [{dir_s}]", file=sys.stderr)

        # L2+ Levels
        print(f"\nL2+ Levels ({len(self.levels)} total):", file=sys.stderr)
        for level in self.levels:
            if level.level == 1:
                continue  # Skip L1, already shown above
            dir_str = "UP" if level.direction == 1 else ("DOWN" if level.direction == -1 else "NEUTRAL")
            print(f"  L{level.level}: dir={dir_str}", file=sys.stderr)
            print(f"    High: {level.high:.5f} @ bar {level.high_bar}", file=sys.stderr)
            print(f"    Low:  {level.low:.5f} @ bar {level.low_bar}", file=sys.stderr)
            print(f"    Origin: bar {level.origin_bar}, price {level.origin_price:.5f}", file=sys.stderr)
            print(f"    Completed waves: {len(level.completed_waves)}", file=sys.stderr)
            for j, (start_bar, start_p, end_bar_w, end_p) in enumerate(level.completed_waves):
                print(f"      [{j}] bar {start_bar} ({start_p:.5f}) -> bar {end_bar_w} ({end_p:.5f})", file=sys.stderr)
            print(f"    Spline segments: {len(level.spline_segments)}", file=sys.stderr)

        print(f"{'='*60}\n", file=sys.stderr)

    def get_debug_state(self, end_bar: int) -> dict:
        """
        Return current wave state as a dictionary for UI debugging.
        """
        # Get current candle data
        current_candle = None
        if 0 <= end_bar < len(self._candles):
            c = self._candles[end_bar]
            current_candle = {
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
            }

        # Build levels data for all levels
        levels_data = []
        for level in self.levels:
            dir_str = "UP" if level.direction == 1 else ("DOWN" if level.direction == -1 else "NEUTRAL")
            level_data = {
                "level": level.level,
                "direction": dir_str,
                "high": level.high,
                "high_bar": level.high_bar,
                "low": level.low,
                "low_bar": level.low_bar,
                "origin_bar": level.origin_bar,
                "origin_price": level.origin_price,
                "completed_waves": [
                    {"start_bar": sb, "start_price": sp, "end_bar": eb, "end_price": ep}
                    for sb, sp, eb, ep in level.completed_waves
                ],
                "spline_segments": [
                    {"start_bar": sb, "start_price": sp, "end_bar": eb, "end_price": ep}
                    for sb, sp, eb, ep in level.spline_segments
                ],
            }
            # Add swing points for L1
            if level.level == 1:
                level_data["swing_points"] = [
                    {"bar": bar, "price": price}
                    for bar, price in zip(level.swing_x, level.swing_y)
                ]
            levels_data.append(level_data)

        return {
            "mode": self._mode,
            "end_bar": end_bar,
            "current_candle": current_candle,
            "levels": levels_data,
            "stitch_permanent_legs": [
                {"start_bar": sb, "start_price": sp, "end_bar": eb, "end_price": ep, "direction": "UP" if d == 1 else "DOWN"}
                for sb, sp, eb, ep, d in self._stitch_permanent_legs
            ],
            "prev_L1_Direction": "UP" if self._prev_L1_Direction == 1 else ("DOWN" if self._prev_L1_Direction == -1 else "NONE"),
            "num_waves_returned": len(self._waves) if hasattr(self, '_waves') else 0,
        }

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
