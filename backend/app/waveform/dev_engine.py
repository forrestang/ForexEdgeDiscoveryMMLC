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

        # L1 State Variables
        self.L1_Direction: int = 0  # -1=DOWN, 0=NEUTRAL, +1=UP
        self.L1_High: float = 0.0
        self.L1_Low: float = 0.0
        self.L1_High_bar: int = 0  # Bar index where L1_High occurred
        self.L1_Low_bar: int = 0   # Bar index where L1_Low occurred

        # L1 Swing Point Arrays (x=bar index, y=price)
        self.L1_swing_x: list[int] = []
        self.L1_swing_y: list[float] = []

        # Spline mode: track all intermediate developing leg segments
        # Each segment is (origin_bar, origin_price, end_bar, end_price)
        self.L1_spline_segments: list[tuple[int, float, int, float]] = []

    def _next_wave_id(self) -> int:
        """Get next unique wave ID."""
        self._wave_id_counter += 1
        return self._wave_id_counter

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

        # Reset L1 state
        self.L1_Direction = 0
        self.L1_High = 0.0
        self.L1_Low = 0.0
        self.L1_High_bar = 0
        self.L1_Low_bar = 0
        self.L1_swing_x = []
        self.L1_swing_y = []
        self.L1_spline_segments = []

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

        # Convert swing points to Wave objects for display
        waves = self._build_waves()

        # In spline mode, add spline waves (intermediate developing legs)
        if self._mode == "spline":
            spline_waves = self._build_spline_waves()
            waves.extend(spline_waves)

        return waves

    def _initialize_first_bar(self, candle: Candle, bar_idx: int) -> None:
        """
        Rule 1: Initialize L1 state on the first bar.

        Step 1: Anchor point at index -1, price = open
        Step 2: Determine direction from close/open relationship
        Step 3: Push swing point (low if UP, high if DOWN)
        """
        # Step 1: First swing point at index -1 (anchor to the left of bar 0)
        self.L1_swing_x.append(-1)
        self.L1_swing_y.append(candle.open)

        # Step 2: Determine L1_Direction from close/open relationship
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

    def _process_bar(self, candle: Candle, bar_idx: int) -> None:
        """
        Process subsequent bars (after bar 0).
        """
        if self.L1_Direction == 1:
            # UP direction rules
            if candle.high > self.L1_High and candle.low >= self.L1_Low:
                # Case 1: New high, no new low - continuing UP
                # Track spline segment (from last swing to new high) for spline mode
                if self._mode == "spline" and len(self.L1_swing_x) > 0:
                    origin_bar = self.L1_swing_x[-1]
                    origin_price = self.L1_swing_y[-1]
                    self.L1_spline_segments.append((origin_bar, origin_price, bar_idx, candle.high))
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
                        self.L1_Low = candle.low
                        self.L1_Low_bar = bar_idx
                        self.L1_High = candle.high
                        self.L1_High_bar = bar_idx
                    else:
                        # Below midpoint: treat as bearish (high first)
                        self.L1_swing_x.append(bar_idx)
                        self.L1_swing_y.append(candle.high)
                        self.L1_High = candle.high
                        self.L1_High_bar = bar_idx
                        self.L1_Low = candle.low
                        self.L1_Low_bar = bar_idx
                        # Direction changes to DOWN
                        self.L1_Direction = -1

        elif self.L1_Direction == -1:
            # DOWN direction rules (mirrored from UP)
            if candle.low < self.L1_Low and candle.high <= self.L1_High:
                # Case 1: New low, no new high - continuing DOWN
                # Track spline segment (from last swing to new low) for spline mode
                if self._mode == "spline" and len(self.L1_swing_x) > 0:
                    origin_bar = self.L1_swing_x[-1]
                    origin_price = self.L1_swing_y[-1]
                    self.L1_spline_segments.append((origin_bar, origin_price, bar_idx, candle.low))
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
                        self.L1_High = candle.high
                        self.L1_High_bar = bar_idx
                        self.L1_Low = candle.low
                        self.L1_Low_bar = bar_idx
                        # Stay DOWN

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
                id=1000 + i,  # Offset ID to avoid conflicts
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
