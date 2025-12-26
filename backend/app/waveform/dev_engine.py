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

        # L2 State Variables
        self.L1_pullback: bool = False  # True when L1 is not making new extreme
        self.L2_High: float = 0.0
        self.L2_Low: float = 0.0
        self.L2_High_bar: int = 0
        self.L2_Low_bar: int = 0
        self.L2_origin_bar: int = 0      # Bar where current L2 started
        self.L2_origin_price: float = 0.0  # Price where current L2 started

        # L2 Completed Waves - each is (origin_bar, origin_price, end_bar, end_price)
        # Only stores waves where actual retracement occurred
        self.L2_completed_waves: list[tuple[int, float, int, float]] = []

        # L2 Spline segments
        self.L2_spline_segments: list[tuple[int, float, int, float]] = []

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

        # Reset L2 state
        self.L1_pullback = False
        self.L2_High = 0.0
        self.L2_Low = 0.0
        self.L2_High_bar = 0
        self.L2_Low_bar = 0
        self.L2_origin_bar = 0
        self.L2_origin_price = 0.0
        self.L2_completed_waves = []
        self.L2_spline_segments = []

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

        # Track count of historical L2s before adding developing leg
        historical_l2_count = len(self.L2_completed_waves)
        self._add_developing_leg_L2(end_bar)

        # Convert swing points to Wave objects for display
        waves = self._build_waves()

        # Add L2 waves (pass historical count to separate developing from historical)
        l2_waves = self._build_L2_waves(historical_l2_count)
        waves.extend(l2_waves)

        # In spline mode, add spline waves (intermediate developing legs)
        if self._mode == "spline":
            spline_waves = self._build_spline_waves()
            waves.extend(spline_waves)
            l2_spline_waves = self._build_L2_spline_waves()
            waves.extend(l2_spline_waves)

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
        """
        # Check if there was retracement in the current L2
        # UP direction: retracement means L2_Low < L2_High (pulled back from high)
        # DOWN direction: retracement means L2_High > L2_Low (rallied from low)
        had_retracement = (self.L2_Low < self.L2_High)

        if had_retracement:
            # Store completed L2 wave: origin → retracement endpoint
            if self.L1_Direction == 1:
                # Was UP, L2 tracked pullback lows
                self.L2_completed_waves.append((
                    self.L2_origin_bar, self.L2_origin_price,
                    self.L2_Low_bar, self.L2_Low
                ))
            else:
                # Was DOWN, L2 tracked pullback highs
                self.L2_completed_waves.append((
                    self.L2_origin_bar, self.L2_origin_price,
                    self.L2_High_bar, self.L2_High
                ))

        # Reset L2 to new origin
        self.L2_High = new_l2_origin_price
        self.L2_Low = new_l2_origin_price
        self.L2_High_bar = bar_idx
        self.L2_Low_bar = bar_idx
        self.L2_origin_bar = bar_idx
        self.L2_origin_price = new_l2_origin_price

    def _update_L2_pullback(self, candle: Candle, bar_idx: int) -> None:
        """
        Update L2 during pullback (when L1 is not making a new extreme).

        UP direction: Track lower lows (L2_Low)
        DOWN direction: Track higher highs (L2_High)
        """
        if self.L1_Direction == 1:
            # UP direction pullback: track lower lows
            # Condition: High <= L2_High AND Low < L2_Low
            if candle.high <= self.L2_High and candle.low < self.L2_Low:
                # Track spline segment for L2 (from origin to new low)
                if self._mode == "spline":
                    self.L2_spline_segments.append((
                        self.L2_origin_bar, self.L2_origin_price,
                        bar_idx, candle.low
                    ))
                self.L2_Low = candle.low
                self.L2_Low_bar = bar_idx

        elif self.L1_Direction == -1:
            # DOWN direction pullback: track higher highs
            # Condition: Low >= L2_Low AND High > L2_High
            if candle.low >= self.L2_Low and candle.high > self.L2_High:
                # Track spline segment for L2 (from origin to new high)
                if self._mode == "spline":
                    self.L2_spline_segments.append((
                        self.L2_origin_bar, self.L2_origin_price,
                        bar_idx, candle.high
                    ))
                self.L2_High = candle.high
                self.L2_High_bar = bar_idx

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

    def _add_developing_leg_L2(self, current_bar: int) -> None:
        """
        Add L2 developing leg from L2 origin to current L2 extreme.

        If L1_Direction is UP: L2 retracement goes to L2_Low
        If L1_Direction is DOWN: L2 retracement goes to L2_High
        """
        # Only add developing leg if there was retracement
        if self.L2_Low < self.L2_High:
            if self.L1_Direction == 1:
                # UP direction: L2 tracks pullback to L2_Low
                self.L2_completed_waves.append((
                    self.L2_origin_bar, self.L2_origin_price,
                    self.L2_Low_bar, self.L2_Low
                ))
            elif self.L1_Direction == -1:
                # DOWN direction: L2 tracks pullback to L2_High
                self.L2_completed_waves.append((
                    self.L2_origin_bar, self.L2_origin_price,
                    self.L2_High_bar, self.L2_High
                ))

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

    def _build_L2_waves(self, historical_l2_count: int) -> list[Wave]:
        """
        Convert L2 completed waves to Wave objects for display.

        L2 waves show the retracement from each L1 extreme.
        Each entry in L2_completed_waves is (origin_bar, origin_price, end_bar, end_price).

        Args:
            historical_l2_count: Number of L2s that were completed before adding developing leg.
                                 Entries after this index are "developing" L2s.

        Mode behavior:
        - Complete mode: Only show the developing L2 (if exists). Never show historical L2s.
        - Spline mode: Show ALL historical L2s as dotted + developing L2 as solid.
        """
        from datetime import timedelta

        waves = []

        if len(self.L2_completed_waves) == 0:
            return waves

        # Separate historical L2s from developing L2
        historical_l2s = self.L2_completed_waves[:historical_l2_count]
        developing_l2s = self.L2_completed_waves[historical_l2_count:]

        if self._mode == "complete":
            # Complete mode: Only show developing L2 (solid), no historical
            waves_to_process = [(i + historical_l2_count, entry, False)
                                for i, entry in enumerate(developing_l2s)]
        else:
            # Spline mode: All historical as dotted + developing as solid
            waves_to_process = [(i, entry, True) for i, entry in enumerate(historical_l2s)]
            waves_to_process += [(i + historical_l2_count, entry, False)
                                 for i, entry in enumerate(developing_l2s)]

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
                id=500 + actual_idx,  # Offset ID to avoid conflicts with L1
                level=2,  # L2 level = cyan color
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

    def _build_L2_spline_waves(self) -> list[Wave]:
        """
        Build L2 spline waves - lines from L2 origin to each intermediate retracement.

        These show the progression of the L2 retracement as L2_Low/L2_High updates.
        """
        from datetime import timedelta

        waves = []

        if len(self.L2_spline_segments) == 0:
            return waves

        # Create a wave for each spline segment
        for i, (origin_bar, origin_price, end_bar, end_price) in enumerate(self.L2_spline_segments):
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
                id=1500 + i,  # Offset ID to avoid conflicts with L1 splines (1000+)
                level=2,  # L2 level = cyan color
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
