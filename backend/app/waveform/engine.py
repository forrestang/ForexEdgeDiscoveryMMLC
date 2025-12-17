from datetime import datetime
from typing import Optional
import polars as pl

from app.waveform.wave import Wave, Candle, Direction


class WaveformEngine:
    """
    Recursive Session Waveform Algorithm Engine.

    Implements the exact specification:
    - Spawn: ANY movement against current wave creates a child wave
    - Extension: Movement in wave direction that makes new extreme extends the wave
    - Erasure: Breaking a wave's START POINT erases it and resumes parent

    Uses path simulation inside OHLC bars:
    - Bullish bar: Open -> Low -> High
    - Bearish bar: Open -> High -> Low
    """

    def __init__(self):
        self.waves: list[Wave] = []
        self.wave_id_counter: int = 0
        self._wave_stack: list[Wave] = []  # Stack of active waves (parent chain)

    def reset(self):
        """Reset engine state for new session processing."""
        self.waves = []
        self.wave_id_counter = 0
        self._wave_stack = []

    def process_session(self, df: pl.DataFrame) -> list[Wave]:
        """
        Process entire session and return wave hierarchy.
        Returns only non-erased waves.
        """
        self.reset()

        candles = self._df_to_candles(df)
        if not candles:
            return []

        # Find first non-neutral bar for initialization
        first_idx = 0
        first = candles[0]

        # Handle neutral first bar - find next directional bar
        if first.close == first.open:
            for i in range(1, len(candles)):
                if candles[i].close != candles[i].open:
                    # Use this bar to determine initial direction
                    if candles[i].is_bullish:
                        # Initialize UP
                        self._initialize_session(first, Direction.UP)
                    else:
                        # Initialize DOWN
                        self._initialize_session(first, Direction.DOWN)
                    first_idx = 1
                    break
            else:
                # All bars neutral - just pick UP
                self._initialize_session(first, Direction.UP)
                first_idx = 1
        else:
            # Normal initialization based on first bar
            if first.is_bullish:
                self._initialize_session(first, Direction.UP)
            else:
                self._initialize_session(first, Direction.DOWN)
            first_idx = 1

        # Process remaining bars
        for i in range(first_idx, len(candles)):
            self._process_bar(candles[i])

        # Sync the deepest active wave to the last bar's close price
        if candles and self._wave_stack:
            last_candle = candles[-1]
            self._sync_current_wave_to_close(last_candle.close, last_candle.timestamp)

        # L1 waves: return ALL (erased or not) - forms session backbone
        # L2+ waves: return only ACTIVE (erased waves disappear)
        return [w for w in self.waves if w.level == 1 or w.is_active]

    def _initialize_session(self, candle: Candle, direction: Direction):
        """Initialize the session with L1 wave."""
        if direction == Direction.UP:
            # Simulate: Open -> Low -> High
            # L1 starts at Low, ends at High
            self._create_wave(
                level=1,
                direction=Direction.UP,
                start_time=candle.timestamp,
                start_price=candle.low,
                end_time=candle.timestamp,
                end_price=candle.high,
                parent=None,
            )
        else:
            # Simulate: Open -> High -> Low
            # L1 starts at High, ends at Low
            self._create_wave(
                level=1,
                direction=Direction.DOWN,
                start_time=candle.timestamp,
                start_price=candle.high,
                end_time=candle.timestamp,
                end_price=candle.low,
                parent=None,
            )

    def _df_to_candles(self, df: pl.DataFrame) -> list[Candle]:
        """Convert DataFrame to list of Candle objects."""
        candles = []
        for row in df.iter_rows(named=True):
            candles.append(
                Candle(
                    timestamp=row["timestamp"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row.get("volume", 0),
                )
            )
        return candles

    def _process_bar(self, candle: Candle):
        """
        Process a single OHLC bar by simulating the tick path inside it.
        """
        if not self._wave_stack:
            return

        h, l = candle.high, candle.low
        ts = candle.timestamp

        # Determine tick simulation order based on bar type
        if candle.is_bullish or candle.close == candle.open:
            # Bullish or neutral: Open -> Low -> High
            # First check if Low creates any action, then High
            self._process_tick(l, ts)
            self._process_tick(h, ts)
        else:
            # Bearish: Open -> High -> Low
            self._process_tick(h, ts)
            self._process_tick(l, ts)

    def _process_tick(self, price: float, ts: datetime):
        """
        Process a single simulated tick price.
        Implements Spawn, Extension, and Erasure rules.
        """
        if not self._wave_stack:
            return

        current = self._wave_stack[-1]

        if current.direction == Direction.UP:
            if price > current.end_price:
                # Extension: new high
                current.end_price = price
                current.end_time = ts
                # Propagate this new high to L1 ancestor (if same direction)
                self._propagate_extreme_to_l1(current)
                # Check if this extension breaks any parent's start point
                self._check_parent_erasure(price, ts)
            elif price < current.end_price:
                # Movement against - check for erasure or spawn
                if price < current.start_price:
                    # Erasure: broke start point (must exceed, not just touch)
                    self._erase_wave(current, price, ts)
                else:
                    # Spawn: any downward movement creates child
                    self._spawn_child(Direction.DOWN, current.end_price, current.end_time, price, ts)

        else:  # DOWN wave
            if price < current.end_price:
                # Extension: new low
                current.end_price = price
                current.end_time = ts
                # Propagate this new low to L1 ancestor (if same direction)
                self._propagate_extreme_to_l1(current)
                # Check if this extension breaks any parent's start point
                self._check_parent_erasure(price, ts)
            elif price > current.end_price:
                # Movement against - check for erasure or spawn
                if price > current.start_price:
                    # Erasure: broke start point (must exceed, not just touch)
                    self._erase_wave(current, price, ts)
                else:
                    # Spawn: any upward movement creates child
                    self._spawn_child(Direction.UP, current.end_price, current.end_time, price, ts)

    def _check_parent_erasure(self, price: float, ts: datetime):
        """
        After an extension, check if the price breaks any parent wave's start point.
        If so, erase from that parent down.
        """
        # Walk up the stack looking for a parent whose start is broken
        erase_from_idx = None
        for i in range(len(self._wave_stack) - 2, -1, -1):  # Skip current, go backwards
            parent = self._wave_stack[i]
            if parent.direction == Direction.UP and price < parent.start_price:
                erase_from_idx = i
            elif parent.direction == Direction.DOWN and price > parent.start_price:
                erase_from_idx = i

        if erase_from_idx is not None:
            # Erase from this parent onwards
            wave_to_erase = self._wave_stack[erase_from_idx]
            # Mark all children as erased
            for i in range(len(self._wave_stack) - 1, erase_from_idx, -1):
                self._wave_stack[i].is_active = False
            # Truncate stack
            self._wave_stack = self._wave_stack[:erase_from_idx]
            # Now erase the parent itself
            self._erase_wave(wave_to_erase, price, ts)

    def _propagate_extreme_to_l1(self, wave: Wave):
        """
        Propagate a wave's extreme to the L1 ancestor currently in the stack.
        This ensures the L1 backbone captures all highs/lows from active child waves.

        IMPORTANT: Only update L1 if it's in the current stack (still active).
        Updating historical L1 waves breaks the zigzag connectivity.
        """
        if wave.level == 1:
            return  # Already L1, nothing to propagate

        # Find L1 in the CURRENT STACK (not historical waves)
        for ancestor in self._wave_stack:
            if ancestor.level == 1:
                if wave.direction == Direction.UP and ancestor.direction == Direction.UP:
                    # Both UP - propagate high
                    if wave.end_price > ancestor.end_price:
                        ancestor.end_price = wave.end_price
                        ancestor.end_time = wave.end_time
                elif wave.direction == Direction.DOWN and ancestor.direction == Direction.DOWN:
                    # Both DOWN - propagate low
                    if wave.end_price < ancestor.end_price:
                        ancestor.end_price = wave.end_price
                        ancestor.end_time = wave.end_time
                break

    def _erase_wave(self, wave: Wave, price: float, ts: datetime):
        """
        Erase a wave and resume parent.
        The erased wave stays in self.waves but is marked is_active=False.
        """
        wave.is_active = False

        # Remove from stack
        if self._wave_stack and self._wave_stack[-1] == wave:
            self._wave_stack.pop()

        if self._wave_stack:
            # Resume parent - only update end if new price is more favorable
            # This preserves propagated extremes from erased child waves
            parent = self._wave_stack[-1]
            if parent.direction == Direction.UP:
                # For UP wave, only update if price is higher (new high)
                if price > parent.end_price:
                    parent.end_price = price
                    parent.end_time = ts
                    # Propagate parent's new high to L1 ancestor
                    self._propagate_extreme_to_l1(parent)
                    # Check if this extension breaks any grandparent's start
                    self._check_parent_erasure(price, ts)
            else:
                # For DOWN wave, only update if price is lower (new low)
                if price < parent.end_price:
                    parent.end_price = price
                    parent.end_time = ts
                    # Propagate parent's new low to L1 ancestor
                    self._propagate_extreme_to_l1(parent)
                    # Check if this extension breaks any grandparent's start
                    self._check_parent_erasure(price, ts)

            # Also propagate erased wave's extreme to the most recent matching L1
            self._propagate_extreme_to_l1(wave)

            # Recursively check if parent should also be erased
            if parent.direction == Direction.UP and price < parent.start_price:
                self._erase_wave(parent, price, ts)
            elif parent.direction == Direction.DOWN and price > parent.start_price:
                self._erase_wave(parent, price, ts)
        else:
            # No parent - create new L1 in opposite direction
            # New L1 starts from the erased wave's END (the swing point),
            # and ends at current price
            new_direction = Direction.DOWN if wave.direction == Direction.UP else Direction.UP
            self._create_wave(
                level=1,
                direction=new_direction,
                start_time=wave.end_time,    # Start from the swing point
                start_price=wave.end_price,  # Start from the swing price
                end_time=ts,
                end_price=price,
                parent=None,
            )

    def _spawn_child(self, direction: Direction, start_price: float, start_time: datetime,
                     end_price: float, end_time: datetime):
        """Spawn a child wave in the opposite direction."""
        if not self._wave_stack:
            return

        parent = self._wave_stack[-1]
        self._create_wave(
            level=parent.level + 1,
            direction=direction,
            start_time=start_time,
            start_price=start_price,
            end_time=end_time,
            end_price=end_price,
            parent=parent,
        )
        # Check if the spawn price breaks any grandparent's start point
        # This handles cases where spawning at a new extreme should erase ancestors
        self._check_parent_erasure(end_price, end_time)

    def _create_wave(
        self,
        level: int,
        direction: Direction,
        start_time: datetime,
        start_price: float,
        end_time: datetime,
        end_price: float,
        parent: Optional[Wave],
    ) -> Wave:
        """Create a new wave and push to stack."""
        self.wave_id_counter += 1
        wave = Wave(
            id=self.wave_id_counter,
            level=level,
            direction=direction,
            start_time=start_time,
            start_price=start_price,
            end_time=end_time,
            end_price=end_price,
            parent_id=parent.id if parent else None,
            is_active=True,
        )
        self.waves.append(wave)
        self._wave_stack.append(wave)
        return wave

    def _sync_current_wave_to_close(self, close_price: float, ts: datetime):
        """
        Sync the deepest active wave's endpoint to the bar's close price.
        This represents the current/live price position.
        """
        if not self._wave_stack:
            return

        current = self._wave_stack[-1]
        current.end_price = close_price
        current.end_time = ts
