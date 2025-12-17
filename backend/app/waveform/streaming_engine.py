"""
Streaming Waveform Engine with bar-by-bar state capture.

This wraps the core WaveformEngine to enable incremental processing
and state snapshot capture for the Edge Finder system.
"""

from datetime import datetime
from typing import Optional
import polars as pl

from app.waveform.wave import Wave, Candle, Direction
from app.waveform.state_snapshot import WaveSnapshot, StackSnapshot


class StreamingWaveformEngine:
    """
    Waveform engine that captures state snapshots at every bar.

    This enables the Edge Finder to capture the wave stack state
    at each point in time for matrix serialization and ML training.
    """

    def __init__(self):
        # Core state (mirrors WaveformEngine)
        self.waves: list[Wave] = []
        self.wave_id_counter: int = 0
        self._wave_stack: list[Wave] = []

        # Streaming state
        self._candles: list[Candle] = []
        self._snapshots: list[StackSnapshot] = []
        self._bar_index: int = 0
        self._initialized: bool = False

        # Wave counts per level (for leg_count feature)
        self._level_counts: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        # Track wave start bar indices
        self._wave_start_bars: dict[int, int] = {}  # wave_id -> start_bar_index

    def reset(self):
        """Reset engine state for new session processing."""
        self.waves = []
        self.wave_id_counter = 0
        self._wave_stack = []
        self._candles = []
        self._snapshots = []
        self._bar_index = 0
        self._initialized = False
        self._level_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self._wave_start_bars = {}

    def process_session_with_snapshots(
        self, df: pl.DataFrame
    ) -> tuple[list[Wave], list[StackSnapshot]]:
        """
        Process entire session and return waves + snapshots at every bar.

        Returns:
            tuple: (waves, snapshots) where:
                - waves: List of Wave objects (same as original engine)
                - snapshots: List of StackSnapshot, one per bar
        """
        self.reset()

        candles = self._df_to_candles(df)
        if not candles:
            return [], []

        self._candles = candles

        # Find first non-neutral bar for initialization
        first = candles[0]

        # Handle neutral first bar - find next directional bar
        if first.close == first.open:
            init_direction = Direction.UP  # default
            for i in range(1, len(candles)):
                if candles[i].close != candles[i].open:
                    init_direction = Direction.UP if candles[i].is_bullish else Direction.DOWN
                    break
            self._initialize_session(first, init_direction)
        else:
            init_direction = Direction.UP if first.is_bullish else Direction.DOWN
            self._initialize_session(first, init_direction)

        # Capture snapshot after first bar
        self._capture_snapshot(first)
        self._bar_index = 1

        # Process remaining bars
        for i in range(1, len(candles)):
            self._process_bar(candles[i])
            self._capture_snapshot(candles[i])
            self._bar_index += 1

        # Sync deepest wave to last close
        if candles and self._wave_stack:
            last_candle = candles[-1]
            self._sync_current_wave_to_close(last_candle.close, last_candle.timestamp)

        # Return waves (L1 all + L2+ active only) and all snapshots
        waves = [w for w in self.waves if w.level == 1 or w.is_active]
        return waves, self._snapshots

    def _capture_snapshot(self, candle: Candle) -> StackSnapshot:
        """Capture current wave stack state as a snapshot."""
        wave_snapshots = []

        for wave in self._wave_stack:
            # Calculate duration in bars
            start_bar = self._wave_start_bars.get(wave.id, 0)
            duration = self._bar_index - start_bar + 1

            # Direction as +1/-1
            direction = 1 if wave.direction == Direction.UP else -1

            # Amplitude (signed: positive for UP, negative for DOWN)
            amplitude = wave.end_price - wave.start_price

            wave_snapshots.append(WaveSnapshot(
                level=wave.level,
                direction=direction,
                amplitude=amplitude,
                duration_bars=duration,
                start_bar_index=start_bar,
            ))

        snapshot = StackSnapshot(
            bar_index=self._bar_index,
            timestamp=candle.timestamp,
            close_price=candle.close,
            waves=wave_snapshots,
            l1_count=self._level_counts[1],
            l2_count=self._level_counts[2],
            l3_count=self._level_counts[3],
            l4_count=self._level_counts[4],
            l5_count=self._level_counts[5],
        )

        self._snapshots.append(snapshot)
        return snapshot

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

    def _initialize_session(self, candle: Candle, direction: Direction):
        """Initialize the session with L1 wave."""
        if direction == Direction.UP:
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
            self._create_wave(
                level=1,
                direction=Direction.DOWN,
                start_time=candle.timestamp,
                start_price=candle.high,
                end_time=candle.timestamp,
                end_price=candle.low,
                parent=None,
            )
        self._initialized = True

    def _process_bar(self, candle: Candle):
        """Process a single OHLC bar by simulating the tick path inside it."""
        if not self._wave_stack:
            return

        h, l = candle.high, candle.low
        ts = candle.timestamp

        # Determine tick simulation order based on bar type
        if candle.is_bullish or candle.close == candle.open:
            # Bullish or neutral: Low first, then High
            self._process_tick(l, ts)
            self._process_tick(h, ts)
        else:
            # Bearish: High first, then Low
            self._process_tick(h, ts)
            self._process_tick(l, ts)

    def _process_tick(self, price: float, ts: datetime):
        """Process a single simulated tick price."""
        if not self._wave_stack:
            return

        current = self._wave_stack[-1]

        if current.direction == Direction.UP:
            if price > current.end_price:
                # Extension
                current.end_price = price
                current.end_time = ts
                self._propagate_extreme_to_l1(current)
                self._check_parent_erasure(price, ts)
            elif price < current.end_price:
                if price < current.start_price:
                    # Erasure
                    self._erase_wave(current, price, ts)
                else:
                    # Spawn child
                    self._spawn_child(Direction.DOWN, current.end_price, current.end_time, price, ts)
        else:  # DOWN
            if price < current.end_price:
                # Extension
                current.end_price = price
                current.end_time = ts
                self._propagate_extreme_to_l1(current)
                self._check_parent_erasure(price, ts)
            elif price > current.end_price:
                if price > current.start_price:
                    # Erasure
                    self._erase_wave(current, price, ts)
                else:
                    # Spawn child
                    self._spawn_child(Direction.UP, current.end_price, current.end_time, price, ts)

    def _check_parent_erasure(self, price: float, ts: datetime):
        """Check if price breaks any parent wave's start point."""
        erase_from_idx = None
        for i in range(len(self._wave_stack) - 2, -1, -1):
            parent = self._wave_stack[i]
            if parent.direction == Direction.UP and price < parent.start_price:
                erase_from_idx = i
            elif parent.direction == Direction.DOWN and price > parent.start_price:
                erase_from_idx = i

        if erase_from_idx is not None:
            wave_to_erase = self._wave_stack[erase_from_idx]
            for i in range(len(self._wave_stack) - 1, erase_from_idx, -1):
                self._wave_stack[i].is_active = False
            self._wave_stack = self._wave_stack[:erase_from_idx]
            self._erase_wave(wave_to_erase, price, ts)

    def _propagate_extreme_to_l1(self, wave: Wave):
        """Propagate wave's extreme to L1 ancestor in current stack."""
        if wave.level == 1:
            return

        for ancestor in self._wave_stack:
            if ancestor.level == 1:
                if wave.direction == Direction.UP and ancestor.direction == Direction.UP:
                    if wave.end_price > ancestor.end_price:
                        ancestor.end_price = wave.end_price
                        ancestor.end_time = wave.end_time
                elif wave.direction == Direction.DOWN and ancestor.direction == Direction.DOWN:
                    if wave.end_price < ancestor.end_price:
                        ancestor.end_price = wave.end_price
                        ancestor.end_time = wave.end_time
                break

    def _erase_wave(self, wave: Wave, price: float, ts: datetime):
        """Erase a wave and resume parent."""
        wave.is_active = False

        if self._wave_stack and self._wave_stack[-1] == wave:
            self._wave_stack.pop()

        if self._wave_stack:
            parent = self._wave_stack[-1]
            if parent.direction == Direction.UP:
                if price > parent.end_price:
                    parent.end_price = price
                    parent.end_time = ts
                    self._propagate_extreme_to_l1(parent)
                    self._check_parent_erasure(price, ts)
            else:
                if price < parent.end_price:
                    parent.end_price = price
                    parent.end_time = ts
                    self._propagate_extreme_to_l1(parent)
                    self._check_parent_erasure(price, ts)

            self._propagate_extreme_to_l1(wave)

            if parent.direction == Direction.UP and price < parent.start_price:
                self._erase_wave(parent, price, ts)
            elif parent.direction == Direction.DOWN and price > parent.start_price:
                self._erase_wave(parent, price, ts)
        else:
            # No parent - create new L1
            new_direction = Direction.DOWN if wave.direction == Direction.UP else Direction.UP
            self._create_wave(
                level=1,
                direction=new_direction,
                start_time=wave.end_time,
                start_price=wave.end_price,
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

        # Track wave start bar for duration calculation
        self._wave_start_bars[wave.id] = self._bar_index

        # Increment level count (for leg_count feature)
        if level in self._level_counts:
            self._level_counts[level] += 1

        return wave

    def _sync_current_wave_to_close(self, close_price: float, ts: datetime):
        """Sync deepest active wave's endpoint to bar's close price."""
        if not self._wave_stack:
            return
        current = self._wave_stack[-1]
        current.end_price = close_price
        current.end_time = ts
