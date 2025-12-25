from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Literal


class CandleData(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class WaveData(BaseModel):
    id: int
    level: int
    direction: Literal["up", "down"]
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    color: str
    parent_id: Optional[int] = None
    is_spline: bool = False  # True for intermediate developing leg lines


class WaveSnapshotData(BaseModel):
    """Snapshot of a single wave's state at a specific bar."""
    level: int              # 1-5 (L1 through L5)
    direction: int          # +1 (UP) or -1 (DOWN)
    amplitude: float        # end_price - start_price (signed)
    duration_bars: int      # Number of bars since wave started
    start_bar_index: int    # Bar index when this wave started


class StackSnapshotData(BaseModel):
    """Complete MMLC state at a specific bar for debugging."""
    bar_index: int
    timestamp: datetime
    close_price: float
    waves: list[WaveSnapshotData]  # Active waves (L1 first, deepest last)
    l1_count: int  # Cumulative L1 leg count
    l2_count: int  # Cumulative L2 leg count
    l3_count: int  # Cumulative L3 leg count
    l4_count: int  # Cumulative L4 leg count
    l5_count: int  # Cumulative L5 leg count


class ChartResponse(BaseModel):
    pair: str
    timeframe: str
    date: str
    session: str
    candles: list[CandleData]
    waveform: list[WaveData]
    debug: Optional[list[str]] = None
    snapshot: Optional[StackSnapshotData] = None  # MMLC state at bar_index (if provided)
