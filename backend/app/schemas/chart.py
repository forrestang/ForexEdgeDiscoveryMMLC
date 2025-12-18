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


class ChartResponse(BaseModel):
    pair: str
    timeframe: str
    date: str
    session: str
    candles: list[CandleData]
    waveform: list[WaveData]
    debug: Optional[list[str]] = None
