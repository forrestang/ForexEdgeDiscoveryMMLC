from pydantic import BaseModel
from typing import Optional
from datetime import date


class InstrumentInfo(BaseModel):
    pair: str
    timeframes: list[str]
    start_date: date
    end_date: date
    file_count: int


class InstrumentsResponse(BaseModel):
    instruments: list[InstrumentInfo]


class InstrumentMetadata(BaseModel):
    pair: str
    available_timeframes: list[str]
    start_date: date
    end_date: date
    total_bars: int
