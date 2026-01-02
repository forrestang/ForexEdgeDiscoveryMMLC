"""
MMLC Development API Endpoints

Provides endpoints for the MMLC development sandbox page.
"""

from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import Optional
from datetime import date
from pydantic import BaseModel

from app.config import settings, SessionType, TimeframeType
from app.schemas.chart import CandleData, WaveData
from app.core.cache_manager import load_from_cache
from app.core.session_filter import filter_by_date_and_session
from app.core.upsampler import upsample_ohlc
from app.waveform.dev_engine import MMLCDevEngine
from app.waveform.wave import Candle

router = APIRouter()


class DevSessionResponse(BaseModel):
    """Response for loading a session."""
    pair: str
    date: str
    session: str
    timeframe: str
    candles: list[CandleData]
    total_bars: int


class StitchAnnotation(BaseModel):
    """Debug annotation for stitch mode."""
    bar: int
    timestamp: str
    price: float
    level: int
    is_high: bool
    text: str  # e.g., "[1.13045 - 3 bars]"


class SwingLabelData(BaseModel):
    """Swing label for chart annotation."""
    bar: int
    timestamp: str
    price: float
    is_high: bool
    child_level: int  # e.g., 2 for L2
    child_price: float  # Price of the child swing
    bars_ago: int  # Bars from child swing to this swing


class DebugState(BaseModel):
    """Debug state for wave visualization."""
    mode: str
    end_bar: int
    current_candle: Optional[dict]
    levels: list[dict]  # All wave levels (L1, L2, L3, etc.) with all their properties
    stitch_permanent_legs: list[dict]
    prev_L1_Direction: str
    num_waves_returned: int


class DevRunResponse(BaseModel):
    """Response for running the dev engine."""
    waves: list[WaveData]
    start_bar: int
    end_bar: int
    bars_processed: int
    annotations: list[StitchAnnotation] = []
    swing_labels: list[SwingLabelData] = []
    debug_state: Optional[DebugState] = None


@router.get("/session/{pair}", response_model=DevSessionResponse)
async def load_session(
    pair: str,
    date: date = Query(..., description="Date to load (YYYY-MM-DD)"),
    session: SessionType = Query("london", description="Trading session"),
    timeframe: TimeframeType = Query("M5", description="Display timeframe"),
    working_directory: Optional[str] = Query(None),
):
    """
    Load a session's candles for the MMLC dev page.

    Returns just the candles - no waveform calculation.
    """
    cache_path = None
    if working_directory:
        cache_path = Path(working_directory) / settings.cache_folder_name

    # Load M1 data
    df_m1 = load_from_cache(pair.upper(), "M1", cache_path)

    if df_m1 is None:
        raise HTTPException(
            status_code=404,
            detail=f"Instrument not found: {pair}"
        )

    # Filter by date and session
    df_session = filter_by_date_and_session(df_m1, date, session)

    if len(df_session) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for {pair} on {date} during {session} session"
        )

    # Upsample to display timeframe
    df_display = upsample_ohlc(df_session, timeframe)

    # Convert candles to response format
    candles = []
    for row in df_display.iter_rows(named=True):
        candles.append(
            CandleData(
                timestamp=row["timestamp"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"]
            )
        )

    return DevSessionResponse(
        pair=pair.upper(),
        date=str(date),
        session=session,
        timeframe=timeframe,
        candles=candles,
        total_bars=len(candles),
    )


@router.get("/run/{pair}", response_model=DevRunResponse)
async def run_dev_engine(
    pair: str,
    date: date = Query(..., description="Date (YYYY-MM-DD)"),
    session: SessionType = Query("london", description="Trading session"),
    timeframe: TimeframeType = Query("M5", description="Display timeframe"),
    start_bar: int = Query(0, description="First bar to process (0-indexed)"),
    end_bar: Optional[int] = Query(None, description="Last bar to process (inclusive)"),
    mode: str = Query("complete", description="Display mode: 'complete', 'spline', or 'stitch'"),
    working_directory: Optional[str] = Query(None),
):
    """
    Run the MMLC dev engine on the session.

    Processes bars from start_bar to end_bar and returns the resulting waveform.
    """
    cache_path = None
    if working_directory:
        cache_path = Path(working_directory) / settings.cache_folder_name

    # Load M1 data
    df_m1 = load_from_cache(pair.upper(), "M1", cache_path)

    if df_m1 is None:
        raise HTTPException(
            status_code=404,
            detail=f"Instrument not found: {pair}"
        )

    # Filter by date and session
    df_session = filter_by_date_and_session(df_m1, date, session)

    if len(df_session) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for {pair} on {date} during {session} session"
        )

    # Upsample to display timeframe
    df_display = upsample_ohlc(df_session, timeframe)

    # Convert to Candle objects for the engine
    candles = []
    for row in df_display.iter_rows(named=True):
        candles.append(
            Candle(
                timestamp=row["timestamp"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
        )

    # Default end_bar to last bar
    if end_bar is None:
        end_bar = len(candles) - 1

    # Run the dev engine
    engine = MMLCDevEngine()
    waves = engine.process_session(candles, start_bar, end_bar, mode=mode)

    # Convert waves to response format
    wave_data = []
    for wave in waves:
        wave_data.append(
            WaveData(
                id=wave.id,
                level=wave.level,
                direction=wave.direction.value,
                start_time=wave.start_time,
                end_time=wave.end_time,
                start_price=wave.start_price,
                end_price=wave.end_price,
                color=wave.color,
                parent_id=wave.parent_id,
                is_spline=wave.is_spline,
            )
        )

    # Convert stitch annotations if in stitch mode
    annotations = []
    if mode == "stitch":
        for ann in engine.stitch_annotations:
            annotations.append(
                StitchAnnotation(
                    bar=ann['bar'],
                    timestamp=ann['timestamp'],
                    price=ann['price'],
                    level=ann['level'],
                    is_high=ann['is_high'],
                    text=ann['text'],
                )
            )

    # Convert swing labels
    swing_labels = []
    for label in engine.swing_labels:
        if label.child_level is not None and label.child_price is not None and label.bars_ago is not None:
            swing_labels.append(
                SwingLabelData(
                    bar=label.bar,
                    timestamp=label.timestamp.isoformat(),
                    price=label.price,
                    is_high=label.is_high,
                    child_level=label.child_level,
                    child_price=label.child_price,
                    bars_ago=label.bars_ago,
                )
            )

    # Get debug state from engine
    debug_state = DebugState(**engine.get_debug_state(end_bar))

    return DevRunResponse(
        waves=wave_data,
        start_bar=start_bar,
        end_bar=end_bar,
        bars_processed=end_bar - start_bar + 1,
        annotations=annotations,
        swing_labels=swing_labels,
        debug_state=debug_state,
    )
