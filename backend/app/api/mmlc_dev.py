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
    stitch_swings: list[dict] = []  # Swing points: bar, price, direction (+1/-1)
    stitch_swings_level: list[int] = []  # Parallel array: level each swing originated from (0=OPEN, 1=L1, 2=L2, etc.)
    prev_L1_Direction: str
    num_waves_returned: int
    L1_count: int = 0  # Number of extremes in current L1 direction
    L1_leg: int = 0  # Overall L1 swing number (never resets)


class MMLCSwingData(BaseModel):
    """A single swing point (vertex) in the waveform for mmlcOut."""
    bar: int              # Bar index (-1 for preswing)
    price: float          # Price at swing point
    direction: int        # +1 (HIGH), -1 (LOW), 0 (OPEN anchor)
    level: int            # Wave level (0=OPEN, 1=L1, 2=L2, etc.)


class MMLCLegData(BaseModel):
    """A single leg in the waveform history for mmlcOut (deprecated)."""
    start_bar: int
    start_price: float
    end_bar: int
    end_price: float
    level: int
    direction: int  # +1 or -1
    is_developing: bool


class MMLCBarData(BaseModel):
    """MMLC state captured at a single bar for mmlcOut."""
    bar_index: int
    session_open_price: float
    total_session_bars: int
    current_close: float
    swings: list[MMLCSwingData] = []  # Primary output - swing array
    legs: list[MMLCLegData] = []       # Deprecated


# LSTM Output Models
class LSTMVectorData(BaseModel):
    """Vector features for LSTM input."""
    price_raw: float
    price_delta: float
    time_delta: int


class LSTMStateData(BaseModel):
    """State classification for LSTM."""
    level: int
    direction: str  # "UP" or "DOWN"
    event: str      # "EXTENSION", "RETRACEMENT", or "REVERSAL"


class LSTMOutcomeData(BaseModel):
    """Forward-looking outcome for supervised training."""
    next_bar_delta: float      # Close[T+1] - Close[T]
    session_close_delta: float # Close[SessionEnd] - Close[T]
    session_max_up: float      # MFE: Max(High[T...End]) - Close[T]
    session_max_down: float    # MAE: Min(Low[T...End]) - Close[T] (negative)


class LSTMBarPayloadData(BaseModel):
    """LSTM-focused output for each bar."""
    sequence_id: int
    timestamp: str
    total_session_bars: int
    vector: LSTMVectorData
    state: LSTMStateData
    outcome: Optional[LSTMOutcomeData] = None


class DevRunResponse(BaseModel):
    """Response for running the dev engine."""
    waves: list[WaveData]
    start_bar: int
    end_bar: int
    bars_processed: int
    annotations: list[StitchAnnotation] = []
    swing_labels: list[SwingLabelData] = []
    debug_state: Optional[DebugState] = None
    mmlc_out: list[MMLCBarData] = []  # Autoencoder training data
    lstm_out: list[LSTMBarPayloadData] = []  # LSTM training data


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

    # Convert mmlcOut to response format
    mmlc_out_data = []
    print(f"[MMLC_OUT API] engine.mmlc_out has {len(engine.mmlc_out)} snapshots", flush=True)
    for snapshot in engine.mmlc_out:
        mmlc_out_data.append(
            MMLCBarData(
                bar_index=snapshot.bar_index,
                session_open_price=snapshot.session_open_price,
                total_session_bars=snapshot.total_session_bars,
                current_close=snapshot.current_close,
                swings=[
                    MMLCSwingData(
                        bar=swing.bar,
                        price=swing.price,
                        direction=swing.direction,
                        level=swing.level,
                    )
                    for swing in snapshot.swings
                ],
                legs=[
                    MMLCLegData(
                        start_bar=leg.start_bar,
                        start_price=leg.start_price,
                        end_bar=leg.end_bar,
                        end_price=leg.end_price,
                        level=leg.level,
                        direction=leg.direction,
                        is_developing=leg.is_developing,
                    )
                    for leg in snapshot.legs
                ]
            )
        )

    # Convert lstm_out to response format
    lstm_out_data = []
    for payload in engine.lstm_out:
        outcome_data = None
        if payload.outcome is not None:
            outcome_data = LSTMOutcomeData(
                next_bar_delta=payload.outcome.next_bar_delta,
                session_close_delta=payload.outcome.session_close_delta,
                session_max_up=payload.outcome.session_max_up,
                session_max_down=payload.outcome.session_max_down,
            )
        lstm_out_data.append(
            LSTMBarPayloadData(
                sequence_id=payload.sequence_id,
                timestamp=payload.timestamp,
                total_session_bars=payload.total_session_bars,
                vector=LSTMVectorData(
                    price_raw=payload.price_raw,
                    price_delta=payload.price_delta,
                    time_delta=payload.time_delta,
                ),
                state=LSTMStateData(
                    level=payload.level,
                    direction=payload.direction,
                    event=payload.event,
                ),
                outcome=outcome_data,
            )
        )

    return DevRunResponse(
        waves=wave_data,
        start_bar=start_bar,
        end_bar=end_bar,
        bars_processed=end_bar - start_bar + 1,
        annotations=annotations,
        swing_labels=swing_labels,
        debug_state=debug_state,
        mmlc_out=mmlc_out_data,
        lstm_out=lstm_out_data,
    )
