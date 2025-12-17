from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import Optional
from datetime import date

from app.config import settings, SessionType, TimeframeType, TIMEFRAME_TIMEDELTAS
from app.schemas.chart import ChartResponse, CandleData, WaveData
from app.core.cache_manager import load_from_cache
from app.core.session_filter import filter_by_date_and_session
from app.core.upsampler import upsample_ohlc
from app.waveform.engine import WaveformEngine

router = APIRouter()


@router.get("/{pair}", response_model=ChartResponse)
async def get_chart_data(
    pair: str,
    timeframe: TimeframeType = Query("M5", description="Display timeframe"),
    date: date = Query(..., description="Date to display (YYYY-MM-DD)"),
    session: SessionType = Query("full_day", description="Trading session"),
    working_directory: Optional[str] = Query(None)
):
    """
    Get chart data with candlesticks and waveform overlay.

    The waveform is calculated on M1 data, then the display is upsampled to the requested timeframe.
    """
    cache_path = None
    if working_directory:
        cache_path = Path(working_directory) / settings.cache_folder_name

    # Load M1 data for waveform calculation
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

    # Upsample to display timeframe FIRST
    df_display = upsample_ohlc(df_session, timeframe)

    # Calculate waveform on the upsampled data (matches display timeframe)
    engine = WaveformEngine()
    waves = engine.process_session(df_display)

    # Add pre-swing: from open (left of first bar) to first L1's start
    if waves and len(df_display) > 0:
        first_l1 = next((w for w in waves if w.level == 1), None)
        if first_l1:
            first_row = df_display.row(0, named=True)
            first_open = first_row["open"]
            offset = TIMEFRAME_TIMEDELTAS.get(timeframe, TIMEFRAME_TIMEDELTAS["M5"])

            from app.waveform.wave import Wave, Direction
            pre_direction = Direction.DOWN if first_l1.start_price < first_open else Direction.UP
            pre_wave = Wave(
                id=0,
                level=1,
                direction=pre_direction,
                start_time=first_l1.start_time - offset,
                start_price=first_open,
                end_time=first_l1.start_time,
                end_price=first_l1.start_price,
                parent_id=None,
                is_active=True,
            )
            waves.insert(0, pre_wave)

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

    # Convert waves to response format
    waveform = []
    for wave in waves:
        waveform.append(
            WaveData(
                id=wave.id,
                level=wave.level,
                direction=wave.direction.value,
                start_time=wave.start_time,
                end_time=wave.end_time,
                start_price=wave.start_price,
                end_price=wave.end_price,
                color=wave.color,
                parent_id=wave.parent_id
            )
        )

    return ChartResponse(
        pair=pair.upper(),
        timeframe=timeframe,
        date=str(date),
        session=session,
        candles=candles,
        waveform=waveform
    )


@router.get("/{pair}/preview")
async def get_chart_preview(
    pair: str,
    date: date = Query(..., description="Date to preview"),
    session: SessionType = Query("full_day"),
    working_directory: Optional[str] = Query(None)
):
    """Get a lightweight preview of available data (no waveform calculation)."""
    cache_path = None
    if working_directory:
        cache_path = Path(working_directory) / settings.cache_folder_name

    df = load_from_cache(pair.upper(), "H1", cache_path)

    if df is None:
        raise HTTPException(
            status_code=404,
            detail=f"Instrument not found: {pair}"
        )

    df_session = filter_by_date_and_session(df, date, session)

    if len(df_session) == 0:
        return {
            "pair": pair.upper(),
            "date": str(date),
            "session": session,
            "bar_count": 0,
            "high": None,
            "low": None
        }

    return {
        "pair": pair.upper(),
        "date": str(date),
        "session": session,
        "bar_count": len(df_session),
        "high": float(df_session["high"].max()),
        "low": float(df_session["low"].min())
    }
