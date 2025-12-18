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
from app.waveform.streaming_engine import StreamingWaveformEngine

router = APIRouter()


@router.get("/{pair}", response_model=ChartResponse)
async def get_chart_data(
    pair: str,
    timeframe: TimeframeType = Query("M5", description="Display timeframe"),
    date: date = Query(..., description="Date to display (YYYY-MM-DD)"),
    session: SessionType = Query("full_day", description="Trading session"),
    working_directory: Optional[str] = Query(None),
    bar_index: Optional[int] = Query(None, description="If provided, show waveform state at this bar index (0-indexed)")
):
    """
    Get chart data with candlesticks and waveform overlay.

    The waveform is calculated on M1 data, then the display is upsampled to the requested timeframe.

    If bar_index is provided, the waveform will be filtered to show only the state
    at that specific bar (point-in-time view).
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
    if bar_index is not None:
        # Use streaming engine to get snapshots for point-in-time filtering
        streaming_engine = StreamingWaveformEngine()
        _, snapshots = streaming_engine.process_session_with_snapshots(df_display)
        # Use ALL waves (not filtered) to properly handle point-in-time view
        all_waves = streaming_engine.waves

        # Validate bar_index
        if bar_index < 0 or bar_index >= len(df_display):
            bar_index = len(df_display) - 1  # Clamp to valid range

        # Get the snapshot at the requested bar_index
        if bar_index < len(snapshots):
            snapshot = snapshots[bar_index]
            target_bar = df_display.row(bar_index, named=True)
            target_timestamp = target_bar["timestamp"]
            target_close = target_bar["close"]

            # Filter waves to show only those visible at bar_index
            filtered_waves = []

            # Find the deepest level in the snapshot (the "final leg")
            deepest_level = max((ws.level for ws in snapshot.waves), default=0)

            # Build a map of snapshot waves by (start_bar_index, level) for fast lookup
            snapshot_wave_map = {
                (ws.start_bar_index, ws.level): ws for ws in snapshot.waves
            }

            # Debug info collection
            debug_info = []
            debug_info.append(f"bar_index={bar_index}, deepest_level={deepest_level}")
            debug_info.append(f"snapshot.waves: {[(ws.level, ws.start_bar_index, ws.amplitude) for ws in snapshot.waves]}")

            for wave in all_waves:
                # Get wave's start bar from streaming engine's tracking
                wave_start_bar = streaming_engine._wave_start_bars.get(wave.id, 0)

                # Skip waves that don't exist yet at bar_index
                if wave_start_bar > bar_index:
                    continue

                # Find matching snapshot for this wave
                matching_snapshot = snapshot_wave_map.get((wave_start_bar, wave.level))

                if matching_snapshot:
                    # Wave is ACTIVE at bar_index
                    from app.waveform.wave import Wave as WaveClass

                    if wave.level == deepest_level:
                        # Deepest wave syncs to current bar's close
                        end_time = target_timestamp
                        end_price = target_close
                        debug_info.append(f"Wave L{wave.level} id={wave.id} is DEEPEST -> end_time={end_time}")
                    else:
                        # Non-deepest waves: find the child wave that started from this wave's extreme
                        # Use the child wave's start_time/start_price for continuity
                        child_snapshot = None
                        for ws in snapshot.waves:
                            if ws.level == wave.level + 1:
                                child_snapshot = ws
                                break

                        if child_snapshot:
                            # Find the actual child wave object to get its start_time/start_price
                            child_wave = None
                            for cw in all_waves:
                                cw_start_bar = streaming_engine._wave_start_bars.get(cw.id, -1)
                                if cw.level == child_snapshot.level and cw_start_bar == child_snapshot.start_bar_index:
                                    child_wave = cw
                                    break

                            if child_wave:
                                # Use child's start as parent's end for visual continuity
                                end_time = child_wave.start_time
                                end_price = child_wave.start_price
                                debug_info.append(f"Wave L{wave.level} id={wave.id} -> child L{child_wave.level} id={child_wave.id} start: {end_time}")
                            else:
                                # Fallback to snapshot bar lookup
                                child_start_row = df_display.row(child_snapshot.start_bar_index, named=True)
                                end_time = child_start_row["timestamp"]
                                end_price = wave.start_price + matching_snapshot.amplitude
                                debug_info.append(f"Wave L{wave.level} id={wave.id} found child at bar {child_snapshot.start_bar_index} (fallback)")
                        else:
                            # No child found but wave is not deepest - use target_timestamp
                            end_time = target_timestamp
                            end_price = wave.start_price + matching_snapshot.amplitude
                            debug_info.append(f"Wave L{wave.level} id={wave.id} NO CHILD FOUND -> end_time={end_time}")

                    truncated_wave = WaveClass(
                        id=wave.id,
                        level=wave.level,
                        direction=wave.direction,
                        start_time=wave.start_time,
                        start_price=wave.start_price,
                        end_time=end_time,
                        end_price=end_price,
                        parent_id=wave.parent_id,
                        is_active=True,
                    )
                    filtered_waves.append(truncated_wave)
                else:
                    # Wave is NOT in the active snapshot at bar_index
                    # Only include completed L1 waves (they form the historical structure)
                    # Do NOT include erased L2+ waves - they were temporary
                    if wave.level == 1 and wave.end_time <= target_timestamp:
                        filtered_waves.append(wave)
                        debug_info.append(f"Wave L{wave.level} id={wave.id} COMPLETED L1, including with end_time={wave.end_time}")

            waves = filtered_waves
        else:
            # Snapshot not available - use all waves
            waves = all_waves
            debug_info = ["Snapshot not available for bar_index"]
    else:
        # Full session waveform (original behavior)
        engine = WaveformEngine()
        waves = engine.process_session(df_display)
        debug_info = None  # No debug for full session

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
        waveform=waveform,
        debug=debug_info
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
