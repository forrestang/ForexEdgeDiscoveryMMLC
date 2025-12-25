from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import Optional
from datetime import date

from app.config import settings, SessionType, TimeframeType, TIMEFRAME_TIMEDELTAS
from app.schemas.chart import ChartResponse, CandleData, WaveData, WaveSnapshotData, StackSnapshotData
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
    snapshot_response = None  # For debug panel
    if bar_index is not None:
        # Use streaming engine to get snapshots for point-in-time filtering
        streaming_engine = StreamingWaveformEngine()
        _, snapshots = streaming_engine.process_session_with_snapshots(df_display, debug_bar_index=bar_index)
        # Use ALL waves (not filtered) to properly handle point-in-time view
        all_waves = streaming_engine.waves

        # Validate bar_index
        if bar_index < 0 or bar_index >= len(df_display):
            bar_index = len(df_display) - 1  # Clamp to valid range

        # Get the snapshot at the requested bar_index
        if bar_index < len(snapshots):
            snapshot = snapshots[bar_index]

            # Build snapshot response for debug panel
            snapshot_response = StackSnapshotData(
                bar_index=snapshot.bar_index,
                timestamp=snapshot.timestamp,
                close_price=snapshot.close_price,
                waves=[
                    WaveSnapshotData(
                        level=ws.level,
                        direction=ws.direction,
                        amplitude=ws.amplitude,
                        duration_bars=ws.duration_bars,
                        start_bar_index=ws.start_bar_index,
                    )
                    for ws in snapshot.waves
                ],
                l1_count=snapshot.l1_count,
                l2_count=snapshot.l2_count,
                l3_count=snapshot.l3_count,
                l4_count=snapshot.l4_count,
                l5_count=snapshot.l5_count,
            )
            target_bar = df_display.row(bar_index, named=True)
            target_timestamp = target_bar["timestamp"]
            target_close = target_bar["close"]

            # Build waves directly from snapshot data (no matching against all_waves needed)
            from app.waveform.wave import Wave as WaveClass, Direction

            filtered_waves = []
            debug_info = []
            debug_info.append(f"API bar_index={bar_index}, len(snapshots)={len(snapshots)}")
            debug_info.append(f"snapshot.waves ({len(snapshot.waves)} waves): {[(ws.level, ws.start_bar_index, ws.amplitude) for ws in snapshot.waves]}")

            # Find the deepest level in the snapshot
            deepest_level = max((ws.level for ws in snapshot.waves), default=0)
            debug_info.append(f"deepest_level={deepest_level}")

            # Build active waves from snapshot data
            # Sort by level so we can link parent->child
            sorted_snapshot_waves = sorted(snapshot.waves, key=lambda w: w.level)

            for i, ws in enumerate(sorted_snapshot_waves):
                # Direction from snapshot (+1 = UP, -1 = DOWN)
                direction = Direction.UP if ws.direction > 0 else Direction.DOWN

                # For L1 waves, we need to find where the start_price actually occurred,
                # not where the wave was "created" (start_bar_index)
                # This is because L1 waves can be created mid-bar when erasing previous L1s

                # For start_price, we need to figure out where the wave started
                # For L1: starts at the open of bar 0 or the extreme of previous movement
                # For L2+: starts at the extreme of parent wave
                if ws.level == 1:
                    # L1 wave - amplitude tells us displacement from start to current end
                    # We need to find start_price such that start_price + amplitude = end_price
                    # end_price at bar_index is the extreme (HIGH for UP, LOW for DOWN) or child start
                    if ws.level == deepest_level:
                        # Use the extreme of the current bar, not the close
                        if direction == Direction.UP:
                            end_price = target_bar["high"]
                        else:
                            end_price = target_bar["low"]
                    else:
                        # Find child wave to get end_price
                        child_ws = next((cw for cw in sorted_snapshot_waves if cw.level == ws.level + 1), None)
                        if child_ws:
                            child_start_row = df_display.row(child_ws.start_bar_index, named=True)
                            # Child starts at parent's extreme
                            end_price = child_start_row["close"] if child_ws.direction > 0 else child_start_row["close"]
                            # Actually, for DOWN parent, child UP starts at LOW; for UP parent, child DOWN starts at HIGH
                            if direction == Direction.DOWN:
                                end_price = child_start_row["low"]
                            else:
                                end_price = child_start_row["high"]
                        else:
                            end_price = target_close
                    start_price = end_price - ws.amplitude

                    # For L1, find the bar where start_price actually occurred
                    # Search for the bar with the closest low (for UP) or high (for DOWN)
                    start_bar_idx = 0
                    best_diff = float('inf')
                    for search_idx in range(bar_index + 1):
                        search_row = df_display.row(search_idx, named=True)
                        if direction == Direction.UP:
                            diff = abs(search_row["low"] - start_price)
                        else:
                            diff = abs(search_row["high"] - start_price)
                        if diff < best_diff:
                            best_diff = diff
                            start_bar_idx = search_idx
                    start_row = df_display.row(start_bar_idx, named=True)
                    start_time = start_row["timestamp"]
                    # Also update start_price to match the actual candle extreme
                    if direction == Direction.UP:
                        start_price = start_row["low"]
                    else:
                        start_price = start_row["high"]
                else:
                    # L2+ wave - starts at parent's extreme
                    # Parent's extreme = parent's end_price = this wave's start_price
                    # We already built parent, so find it
                    parent_wave = next((pw for pw in filtered_waves if pw.level == ws.level - 1), None)
                    if parent_wave:
                        start_price = parent_wave.end_price
                        start_time = parent_wave.end_time  # L2+ starts when parent ends
                    else:
                        # Fallback: compute from amplitude
                        if ws.level == deepest_level:
                            # Use the extreme of the current bar, not the close
                            if direction == Direction.UP:
                                end_price = target_bar["high"]
                            else:
                                end_price = target_bar["low"]
                        else:
                            child_ws = next((cw for cw in sorted_snapshot_waves if cw.level == ws.level + 1), None)
                            if child_ws:
                                child_start_row = df_display.row(child_ws.start_bar_index, named=True)
                                if direction == Direction.DOWN:
                                    end_price = child_start_row["low"]
                                else:
                                    end_price = child_start_row["high"]
                            else:
                                end_price = target_close
                        start_price = end_price - ws.amplitude
                        # Fallback start_time from start_bar_index
                        start_row = df_display.row(ws.start_bar_index, named=True)
                        start_time = start_row["timestamp"]

                # Compute end_time and end_price
                if ws.level == deepest_level:
                    # Deepest wave ends at the extreme of current bar (HIGH for UP, LOW for DOWN)
                    end_time = target_timestamp
                    if direction == Direction.UP:
                        end_price = target_bar["high"]
                    else:
                        end_price = target_bar["low"]
                else:
                    # Non-deepest: ends where child starts
                    child_ws = next((cw for cw in sorted_snapshot_waves if cw.level == ws.level + 1), None)
                    if child_ws:
                        child_start_row = df_display.row(child_ws.start_bar_index, named=True)
                        end_time = child_start_row["timestamp"]
                        # Child starts at parent's extreme (high for UP parent, low for DOWN parent)
                        if direction == Direction.UP:
                            end_price = child_start_row["high"]
                        else:
                            end_price = child_start_row["low"]
                    else:
                        end_time = target_timestamp
                        end_price = start_price + ws.amplitude

                wave = WaveClass(
                    id=ws.level * 1000 + ws.start_bar_index,  # Unique ID based on level and start
                    level=ws.level,
                    direction=direction,
                    start_time=start_time,
                    start_price=start_price,
                    end_time=end_time,
                    end_price=end_price,
                    parent_id=(ws.level - 1) * 1000 + sorted_snapshot_waves[i-1].start_bar_index if ws.level > 1 and i > 0 else None,
                    is_active=True,
                )
                filtered_waves.append(wave)
                debug_info.append(f"Built L{ws.level} {'UP' if direction == Direction.UP else 'DOWN'} start={start_price:.5f} end={end_price:.5f}")

            # Add synthetic child wave from extreme to close (the "retracement to close")
            if filtered_waves and deepest_level > 0:
                deepest_wave = next((w for w in filtered_waves if w.level == deepest_level), None)
                if deepest_wave:
                    # Check if close differs from the extreme
                    if deepest_wave.direction == Direction.UP and target_close < deepest_wave.end_price:
                        # Add L(n+1) DOWN from high to close
                        retracement_wave = WaveClass(
                            id=deepest_wave.id + 1,
                            level=deepest_level + 1,
                            direction=Direction.DOWN,
                            start_time=target_timestamp,
                            start_price=deepest_wave.end_price,
                            end_time=target_timestamp,
                            end_price=target_close,
                            parent_id=deepest_wave.id,
                            is_active=True,
                        )
                        filtered_waves.append(retracement_wave)
                        debug_info.append(f"Added retracement L{deepest_level + 1} DOWN from {deepest_wave.end_price:.5f} to {target_close:.5f}")
                    elif deepest_wave.direction == Direction.DOWN and target_close > deepest_wave.end_price:
                        # Add L(n+1) UP from low to close
                        retracement_wave = WaveClass(
                            id=deepest_wave.id + 1,
                            level=deepest_level + 1,
                            direction=Direction.UP,
                            start_time=target_timestamp,
                            start_price=deepest_wave.end_price,
                            end_time=target_timestamp,
                            end_price=target_close,
                            parent_id=deepest_wave.id,
                            is_active=True,
                        )
                        filtered_waves.append(retracement_wave)
                        debug_info.append(f"Added retracement L{deepest_level + 1} UP from {deepest_wave.end_price:.5f} to {target_close:.5f}")

            # Add historical L1 waves that ended BEFORE or AT the current active L1's start
            # This shows the complete path from session start to current position
            current_l1 = next((w for w in filtered_waves if w.level == 1), None)
            if current_l1:
                historical_l1s = [w for w in all_waves if w.level == 1 and w.end_time <= current_l1.start_time]
                filtered_waves = historical_l1s + filtered_waves

            waves = filtered_waves
            debug_info.append(f"Total waves: {len(filtered_waves)}")
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
            first_timestamp = first_row["timestamp"]

            from app.waveform.wave import Wave, Direction
            pre_direction = Direction.DOWN if first_l1.start_price < first_open else Direction.UP
            pre_wave = Wave(
                id=0,
                level=1,
                direction=pre_direction,
                start_time=first_timestamp,
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
        debug=debug_info,
        snapshot=snapshot_response,
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
