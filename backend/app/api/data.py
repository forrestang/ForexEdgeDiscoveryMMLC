from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import Optional
from datetime import date

from app.config import settings
from app.schemas.data import InstrumentsResponse, InstrumentInfo, InstrumentMetadata
from app.core.cache_manager import get_cached_instruments, load_from_cache, clear_pair_cache

router = APIRouter()


@router.get("", response_model=InstrumentsResponse)
async def list_instruments(
    working_directory: Optional[str] = Query(None, description="Override default working directory")
):
    """List all cached instruments with their metadata."""
    cache_path = None
    if working_directory:
        cache_path = Path(working_directory) / settings.cache_folder_name

    instruments_data = get_cached_instruments(cache_path)

    instruments = []
    for data in instruments_data:
        instruments.append(
            InstrumentInfo(
                pair=data["pair"],
                timeframes=data["timeframes"],
                start_date=data["start_date"] or date.today(),
                end_date=data["end_date"] or date.today(),
                file_count=data["file_count"]
            )
        )

    return InstrumentsResponse(instruments=instruments)


@router.get("/{pair}/metadata", response_model=InstrumentMetadata)
async def get_instrument_metadata(
    pair: str,
    working_directory: Optional[str] = Query(None)
):
    """Get detailed metadata for a specific instrument."""
    cache_path = None
    if working_directory:
        cache_path = Path(working_directory) / settings.cache_folder_name

    # Load M1 data to get full metadata
    df = load_from_cache(pair.upper(), "M1", cache_path)

    if df is None:
        raise HTTPException(
            status_code=404,
            detail=f"Instrument not found: {pair}"
        )

    # Get available timeframes
    instruments_data = get_cached_instruments(cache_path)
    pair_data = next((d for d in instruments_data if d["pair"] == pair.upper()), None)

    if pair_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Instrument not found: {pair}"
        )

    return InstrumentMetadata(
        pair=pair.upper(),
        available_timeframes=pair_data["timeframes"],
        start_date=df["timestamp"].min().date(),
        end_date=df["timestamp"].max().date(),
        total_bars=len(df)
    )


@router.delete("/{pair}")
async def delete_instrument(
    pair: str,
    working_directory: Optional[str] = Query(None)
):
    """Remove cached data for an instrument."""
    cache_path = None
    if working_directory:
        cache_path = Path(working_directory) / settings.cache_folder_name

    count = clear_pair_cache(pair.upper(), cache_path)

    if count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Instrument not found: {pair}"
        )

    return {"message": f"Removed {count} cache files for {pair.upper()}"}


@router.get("/{pair}/dates")
async def get_available_dates(
    pair: str,
    working_directory: Optional[str] = Query(None)
):
    """Get list of dates with data available for an instrument."""
    cache_path = None
    if working_directory:
        cache_path = Path(working_directory) / settings.cache_folder_name

    df = load_from_cache(pair.upper(), "M1", cache_path)

    if df is None:
        raise HTTPException(
            status_code=404,
            detail=f"Instrument not found: {pair}"
        )

    # Get unique dates
    dates = df.select("timestamp").with_columns(
        df["timestamp"].dt.date().alias("date")
    ).select("date").unique().sort("date")

    return {"pair": pair.upper(), "dates": dates["date"].to_list()}
