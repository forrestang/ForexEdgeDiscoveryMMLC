from pathlib import Path
from datetime import timedelta
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    default_working_directory: Path = Path(r"C:\Users\lawfp\Desktop\Data4")
    data_folder_name: str = "data"
    cache_folder_name: str = "cache"

    @property
    def data_path(self) -> Path:
        return self.default_working_directory / self.data_folder_name

    @property
    def cache_path(self) -> Path:
        return self.default_working_directory / self.cache_folder_name


# Session time boundaries in UTC (hour, minute)
SESSION_TIMES = {
    "full_day": ((0, 0), (22, 0)),
    "asia": ((0, 0), (9, 0)),
    "london": ((8, 0), (17, 0)),
    "ny": ((13, 0), (22, 0)),
    # Combo sessions
    "asia_london": ((0, 0), (17, 0)),   # Asia start to London end
    "london_ny": ((8, 0), (22, 0)),     # London start to NY end
}

SessionType = Literal["full_day", "asia", "london", "ny", "asia_london", "london_ny"]

TIMEFRAMES = ["M1", "M5", "M10", "M15", "M30", "H1", "H4"]
TimeframeType = Literal["M1", "M5", "M10", "M15", "M30", "H1", "H4"]

# Map timeframe strings to Polars interval format
TIMEFRAME_INTERVALS = {
    "M1": "1m",
    "M5": "5m",
    "M10": "10m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
}

# Map timeframe strings to timedelta for offset calculations
TIMEFRAME_TIMEDELTAS = {
    "M1": timedelta(minutes=1),
    "M5": timedelta(minutes=5),
    "M10": timedelta(minutes=10),
    "M15": timedelta(minutes=15),
    "M30": timedelta(minutes=30),
    "H1": timedelta(hours=1),
    "H4": timedelta(hours=4),
}

# Wave colors - cycle through these (L1=index 0, L2=index 1, etc.)
WAVE_COLORS = [
    "#FFD700",  # Yellow (L1)
    "#00FFFF",  # Cyan (L2)
    "#FF0000",  # Red (L3)
    "#800080",  # Purple (L4)
    "#90EE90",  # Light Green (L5)
]


settings = Settings()
