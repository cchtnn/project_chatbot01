from pathlib import Path
import pandas as pd
from core import get_logger
from pathlib import Path

logger = get_logger(__name__)
_cache: dict[str, pd.DataFrame] = {}

# backend/ directory
BACKEND_ROOT = Path(__file__).resolve().parents[1]

PROJECT_ROOT = BACKEND_ROOT.parent


def get_transcript_df() -> pd.DataFrame:
    if "transcripts" in _cache:
        return _cache["transcripts"]

    path = BACKEND_ROOT / "data" / "documents" / "transcripts" / "merged_transcripts.csv"
    if not path.exists():
        logger.warning(f"No merged transcripts found at {path}")
        _cache["transcripts"] = pd.DataFrame()
        return _cache["transcripts"]

    logger.info(f"Loading merged transcripts from {path}")
    df = pd.read_csv(path)
    _cache["transcripts"] = df
    return df


def get_payroll_df() -> pd.DataFrame:
    """2026 payroll calendar CSV."""
    if "payroll" in _cache:
        return _cache["payroll"]

    path = (
        PROJECT_ROOT
        / "data"
        / "documents"
        / "payroll_cal"
        / "csv_files"
        / "2026Payroll_Calendar_payroll.csv"
    )
    if not path.exists():
        logger.warning(f"No payroll calendar found at {path}")
        _cache["payroll"] = pd.DataFrame()
        return _cache["payroll"]

    logger.info(f"Loading payroll calendar from {path}")
    df = pd.read_csv(path)
    _cache["payroll"] = df
    return df


import json
from pathlib import Path
import pandas as pd
from core import get_logger

logger = get_logger(__name__)
_cache: dict[str, pd.DataFrame] = {}

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent


def get_bor_schedule_df() -> pd.DataFrame:
    """BOR schedule flattened from bor_json.json."""
    if "bor" in _cache:
        return _cache["bor"]

    path = PROJECT_ROOT / "data" / "documents" / "bor_json.json"
    if not path.exists():
        logger.warning(f"No BOR JSON found at {path}")
        _cache["bor"] = pd.DataFrame()
        return _cache["bor"]

    logger.info(f"Loading BOR JSON from {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Expect: list of meeting/event dicts
    df = pd.json_normalize(data)
    logger.info("BOR JSON normalized shape=%s cols=%s", df.shape, list(df.columns))
    _cache["bor"] = df
    return df

def get_transcript_csv_path() -> str:
    """Get the path to the transcript CSV file."""
    # Adjust this path based on your project structure
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "data" / "documents" / "transcripts" / "merged_transcripts.csv"
    
    # Alternative: Check if path is in settings
    from config import get_settings
    settings = get_settings()
    if hasattr(settings, 'TRANSCRIPT_CSV_PATH'):
        return settings.TRANSCRIPT_CSV_PATH
    
    return str(csv_path)

