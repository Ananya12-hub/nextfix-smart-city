"""Data loading and integrity checks for the three provided dataset files."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load(data_dir=DATA_DIR):
    """Return (locations, incidents, complaints) as DataFrames, validated."""
    data_dir = Path(data_dir)
    L = pd.read_csv(data_dir / "locations.csv")
    I = pd.read_csv(data_dir / "incidents.csv", parse_dates=["timestamp"])
    C = pd.read_csv(data_dir / "public_complaints.csv", parse_dates=["created_at"])
    validate(L, I, C)
    return L, I, C


def validate(L, I, C):
    """Fail loudly if the join key or IDs are broken."""
    assert L.location_id.is_unique, "duplicate location_id"
    assert I.incident_id.is_unique and C.complaint_id.is_unique, "duplicate record IDs"
    assert I.location_id.isin(L.location_id).all(), "incident with unknown location_id"
    assert C.location_id.isin(L.location_id).all(), "complaint with unknown location_id"
    assert not L.isna().any().any(), "missing values in locations.csv"
