"""
tests/test_features.py
----------------------
Unit tests for feature engineering and heuristic labeling logic in src/build_features.py.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_features import (
    compute_time_of_day,
    compute_season,
    assign_category,
    compute_historical_frequency,
)


def test_compute_time_of_day():
    """Verify time-of-day bucketing across all 24 hours."""
    times = pd.Series([230, 815, 1420, 2045, None, 0])
    tod = compute_time_of_day(times)

    assert tod.iloc[0] == "Night"      # 02:30
    assert tod.iloc[1] == "Morning"    # 08:15
    assert tod.iloc[2] == "Afternoon"  # 14:20
    assert tod.iloc[3] == "Evening"    # 20:45
    assert tod.iloc[4] == "Night"      # default for None (00:00)
    assert tod.iloc[5] == "Night"      # 00:00


def test_compute_season():
    """Verify meteorological season classification for Indian climate."""
    months = pd.Series([1, 4, 7, 10, 11, 12])
    seasons = compute_season(months)

    assert seasons.iloc[0] == "Winter"        # January
    assert seasons.iloc[1] == "Summer"        # April
    assert seasons.iloc[2] == "Monsoon"       # July
    assert seasons.iloc[3] == "Post-Monsoon"  # October
    assert seasons.iloc[4] == "Post-Monsoon"  # November
    assert seasons.iloc[5] == "Winter"        # December


def test_assign_category_industrial():
    """Persistent emitter (freq >= 3) or near industrial site should classify as Industrial."""
    row1 = pd.Series({
        "land_use_type": "industrial",
        "frp": 15.0,
        "distance_to_industrial": 0.0,
        "historical_frequency": 4
    })
    assert assign_category(row1) == "Industrial (Normal)"

    row2 = pd.Series({
        "land_use_type": "farmland",
        "frp": 12.0,
        "distance_to_industrial": 400.0,  # < 1000m
        "historical_frequency": 2
    })
    assert assign_category(row2) == "Industrial (Normal)"


def test_assign_category_wildfire():
    """High FRP (>= 7.0 MW) + sudden onset (freq <= 1) on non-industrial land is Wildfire."""
    row = pd.Series({
        "land_use_type": "forest",
        "frp": 25.0,
        "distance_to_industrial": 20000.0,
        "historical_frequency": 0
    })
    assert assign_category(row) == "Wildfire Risk"


def test_assign_category_agricultural():
    """Farmland with moderate repetitive activity or low FRP is Agricultural Burning."""
    row = pd.Series({
        "land_use_type": "farmland",
        "frp": 3.5,
        "distance_to_industrial": 15000.0,
        "historical_frequency": 1
    })
    assert assign_category(row) == "Agricultural Burning"


def test_compute_historical_frequency():
    """Hotspots within 0.01 deg grid of each other should have frequency >= 1."""
    data = pd.DataFrame({
        "latitude": [28.6139, 28.6140, 28.6141, 19.0760],
        "longitude": [77.2090, 77.2091, 77.2092, 72.8777],
        "acq_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-01"]
    })
    freq = compute_historical_frequency(data)
    assert freq.iloc[0] == 2
    assert freq.iloc[1] == 2
    assert freq.iloc[2] == 2
    assert freq.iloc[3] == 0
