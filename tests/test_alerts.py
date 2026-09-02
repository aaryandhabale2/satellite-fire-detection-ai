"""
tests/test_alerts.py
--------------------
Tests for alert formatting, Google Maps link generation, and threshold filtering in src/send_alerts.py.
"""

import pandas as pd
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from send_alerts import maps_link, build_html_email, build_text_email, send_alert_email, send_alerts


def test_maps_link():
    """Verify Google Maps URL formatting."""
    url = maps_link(28.6139, 77.2090)
    assert url == "https://www.google.com/maps?q=28.61390,77.20900"


def test_build_text_email():
    """Verify plain text email formatting."""
    df = pd.DataFrame([
        {
            "latitude": 20.5,
            "longitude": 80.5,
            "frp": 65.0,
            "acq_date": "2026-03-01",
            "land_use_type": "forest",
        }
    ])
    text = build_text_email(df, threshold=50.0)
    assert "WILDFIRE RISK ALERT" in text
    assert "65.0 MW" in text
    assert "20.5000N, 80.5000E" in text


def test_send_alerts_dry_run():
    """Verify dry_run execution without real SMTP dispatch."""
    df = pd.DataFrame([
        {"category": "Wildfire Risk", "frp": 85.0, "latitude": 21.0, "longitude": 81.0, "acq_date": "2026-03-01"},
        {"category": "Industrial (Normal)", "frp": 25.0, "latitude": 22.0, "longitude": 82.0, "acq_date": "2026-03-01"},
    ])
    success = send_alerts(df, frp_threshold=50.0, dry_run=True)
    assert success is True


def test_build_html_email():
    """Verify HTML email generation contains essential alert details."""
    hotspots = pd.DataFrame([
        {
            "latitude": 21.5,
            "longitude": 84.2,
            "frp": 75.5,
            "acq_date": "2026-04-15",
            "acq_time": "1330",
            "land_use_type": "forest",
            "historical_frequency": 0,
            "time_of_day": "Afternoon",
            "season": "Summer",
        }
    ])

    html = build_html_email(hotspots, threshold=50.0)
    assert "Wildfire Alert" in html or "Wildfire Risk" in html
    assert "75.5 MW" in html
    assert "21.5000°N, 84.2000°E" in html
    assert "maps.google.com" in html or "google.com/maps" in html
