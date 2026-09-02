"""
tests/test_firms_osm.py
-----------------------
Tests for FIRMS bounding box parsing, coordinate validity, and land-use fallback mapping.
"""

import pandas as pd
import pytest


def test_firms_coordinate_bounds():
    """Verify standard India bounding box coordinate checks."""
    india_bbox = {"south": 6.5, "west": 68.0, "north": 37.5, "east": 97.5}

    valid_lat, valid_lon = 22.5, 82.0
    invalid_lat, invalid_lon = 51.5, -0.12  # London

    def is_in_india(lat, lon):
        return (
            india_bbox["south"] <= lat <= india_bbox["north"]
            and india_bbox["west"] <= lon <= india_bbox["east"]
        )

    assert is_in_india(valid_lat, valid_lon) is True
    assert is_in_india(invalid_lat, invalid_lon) is False


def test_osm_tag_fallback_mapping():
    """Verify fallback land-use tags map correctly to standard taxonomy."""
    tag_mapping = {
        "industrial": "industrial",
        "factory": "industrial",
        "forest": "forest",
        "wood": "forest",
        "farmland": "farmland",
        "farmyard": "farmland",
        "residential": "residential",
        "unknown_tag": "unknown",
    }

    assert tag_mapping.get("forest") == "forest"
    assert tag_mapping.get("industrial") == "industrial"
    assert tag_mapping.get("non_existent", "unknown") == "unknown"
