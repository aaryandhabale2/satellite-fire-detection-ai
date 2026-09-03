"""
tests/test_model.py
-------------------
Tests for XGBoost model bundle loading, feature columns schema, and real-time inference.
"""

import pickle
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fire_classifier.pkl"


def test_model_bundle_exists():
    """Verify that the trained model bundle is present."""
    assert MODEL_PATH.exists(), "Trained model pickle file does not exist. Run src/train_model.py first."


def test_model_bundle_structure():
    """Verify all required keys and objects exist in the model bundle."""
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    assert isinstance(bundle, dict)
    assert "model" in bundle
    assert "encoders" in bundle
    assert "feature_cols" in bundle
    assert "class_names" in bundle
    assert "target" in bundle["encoders"]

    # Target encoder classes
    classes = list(bundle["encoders"]["target"].classes_)
    assert len(classes) == 4
    assert "Wildfire Risk" in classes
    assert "Agricultural Burning" in classes
    assert "Industrial (Normal)" in classes
    assert "Anomaly/Unclassified" in classes


def test_model_inference_prediction():
    """Verify that the model can perform multi-class predictions and return valid probabilities."""
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    encoders = bundle["encoders"]
    feat_cols = bundle["feature_cols"]
    target_le = encoders["target"]

    # Create a synthetic sample (High intensity forest fire)
    sample_df = pd.DataFrame([{
        "brightness": 380.0,
        "frp": 120.0,
        "log_frp": np.log1p(120.0),
        "distance_to_industrial": 35000.0,
        "historical_frequency": 0,
        "month": 4,
        "land_use_type": "forest",
        "time_of_day": "Afternoon",
        "season": "Summer",
        "scan": 0.4,
        "track": 0.4,
        "confidence_enc": 2,
        "delta_brightness": 70.0,
        "co_confirmed": 1
    }])

    # Encode categorical features
    for col, le in encoders.items():
        if col != "target" and col in sample_df.columns:
            enc_col = f"{col}_enc"
            sample_df[enc_col] = le.transform(
                sample_df[col].fillna("unknown").astype(str).map(
                    lambda x, le=le: x if x in le.classes_ else le.classes_[0]
                )
            )

    available_cols = [c for c in feat_cols if c in sample_df.columns]
    X = sample_df[available_cols].fillna(0).astype(float)

    # Predict
    pred_idx = model.predict(X)
    pred_probs = model.predict_proba(X)
    pred_label = target_le.inverse_transform(pred_idx)[0]

    assert isinstance(pred_label, str)
    assert pred_probs.shape == (1, 4)
    assert np.isclose(pred_probs.sum(), 1.0)
