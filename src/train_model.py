"""
src/train_model.py
------------------
Train an XGBoost multi-class classifier to predict fire/hotspot category
from the engineered features in data/features_labeled.csv.

Pipeline
--------
  1. Load data/features_labeled.csv
  2. Encode categorical features (land_use_type, time_of_day, season)
  3. Train / test split (80/20, stratified)
  4. Train XGBoost multi-class classifier
  5. Evaluate: accuracy, precision, recall, F1, confusion matrix
  6. Generate SHAP feature importance summary plot
  7. Save trained model to models/fire_classifier.pkl

Outputs
-------
  models/fire_classifier.pkl    -- trained model bundle (model + encoders)
  models/confusion_matrix.png   -- confusion matrix heatmap
  models/shap_summary.png       -- SHAP beeswarm feature importance plot
  models/model_report.txt       -- text classification report

Usage
-----
    python src/train_model.py
"""

import pickle
import sys
import warnings
from pathlib import Path

# Alert system — imported lazily so a missing .env never breaks training
try:
    from send_alerts import send_alerts as _send_alerts
    _ALERTS_AVAILABLE = True
except ImportError:
    _ALERTS_AVAILABLE = False

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")   # headless backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[1]
FEATURES_FILE = ROOT / "data"   / "features_labeled.csv"
MODEL_DIR     = ROOT / "models"
MODEL_FILE    = MODEL_DIR / "fire_classifier.pkl"
REPORT_FILE   = MODEL_DIR / "model_report.txt"
CM_FILE       = MODEL_DIR / "confusion_matrix.png"
SHAP_FILE     = MODEL_DIR / "shap_summary.png"

# ---------------------------------------------------------------------------
# Feature columns used for training
# Categorical columns will be label-encoded before training.
# All other feature columns should be numeric.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "brightness",
    "frp",
    "log_frp",
    "distance_to_industrial",
    "historical_frequency",
    "month",
]

CATEGORICAL_FEATURES = [
    "land_use_type",   # industrial | forest | farmland | residential | unknown
    "time_of_day",     # Night | Morning | Afternoon | Evening
    "season",          # Winter | Summer | Monsoon | Post-Monsoon
]

# Optional numeric columns — included if present in the CSV
OPTIONAL_NUMERIC = [
    "scan",            # VIIRS scan pixel size
    "track",           # VIIRS track pixel size
    "confidence_enc",  # encoded detection confidence
]

TARGET_COL = "category"


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_and_prepare(filepath: Path):
    """
    Load features_labeled.csv, encode categorical columns, and return
    (X DataFrame, y Series, class_names list, fitted encoders dict).
    """
    if not filepath.exists():
        print(f"[ERROR] {filepath.name} not found.")
        print("        Run Step 3 first:  python src/build_features.py")
        raise SystemExit(1)

    print(f"\n[STEP 4] Loading {filepath.name} ...")
    df = pd.read_csv(filepath)
    print(f"         {len(df):,} samples, {len(df.columns)} columns")

    # Verify target column exists
    if TARGET_COL not in df.columns:
        print(f"[ERROR] Target column '{TARGET_COL}' not found in CSV.")
        print(f"        Available columns: {list(df.columns)}")
        raise SystemExit(1)

    # Drop rows with missing target
    n_before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  [WARN] Dropped {n_before - len(df)} rows with missing category.")

    # ── Encode categorical features ────────────────────────────────────────
    encoders = {}
    encoded_cat_cols = []

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            print(f"  [WARN] Column '{col}' not in data — skipping.")
            continue
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].fillna("unknown").astype(str))
        encoders[col] = le
        encoded_cat_cols.append(f"{col}_enc")
        print(f"  Encoded '{col}' -> '{col}_enc'  classes: {list(le.classes_)}")

    # ── Assemble feature matrix ────────────────────────────────────────────
    available_numeric   = [c for c in NUMERIC_FEATURES if c in df.columns]
    available_optional  = [c for c in OPTIONAL_NUMERIC  if c in df.columns]
    all_feature_cols    = available_numeric + available_optional + encoded_cat_cols

    missing = set(NUMERIC_FEATURES) - set(df.columns)
    if missing:
        print(f"  [WARN] Missing numeric features (skipped): {missing}")

    X = df[all_feature_cols].fillna(0).astype(float)
    print(f"\n  Feature columns used ({len(all_feature_cols)}): {all_feature_cols}")

    # ── Encode target ──────────────────────────────────────────────────────
    target_le = LabelEncoder()
    y = target_le.fit_transform(df[TARGET_COL])
    class_names = list(target_le.classes_)
    encoders["target"] = target_le

    print(f"\n  Target classes ({len(class_names)}):")
    for i, cls in enumerate(class_names):
        cnt = (y == i).sum()
        print(f"    [{i}] {cls:<28}  {cnt:>6,} samples  ({cnt/len(y)*100:.1f}%)")

    return X, y, class_names, encoders


# ---------------------------------------------------------------------------
# Model training with cross-validation
# ---------------------------------------------------------------------------

def train_model(X: pd.DataFrame, y: np.ndarray, class_names: list) -> XGBClassifier:
    """
    Train an XGBoost multi-class classifier.
    Runs 5-fold stratified cross-validation first to report generalisation.
    Then fits the final model on the full training set.
    """
    n_classes = len(class_names)

    model = XGBClassifier(
        n_estimators      = 300,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 3,
        gamma             = 0.1,
        objective         = "multi:softprob",
        num_class         = n_classes,
        eval_metric       = "mlogloss",
        use_label_encoder = False,
        random_state      = 42,
        n_jobs            = -1,
    )

    # 5-fold cross-validation on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n  Running 5-fold stratified cross-validation ...")

    for metric_name, scoring in [("Accuracy", "accuracy"), ("F1-macro", "f1_macro")]:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        print(f"    CV {metric_name:<12}: {scores.mean():.4f}  (+/- {scores.std():.4f})")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: XGBClassifier,
             X_test: pd.DataFrame,
             y_test: np.ndarray,
             class_names: list) -> str:
    """
    Print and return the full classification report.
    Saves confusion matrix plot to models/confusion_matrix.png.
    """
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 62)
    print("  EVALUATION RESULTS  (held-out test set, 20%)")
    print("=" * 62)
    print(f"  Accuracy           : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision (macro)  : {prec:.4f}")
    print(f"  Recall    (macro)  : {rec:.4f}")
    print(f"  F1-score  (macro)  : {f1:.4f}")

    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\n  Per-class report:")
    for line in report.splitlines():
        print(f"    {line}")

    # ── Confusion matrix plot ──────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=25)
    ax.set_title(
        "Confusion Matrix — Industrial Fire Classifier\n"
        "(rows=True label, cols=Predicted label)",
        fontsize=11,
        pad=14,
    )
    plt.tight_layout()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CM_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] Confusion matrix saved --> {CM_FILE.name}")

    return report


# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------

def compute_shap(model: XGBClassifier,
                 X_sample: pd.DataFrame,
                 class_names: list):
    """
    Compute SHAP values using TreeExplainer and save a feature importance plot.
    Uses up to 500 rows for speed on large datasets.

    For multi-class XGBoost, shap_values has shape
    (n_samples, n_features, n_classes). We:
      - Plot a global bar chart (mean |SHAP| across all classes)
      - Print top-10 feature ranking in the console
    """
    print("\n  Computing SHAP values (may take up to ~60 s) ...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    sv = shap_values.values   # shape: (n_samples, n_features, n_classes)

    # ── Global importance bar chart (mean |SHAP| across all classes) ────────
    if sv.ndim == 3:
        mean_abs_shap = np.abs(sv).mean(axis=(0, 2))  # per feature
    else:
        mean_abs_shap = np.abs(sv).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    # Bar chart — clear and works for any number of classes
    fig, ax = plt.subplots(figsize=(10, max(5, len(importance_df) * 0.4)))
    top_n = importance_df.head(20)
    ax.barh(
        top_n["feature"][::-1],
        top_n["mean_abs_shap"][::-1],
        color="#4a90d9",
        edgecolor="none",
    )
    ax.set_xlabel("Mean |SHAP value| (average impact on model output)", fontsize=11)
    ax.set_title(
        "SHAP Feature Importance — Industrial Fire Classifier\n"
        "(mean absolute SHAP value across all classes)",
        fontsize=12,
        pad=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(SHAP_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] SHAP summary plot saved --> {SHAP_FILE.name}")

    # ── Console ranking ──────────────────────────────────────────────────────
    print("\n  Top features by mean |SHAP| value (across all classes):")
    max_val = importance_df["mean_abs_shap"].max()
    for _, row in importance_df.head(10).iterrows():
        bar = "#" * int(row["mean_abs_shap"] / max_val * 30)
        print(f"    {row['feature']:<28}  {row['mean_abs_shap']:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and prepare data ───────────────────────────────────────────
    X, y, class_names, encoders = load_and_prepare(FEATURES_FILE)

    # Guard: need enough samples for a meaningful split
    if len(X) < 10:
        print(
            "\n[WARN] Very few samples in features_labeled.csv.\n"
            "       Complete Steps 1-3 with real FIRMS data first.\n"
            "       Training on available data anyway ..."
        )

    # Check that all classes have enough samples for stratified split
    min_class_count = min(np.bincount(y))
    if min_class_count < 2:
        print(
            "\n[WARN] At least one class has only 1 sample.\n"
            "       Using random split instead of stratified split."
        )
        stratify = None
    else:
        stratify = y

    # ── 2. Train / test split (80/20) ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=stratify,
    )
    print(f"\n  Train samples : {len(X_train):,}")
    print(f"  Test  samples : {len(X_test):,}")

    # ── 3. Cross-validation + build model ─────────────────────────────────
    model = train_model(X_train, y_train, class_names)

    print("\n  Fitting final model on training set ...")
    model.fit(X_train, y_train)
    print("  Done.\n")

    # ── 4. Evaluate on test set ────────────────────────────────────────────
    report = evaluate(model, X_test, y_test, class_names)

    # ── 5. SHAP — use test set (up to 500 rows) ───────────────────────────
    shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    compute_shap(model, shap_sample, class_names)

    # ── 6. Save model bundle ───────────────────────────────────────────────
    bundle = {
        "model":         model,
        "encoders":      encoders,           # target + categorical encoders
        "feature_cols":  list(X.columns),    # exact feature column order
        "class_names":   class_names,
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  [OK] Model saved --> {MODEL_FILE.name}")

    # ── 7. Save text report ────────────────────────────────────────────────
    report_text = (
        f"Industrial Fire Classifier — Model Report\n"
        f"==========================================\n\n"
        f"Features used : {list(X.columns)}\n"
        f"Classes       : {class_names}\n"
        f"Train samples : {len(X_train)}\n"
        f"Test  samples : {len(X_test)}\n\n"
        f"CLASSIFICATION REPORT (test set)\n"
        f"---------------------------------\n"
        + report
        + f"\nCONFUSION MATRIX\n"
        + f"-----------------\n"
        + str(confusion_matrix(y_test, model.predict(X_test)))
        + f"\n\nRows = True labels, Cols = Predicted labels\n"
        + f"Classes: {class_names}\n"
    )
    REPORT_FILE.write_text(report_text, encoding="utf-8")
    print(f"  [OK] Text report saved --> {REPORT_FILE.name}")

    # ── 8. Final summary ───────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  STEP 4 SUMMARY")
    print("=" * 62)
    print(f"  Model     : XGBClassifier (300 trees, depth=5)")
    print(f"  Classes   : {class_names}")
    print(f"  Train/Test: {len(X_train):,} / {len(X_test):,} samples")
    print()
    print("  Output files:")
    print(f"    {MODEL_FILE.name:<30} trained model + encoders")
    print(f"    {CM_FILE.name:<30} confusion matrix plot")
    print(f"    {SHAP_FILE.name:<30} SHAP feature importance plot")
    print(f"    {REPORT_FILE.name:<30} full classification report")
    print("=" * 62)
    print()
    print("[DONE] Model training complete.")
    print("       Review confusion_matrix.png and shap_summary.png in models/")
    print("       Proceed to Step 5:  streamlit run app/dashboard.py")
    print()

    # ── 9. Email alerts ────────────────────────────────────────────────────
    if _ALERTS_AVAILABLE:
        print("\n" + "=" * 62)
        print("  STEP 4b — WILDFIRE RISK EMAIL ALERTS")
        print("=" * 62)
        # Pass the full dataset; send_alerts() handles filtering internally
        try:
            import pandas as _pd
            df_full = _pd.read_csv(DATA_FILE)
            _send_alerts(df=df_full)
        except Exception as _exc:
            print(f"  [Alerts] Could not run alert check: {_exc}")
        print("=" * 62)
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
