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
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
    learning_curve,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
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
LC_FILE       = MODEL_DIR / "learning_curve.png"

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
    "scan",              # VIIRS scan pixel size
    "track",             # VIIRS track pixel size
    "confidence_enc",    # encoded detection confidence (l=0, n=1, h=2)
    "delta_brightness",  # bright_ti4 - bright_ti5 (genuine fire signal)
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

    Pipeline:
      1. Quick 5-fold CV on a baseline model to report generalisation.
      2. RandomizedSearchCV over a small hyperparameter grid (10 combos x 3-fold)
         to find robust parameters without overfitting the 475-row dataset.
      3. Fit the best estimator on the full training set.

    Class imbalance (~54% Anomaly vs ~9% Wildfire) is handled by passing
    per-sample weights derived from class frequencies.
    """
    n_classes = len(class_names)

    # ── Base model for CV reference ─────────────────────────────────────────────
    base_model = XGBClassifier(
        n_estimators     = 200,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        gamma            = 0.3,
        reg_lambda       = 2.0,      # L2 regularisation — key to prevent memorisation
        reg_alpha        = 0.1,      # L1 regularisation
        objective        = "multi:softprob",
        num_class        = n_classes,
        eval_metric      = "mlogloss",
        use_label_encoder= False,
        random_state     = 42,
        n_jobs           = -1,
    )

    # 5-fold cross-validation reference
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n  Running 5-fold CV on baseline model ...")
    for metric_name, scoring in [("Accuracy", "accuracy"), ("F1-macro", "f1_macro")]:
        scores = cross_val_score(base_model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        print(f"    CV {metric_name:<12}: {scores.mean():.4f}  (+/- {scores.std():.4f})")

    # ── RandomizedSearchCV ───────────────────────────────────────────────
    param_dist = {
        "n_estimators"    : [100, 150, 200, 300],
        "max_depth"       : [3, 4, 5],
        "learning_rate"   : [0.03, 0.05, 0.08, 0.1],
        "subsample"       : [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [3, 5, 7],
        "gamma"           : [0.1, 0.3, 0.5],
        "reg_lambda"      : [1.0, 2.0, 3.0],
    }

    search_model = XGBClassifier(
        objective         = "multi:softprob",
        num_class         = n_classes,
        eval_metric       = "mlogloss",
        use_label_encoder = False,
        random_state      = 42,
        n_jobs            = -1,
    )

    print("\n  Running RandomizedSearchCV (10 combos x 3-fold) ...")
    rscv = RandomizedSearchCV(
        estimator          = search_model,
        param_distributions= param_dist,
        n_iter             = 10,
        cv                 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring            = "f1_macro",
        refit              = True,
        random_state       = 42,
        n_jobs             = -1,
        verbose            = 0,
    )
    rscv.fit(X, y)
    print(f"    Best F1-macro (CV): {rscv.best_score_:.4f}")
    print(f"    Best params: {rscv.best_params_}")

    return rscv.best_estimator_


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

    acc      = accuracy_score(y_test, y_pred)
    bal_acc  = balanced_accuracy_score(y_test, y_pred)
    prec     = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec      = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1       = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 62)
    print("  EVALUATION RESULTS  (held-out test set, 20%)")
    print("=" * 62)
    print(f"  Accuracy           : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Balanced Accuracy  : {bal_acc:.4f}  ({bal_acc*100:.2f}%)")
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

    # ── Confusion matrix plot ─────────────────────────────────────────────
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
# Learning curve
# ---------------------------------------------------------------------------

def plot_learning_curve(model: XGBClassifier, X: pd.DataFrame, y: np.ndarray):
    """
    Generate and save a learning curve (train vs CV accuracy as training
    size increases). This visually confirms the model is not simply
    memorising the data: a well-generalising model shows train and CV
    curves converging as more data is added.
    """
    print("\n  Plotting learning curve ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.2, 1.0, 8),
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_mean, "o-", color="#2196F3", label="Training F1")
    ax.fill_between(train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.15, color="#2196F3")
    ax.plot(train_sizes, val_mean, "s-", color="#FF5722", label="CV F1 (macro)")
    ax.fill_between(train_sizes,
                    val_mean - val_std,
                    val_mean + val_std,
                    alpha=0.15, color="#FF5722")
    ax.set_xlabel("Training samples", fontsize=11)
    ax.set_ylabel("F1-score (macro)", fontsize=11)
    ax.set_title(
        "Learning Curve — Industrial Fire Classifier\n"
        "(Training vs Cross-validation F1-macro)",
        fontsize=12, pad=12,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(LC_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Learning curve saved --> {LC_FILE.name}")


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

    # ── 3. Cross-validation + hyperparameter search ───────────────────────
    model = train_model(X_train, y_train, class_names)

    # ── 4. Fit final model with class-weighted samples ─────────────────────
    # Compensate for heavy class imbalance (~54% Anomaly, ~9% Wildfire)
    print("\n  Fitting final model on training set (with class weights) ...")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    print("  Done.\n")

    # ── 5. Evaluate on test set ─────────────────────────────────────────
    report = evaluate(model, X_test, y_test, class_names)

    # ── 6. Learning curve ──────────────────────────────────────────────
    plot_learning_curve(model, X, y)

    # ── 7. SHAP — use test set (up to 500 rows) ─────────────────────────
    shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    compute_shap(model, shap_sample, class_names)

    # ── 8. Save model bundle ───────────────────────────────────────────────
    bundle = {
        "model":         model,
        "encoders":      encoders,           # target + categorical encoders
        "feature_cols":  list(X.columns),    # exact feature column order
        "class_names":   class_names,
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  [OK] Model saved --> {MODEL_FILE.name}")

    # ── 9. Save text report ────────────────────────────────────────────────
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

    # ── 10. Final summary ─────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  STEP 4 SUMMARY")
    print("=" * 62)
    print(f"  Model     : XGBClassifier (RandomizedSearchCV tuned)")
    print(f"  Classes   : {class_names}")
    print(f"  Train/Test: {len(X_train):,} / {len(X_test):,} samples")
    print()
    print("  Output files:")
    print(f"    {MODEL_FILE.name:<30} trained model + encoders")
    print(f"    {CM_FILE.name:<30} confusion matrix plot")
    print(f"    {SHAP_FILE.name:<30} SHAP feature importance plot")
    print(f"    {LC_FILE.name:<30} learning curve plot")
    print(f"    {REPORT_FILE.name:<30} full classification report")
    print("=" * 62)
    print()
    print("[DONE] Model training complete.")
    print("       Review confusion_matrix.png and shap_summary.png in models/")
    print("       Proceed to Step 5:  streamlit run app/dashboard.py")
    print()

    # ── 11. Email alerts ───────────────────────────────────────────────
    if _ALERTS_AVAILABLE:
        print("\n" + "=" * 62)
        print("  STEP 4b — WILDFIRE RISK EMAIL ALERTS")
        print("=" * 62)
        # Pass the full dataset; send_alerts() handles filtering internally
        try:
            import pandas as _pd
            df_full = _pd.read_csv(FEATURES_FILE)  # was DATA_FILE — fixed
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
