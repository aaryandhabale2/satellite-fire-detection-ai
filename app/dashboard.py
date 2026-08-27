"""
app/dashboard.py  —  STEP 5
------------------------------
Streamlit dashboard for the AI-Based Industrial Fire Detection project.

Run with:
    streamlit run app/dashboard.py

Loads:
    data/features_labeled.csv     -- hotspot data with heuristic labels
    models/fire_classifier.pkl    -- trained XGBoost model (for predictions)
"""

import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Fire Detection — India",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[1]
DATA_FILE    = ROOT / "data"   / "features_labeled.csv"
MODEL_FILE   = ROOT / "models" / "fire_classifier.pkl"
CM_FILE      = ROOT / "models" / "confusion_matrix.png"
SHAP_FILE    = ROOT / "models" / "shap_summary.png"
REPORT_FILE  = ROOT / "models" / "model_report.txt"

# ---------------------------------------------------------------------------
# Category colour scheme
# Folium color names  +  hex colours for Plotly / CSS
# ---------------------------------------------------------------------------
CATEGORY_FOLIUM_COLOR = {
    "Wildfire Risk":        "red",
    "Agricultural Burning": "orange",
    "Industrial (Normal)":  "gray",
    "Anomaly/Unclassified": "beige",
}

CATEGORY_HEX = {
    "Wildfire Risk":        "#ef4444",
    "Agricultural Burning": "#f97316",
    "Industrial (Normal)":  "#6b7280",
    "Anomaly/Unclassified": "#eab308",
}

# Priority order for alerts (Wildfire Risk is the most urgent)
ALERT_PRIORITY = [
    "Wildfire Risk",
    "Agricultural Burning",
    "Anomaly/Unclassified",
    "Industrial (Normal)",
]

# ---------------------------------------------------------------------------
# CSS — clean, professional dark-ish theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] h2 { color: #1e293b; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
}

/* Alert card */
.alert-card {
    background: #fef2f2;
    border-left: 4px solid #ef4444;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.875rem;
    line-height: 1.6;
}
.alert-rank {
    font-weight: 700;
    color: #dc2626;
    font-size: 0.9rem;
}

/* Category badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: white;
}
.badge-wildfire   { background: #ef4444; }
.badge-agri       { background: #f97316; }
.badge-industrial { background: #6b7280; }
.badge-anomaly    { background: #eab308; color: #1e293b; }

/* Section divider */
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #334155;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px;
    margin: 8px 0 14px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data & model loaders — cached so they only run once
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading hotspot data...")
def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_FILE)
    if "acq_date" in df.columns:
        df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    return df


@st.cache_resource(show_spinner="Loading model...")
def load_model() -> dict | None:
    if not MODEL_FILE.exists():
        return None
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


def run_predictions(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Apply the trained model to add a 'predicted_category' column."""
    model      = bundle["model"]
    encoders   = bundle["encoders"]
    feat_cols  = bundle["feature_cols"]
    class_names= bundle["class_names"]
    target_le  = encoders["target"]

    # Re-encode categorical columns using saved encoders
    df = df.copy()
    for col, le in encoders.items():
        if col == "target":
            continue
        enc_col = f"{col}_enc"
        if col in df.columns and enc_col not in df.columns:
            df[enc_col] = le.transform(
                df[col].fillna("unknown").astype(str).map(
                    lambda x, le=le: x if x in le.classes_ else le.classes_[0]
                )
            )

    available = [c for c in feat_cols if c in df.columns]
    X = df[available].fillna(0).astype(float)

    preds = model.predict(X)
    df["predicted_category"] = target_le.inverse_transform(preds)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_category_col(df: pd.DataFrame) -> str:
    """Return 'predicted_category' if available, else fall back to 'category'."""
    if "predicted_category" in df.columns:
        return "predicted_category"
    return "category"


def make_folium_map(df: pd.DataFrame, cat_col: str) -> folium.Map:
    """
    Build a Folium map of India with hotspots clustered and
    colour-coded by category.  Limits markers to 5,000 for speed.
    """
    m = folium.Map(
        location=[22.5, 82.0],
        zoom_start=5,
        tiles="OpenStreetMap",
    )

    # Sample if dataset is large
    plot_df = df.dropna(subset=["latitude", "longitude"])
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(5000, random_state=42)

    cluster = MarkerCluster(
        name="Hotspots",
        options={"maxClusterRadius": 40, "disableClusteringAtZoom": 10},
    ).add_to(m)

    for _, row in plot_df.iterrows():
        cat   = str(row.get(cat_col, "Anomaly/Unclassified"))
        color = CATEGORY_FOLIUM_COLOR.get(cat, "beige")
        frp   = row.get("frp", "?")
        lu    = row.get("land_use_type", "unknown")
        date  = str(row.get("acq_date", ""))[:10]
        freq  = row.get("historical_frequency", "?")

        try:
            radius = max(4, min(14, float(frp) ** 0.45))
        except (TypeError, ValueError):
            radius = 5

        popup_html = f"""
        <div style="font-family:Inter,sans-serif;font-size:13px;min-width:190px">
            <b style="color:{CATEGORY_HEX.get(cat,'#555')}">{cat}</b><hr style="margin:4px 0">
            🌡️ FRP: <b>{frp} MW</b><br>
            🏞️ Land use: <b>{lu}</b><br>
            📅 Date: <b>{date}</b><br>
            🔁 Repeat count: <b>{freq}</b><br>
            📍 ({row['latitude']:.4f}, {row['longitude']:.4f})
        </div>"""

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=cat,
        ).add_to(cluster)

    return m


# ---------------------------------------------------------------------------
# ── MAIN APP ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 8px 0 4px">
  <h1 style="font-size:1.9rem;font-weight:700;color:#0f172a;margin-bottom:4px">
    🔥 AI Industrial Fire Detection — India
  </h1>
  <p style="color:#64748b;font-size:0.92rem;margin:0">
    Classifying NASA FIRMS satellite hotspots using OpenStreetMap context
    and an XGBoost model into <b>Wildfire Risk</b>, <b>Agricultural Burning</b>,
    <b>Industrial (Normal)</b>, and <b>Anomaly/Unclassified</b>.
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Load data & model ──────────────────────────────────────────────────────
df_raw       = load_data()
model_bundle = load_model()

# No data → show setup instructions and stop
if df_raw.empty:
    st.warning(
        "**No data found.** Run the full pipeline first:\n\n"
        "```\n"
        "python src/fetch_firms_data.py       # Step 1\n"
        "python src/fetch_osm_landuse.py      # Step 2\n"
        "python src/build_features.py         # Step 3\n"
        "python src/train_model.py            # Step 4\n"
        "```",
        icon="⚠️",
    )
    st.stop()

# Apply model predictions if model is available
if model_bundle is not None:
    try:
        df_raw = run_predictions(df_raw, model_bundle)
        model_ok = True
    except Exception as e:
        st.warning(f"Model prediction failed ({e}). Showing heuristic labels.")
        model_ok = False
else:
    model_ok = False

cat_col = get_category_col(df_raw)

# ── Sidebar — filters ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Filters")

    # Date range filter
    if "acq_date" in df_raw.columns and df_raw["acq_date"].notna().any():
        min_date = df_raw["acq_date"].min().date()
        max_date = df_raw["acq_date"].max().date()

        st.markdown("**Date Range**")
        date_range = st.date_input(
            "Select range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            label_visibility="collapsed",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            d_start = pd.Timestamp(date_range[0])
            d_end   = pd.Timestamp(date_range[1])
            df = df_raw[df_raw["acq_date"].between(d_start, d_end)].copy()
        else:
            df = df_raw.copy()
    else:
        df = df_raw.copy()
        st.info("No date column found.")

    st.markdown("---")

    # Category filter
    st.markdown("**Category**")
    all_cats = sorted(df[cat_col].dropna().unique().tolist())
    sel_cats = st.multiselect(
        "Select categories",
        options=all_cats,
        default=all_cats,
        label_visibility="collapsed",
    )
    if sel_cats:
        df = df[df[cat_col].isin(sel_cats)]

    st.markdown("---")
    st.markdown(f"**Showing `{len(df):,}` hotspots**")
    st.markdown("---")

    # Legend
    st.markdown("**Legend**")
    for cat, hex_col in CATEGORY_HEX.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">'
            f'<div style="width:14px;height:14px;border-radius:50%;'
            f'background:{hex_col};flex-shrink:0"></div>'
            f'<span style="font-size:0.85rem">{cat}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    # Model status
    if model_ok:
        st.success("✅ Model predictions active")
    else:
        st.warning("⚠️ Using heuristic labels\n(run train_model.py)")

# ── KPI metrics row ────────────────────────────────────────────────────────
total  = len(df)
n_wild = (df[cat_col] == "Wildfire Risk").sum()
n_agri = (df[cat_col] == "Agricultural Burning").sum()
n_ind  = (df[cat_col] == "Industrial (Normal)").sum()
n_anom = (df[cat_col] == "Anomaly/Unclassified").sum()
avg_frp = df["frp"].mean() if "frp" in df.columns else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🔢 Total Hotspots",    f"{total:,}")
c2.metric("🔴 Wildfire Risk",     f"{n_wild:,}",
          delta=f"{n_wild/total*100:.1f}%" if total else None)
c3.metric("🟠 Agri Burning",      f"{n_agri:,}",
          delta=f"{n_agri/total*100:.1f}%" if total else None)
c4.metric("⚫ Industrial Normal", f"{n_ind:,}",
          delta=f"{n_ind/total*100:.1f}%" if total else None)
c5.metric("🟡 Anomaly",           f"{n_anom:,}",
          delta=f"{n_anom/total*100:.1f}%" if total else None)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────
tab_map, tab_trend, tab_alerts, tab_model = st.tabs([
    "🗺️  Live Map",
    "📈  Trend Chart",
    "🚨  Alerts",
    "🤖  Model Info",
])

# ===========================================================================
# TAB 1 — LIVE MAP
# ===========================================================================
with tab_map:
    st.markdown('<div class="section-title">Hotspot Map — India</div>',
                unsafe_allow_html=True)

    col_info, col_tip = st.columns([3, 1])
    with col_info:
        st.caption(
            "Hotspots are clustered for performance. "
            "Zoom in to see individual markers. "
            "Click a marker for details."
        )
    with col_tip:
        if len(df) > 5000:
            st.caption(f"⚡ Showing 5,000 of {len(df):,} hotspots (sampled)")

    if df.empty:
        st.info("No hotspots match the current filters.")
    else:
        fmap = make_folium_map(df, cat_col)
        st_folium(fmap, width="100%", height=520, returned_objects=[])

# ===========================================================================
# TAB 2 — TREND CHART
# ===========================================================================
with tab_trend:
    st.markdown('<div class="section-title">Hotspot Frequency Over Time</div>',
                unsafe_allow_html=True)

    if "acq_date" not in df.columns or df["acq_date"].isna().all():
        st.info("No date information available in the current dataset.")
    elif df.empty:
        st.info("No data matches the current filters.")
    else:
        # Daily counts by category
        daily = (
            df.groupby(["acq_date", cat_col])
            .size()
            .reset_index(name="count")
            .rename(columns={cat_col: "Category"})
        )

        fig = px.line(
            daily,
            x="acq_date",
            y="count",
            color="Category",
            color_discrete_map=CATEGORY_HEX,
            markers=True,
            labels={
                "acq_date": "Date",
                "count":    "Number of Hotspots",
                "Category": "Category",
            },
            title="Daily Hotspot Detections by Category",
        )
        fig.update_layout(
            hovermode="x unified",
            legend_title_text="Category",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_family="Inter",
            title_font_size=15,
            margin=dict(t=50, b=40),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Category distribution bar chart
        cat_totals = df[cat_col].value_counts().reset_index()
        cat_totals.columns = ["Category", "Count"]
        fig2 = px.bar(
            cat_totals,
            x="Category",
            y="Count",
            color="Category",
            color_discrete_map=CATEGORY_HEX,
            title="Total Hotspots by Category (filtered period)",
            text="Count",
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_family="Inter",
            title_font_size=15,
            xaxis_title="",
            margin=dict(t=50, b=20),
        )
        fig2.update_xaxes(showgrid=False)
        fig2.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig2, use_container_width=True)

# ===========================================================================
# TAB 3 — ALERTS PANEL
# ===========================================================================
with tab_alerts:
    st.markdown('<div class="section-title">🚨 Top 10 Wildfire Risk Alerts</div>',
                unsafe_allow_html=True)

    wildfire_df = df[df[cat_col] == "Wildfire Risk"].copy()

    if wildfire_df.empty:
        st.success(
            "✅ No active Wildfire Risk detections in the current filter.\n\n"
            "Adjust the date range or category filters to see more data."
        )
    else:
        st.markdown(
            f"**{len(wildfire_df):,} Wildfire Risk detection(s)** in the "
            f"current filter — top 10 by FRP intensity:"
        )
        st.markdown("")

        # Sort by FRP descending — highest intensity first
        if "frp" in wildfire_df.columns:
            top10 = wildfire_df.nlargest(10, "frp")
        else:
            top10 = wildfire_df.head(10)

        for rank, (_, row) in enumerate(top10.iterrows(), start=1):
            lat   = row.get("latitude",   "?")
            lon   = row.get("longitude",  "?")
            frp   = row.get("frp",        "?")
            date  = str(row.get("acq_date", ""))[:10]
            lu    = row.get("land_use_type", "unknown")
            freq  = row.get("historical_frequency", 0)
            tod   = row.get("time_of_day", "?")
            season= row.get("season", "?")

            # Format FRP for display
            try:
                frp_str = f"{float(frp):.1f} MW"
            except (TypeError, ValueError):
                frp_str = str(frp)

            # Format coordinates
            try:
                coord_str = f"{float(lat):.4f}°N, {float(lon):.4f}°E"
            except (TypeError, ValueError):
                coord_str = f"{lat}, {lon}"

            st.markdown(
                f'<div class="alert-card">'
                f'<span class="alert-rank">#{rank} — FRP: {frp_str}</span>'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;📅 {date}'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;🕐 {tod}<br>'
                f'📍 {coord_str}'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;🏞️ Land use: <b>{lu}</b>'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;Season: {season}'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;🔁 Repeat count: <b>{int(freq) if str(freq).replace(".","").isdigit() else freq}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.caption(
            "ℹ️ FRP = Fire Radiative Power (MW). Higher FRP = more intense fire. "
            "Repeat count = how many times this ~1 km location was detected as a hotspot."
        )

# ===========================================================================
# TAB 4 — MODEL INFO
# ===========================================================================
with tab_model:
    st.markdown('<div class="section-title">Model Performance & Explainability</div>',
                unsafe_allow_html=True)

    if not model_ok:
        st.warning(
            "**Model not loaded.** Run Step 4 first:\n"
            "```\npython src/train_model.py\n```"
        )
    else:
        col_r, col_cm = st.columns(2)

        with col_r:
            st.markdown("**Classification Report**")
            if REPORT_FILE.exists():
                st.code(REPORT_FILE.read_text(encoding="utf-8"), language="text")
            else:
                st.info("model_report.txt not found.")

        with col_cm:
            st.markdown("**Confusion Matrix**")
            if CM_FILE.exists():
                st.image(str(CM_FILE), use_column_width=True)
            else:
                st.info("confusion_matrix.png not found.")

        st.markdown("---")

        st.markdown("**SHAP Feature Importance**")
        if SHAP_FILE.exists():
            st.image(str(SHAP_FILE), use_column_width=True)
            st.caption(
                "Each bar shows the mean absolute SHAP value for that feature "
                "across all classes — a measure of how much it drives predictions."
            )
        else:
            st.info("shap_summary.png not found.")

    st.markdown("---")
    with st.expander("ℹ️ About this model"):
        st.markdown("""
        **Algorithm:** XGBoost multi-class classifier (300 trees, max depth 5)

        **Classes:**
        - 🔴 **Wildfire Risk** — sudden high-intensity fire on forest/farmland
        - 🟠 **Agricultural Burning** — seasonal stubble burning (Oct–Nov)
        - ⚫ **Industrial (Normal)** — persistent industrial thermal source
        - 🟡 **Anomaly/Unclassified** — doesn't fit a clear pattern

        **Features used:**
        brightness, FRP, log(FRP), land use type, distance to industrial zone,
        historical frequency, time of day, season, month

        **Training:** 80/20 stratified split with 5-fold cross-validation

        **Labels:** Heuristic rules applied to the OSM-enriched FIRMS data.
        Replace with verified ground-truth for production use.
        """)

    # Quick stats
    if not df_raw.empty and cat_col in df_raw.columns:
        st.markdown("---")
        st.markdown("**Full dataset category breakdown**")
        summary = df_raw[cat_col].value_counts().reset_index()
        summary.columns = ["Category", "Count"]
        summary["Percentage"] = (summary["Count"] / len(df_raw) * 100).round(1)
        st.dataframe(summary, hide_index=True, use_container_width=True)
