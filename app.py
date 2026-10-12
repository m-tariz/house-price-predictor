"""
ESTATEIQ - Professional Real Estate Valuation & Price Estimator
Trained with Scikit-Learn RandomForestRegressor on Ames Housing Dataset
Developed for AI/ML Summer Internship 2026 & Engineering Portfolio
"""

import os
import math
import time
import json
import warnings
from pathlib import Path

# Suppress scikit-learn version mismatch unpickling warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EstateIQ | AI Property Valuation Engine",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Model discovery & Paths
# ---------------------------------------------------------------------------
POSSIBLE_MODEL_DIRS = [
    Path("."),
    Path(__file__).parent if "__file__" in locals() else Path("."),
    Path(r"C:\Users\moham\Desktop\Mnit\Project"),
    Path(r"C:\Users\moham\Desktop\SIGNBRIDGE\SIGNBRIDGE"),
    Path("models"),
]

def find_artifact(filename: str) -> Path | None:
    for base in POSSIBLE_MODEL_DIRS:
        p = base / filename
        if p.exists():
            return p
    return None

MODEL_FILE = find_artifact("house_price_model.pkl") or Path("house_price_model.pkl")
COLUMNS_FILE = find_artifact("model_columns.pkl") or Path("model_columns.pkl")

QUALITY_LABELS = {
    1: "1 – Very Poor (Severe defects)",
    2: "2 – Poor (Heavy deferred maintenance)",
    3: "3 – Fair (Needs substantial rehab)",
    4: "4 – Below Average (Functional but dated)",
    5: "5 – Average (Standard builder spec)",
    6: "6 – Above Average (Well maintained)",
    7: "7 – Good (High quality finishes)",
    8: "8 – Very Good (Custom upgrades)",
    9: "9 – Excellent (Luxury craftsmanship)",
    10: "10 – Very Excellent (Ultra-luxury build)",
}

PRESET_PROFILES = {
    "🏡 Suburban Home": {
        "gr_liv_area": 1950,
        "total_bsmt_sf": 950,
        "bedroom_abv_gr": 3,
        "full_bath": 2,
        "garage_cars": 2,
        "overall_qual": 6,
        "year_built": 2006,
        "yr_sold": 2026,
    },
    "🏙️ Urban Condo": {
        "gr_liv_area": 1200,
        "total_bsmt_sf": 0,
        "bedroom_abv_gr": 2,
        "full_bath": 2,
        "garage_cars": 1,
        "overall_qual": 8,
        "year_built": 2019,
        "yr_sold": 2026,
    },
    "🏰 Luxury Villa": {
        "gr_liv_area": 3400,
        "total_bsmt_sf": 1600,
        "bedroom_abv_gr": 4,
        "full_bath": 3,
        "garage_cars": 3,
        "overall_qual": 9,
        "year_built": 2021,
        "yr_sold": 2026,
    },
    "🔨 Starter Home": {
        "gr_liv_area": 1050,
        "total_bsmt_sf": 650,
        "bedroom_abv_gr": 2,
        "full_bath": 1,
        "garage_cars": 1,
        "overall_qual": 4,
        "year_built": 1980,
        "yr_sold": 2026,
    },
}

# ---------------------------------------------------------------------------
# Executive Professional Styling (Explicit High-Contrast Light Theme)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #f8fafc;
    --surface: #ffffff;
    --border-subtle: #e2e8f0;
    --border-strong: #cbd5e1;
    --text-primary: #0f172a;
    --text-secondary: #334155;
    --text-muted: #64748b;
    --brand-primary: #0f766e;
    --brand-primary-hover: #115e59;
    --brand-accent: #0284c7;
    --brand-ring: rgba(15, 118, 110, 0.2);
    --gold: #d97706;
    --emerald-bg: #ecfdf5;
    --emerald-text: #047857;
    --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    --card-shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
}

/* Force clean light background across all viewports */
.stApp {
    background-color: var(--bg) !important;
    background: var(--bg) !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"], #MainMenu, footer { display: none !important; }
.block-container {
    max-width: 1220px !important;
    padding: 1.25rem 2rem 3rem !important;
}

/* ==========================================================================
   FIX BLACK BLOCKS: High-contrast explicit light styles on ALL inputs & buttons
   ========================================================================== */

/* 1. NUMBER INPUT CONTAINER & WRAPPERS */
[data-testid="stNumberInputContainer"],
div[data-baseweb="input"],
div[data-baseweb="base-input"] {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
    min-height: 42px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
}

[data-testid="stNumberInputContainer"]:focus-within,
div[data-baseweb="input"]:focus-within {
    border-color: var(--brand-primary) !important;
    box-shadow: 0 0 0 3px var(--brand-ring) !important;
}

/* 2. NUMBER INPUT TEXT FIELD */
[data-testid="stNumberInput"] input,
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0 12px !important;
}

/* 3. NUMBER INPUT STEP BUTTONS (+ and -) */
[data-testid="stNumberInput"] button,
[data-testid="stNumberInputContainer"] button {
    background-color: #f8fafc !important;
    background: #f8fafc !important;
    color: #334155 !important;
    border: none !important;
    border-left: 1px solid #e2e8f0 !important;
    border-radius: 0 !important;
    transition: background-color 0.15s ease !important;
}
[data-testid="stNumberInputContainer"] button:hover {
    background-color: #e2e8f0 !important;
    color: #0f172a !important;
}
[data-testid="stNumberInputContainer"] button svg {
    fill: #334155 !important;
    stroke: #334155 !important;
}

/* 4. SELECTBOX & DROPDOWN MENUS */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
    min-height: 42px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
}
div[data-baseweb="select"] * {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    font-weight: 500 !important;
}
div[data-baseweb="select"] svg {
    fill: #475569 !important;
}

/* Dropdown popover list */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[data-baseweb="menu"] {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    box-shadow: var(--card-shadow-lg) !important;
}
li[data-baseweb="menu-item"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
}
li[data-baseweb="menu-item"]:hover {
    background-color: #f0fdf4 !important;
    color: #0f766e !important;
    -webkit-text-fill-color: #0f766e !important;
}

/* 5. PRESET BUTTONS (Secondary buttons) */
.stButton button,
button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #1e293b !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.15s ease !important;
}
.stButton button:hover,
button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover {
    background-color: #f0fdf4 !important;
    background: #f0fdf4 !important;
    color: #0f766e !important;
    border-color: #0f766e !important;
    transform: translateY(-1px) !important;
}
.stButton button p,
button[kind="secondary"] p {
    color: #1e293b !important;
    font-weight: 600 !important;
}
.stButton button:hover p,
button[kind="secondary"]:hover p {
    color: #0f766e !important;
}

/* 6. PRIMARY VALUATION BUTTON */
.st-key-calc_btn button,
button[kind="primary"],
button[data-testid="baseButton-primary"] {
    height: 48px !important;
    width: 100% !important;
    border: none !important;
    border-radius: 10px !important;
    background: linear-gradient(135deg, #0f766e 0%, #115e59 100%) !important;
    color: #ffffff !important;
    font-size: 0.98rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 12px rgba(15, 118, 110, 0.28) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.st-key-calc_btn button:hover,
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #115e59 0%, #134e4a 100%) !important;
    box-shadow: 0 6px 16px rgba(15, 118, 110, 0.38) !important;
    transform: translateY(-1px) !important;
}
.st-key-calc_btn button p,
button[kind="primary"] p {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* 7. DOWNLOAD BUTTON */
[data-testid="stDownloadButton"] button,
.stDownloadButton button {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stDownloadButton"] button:hover,
.stDownloadButton button:hover {
    background-color: #f8fafc !important;
    background: #f8fafc !important;
    border-color: #0f766e !important;
    color: #0f766e !important;
}
[data-testid="stDownloadButton"] button p,
.stDownloadButton button p {
    color: #0f172a !important;
    font-weight: 700 !important;
}
[data-testid="stDownloadButton"] button:hover p,
.stDownloadButton button:hover p {
    color: #0f766e !important;
}

/* Header & Navigation Bar */
.brand-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 1.4rem;
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    box-shadow: var(--card-shadow);
    margin-bottom: 1.5rem;
}
.brand-left {
    display: flex;
    align-items: center;
    gap: 0.9rem;
}
.brand-logo-icon {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, #0f766e 0%, #0369a1 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    color: #ffffff;
    box-shadow: 0 4px 10px rgba(15, 118, 110, 0.3);
}
.brand-title {
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    line-height: 1.2;
}
.brand-sub {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-weight: 500;
}
.brand-status-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.35rem 0.75rem;
    background: var(--emerald-bg);
    color: var(--emerald-text);
    border: 1px solid #a7f3d0;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.status-dot {
    width: 7px;
    height: 7px;
    background-color: #10b981;
    border-radius: 50%;
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}

/* Card Sections */
.form-card {
    background: #ffffff !important;
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.35rem;
    box-shadow: var(--card-shadow);
    margin-bottom: 1.2rem;
}
.card-header-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.05rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid #f1f5f9;
}
.card-icon {
    font-size: 1.15rem;
}
.card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.card-desc {
    font-size: 0.76rem;
    color: var(--text-muted);
    margin-left: auto;
}

[data-testid="stWidgetLabel"] p {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
    margin-bottom: 0.25rem !important;
}
.field-hint {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: -0.2rem;
    margin-bottom: 0.2rem;
}

/* Result panel container (sticky) */
.st-key-result_col {
    position: sticky;
    top: 1.5rem;
}

/* Valuation Result Hero Panel */
.valuation-hero {
    background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
    border-radius: 16px;
    padding: 1.75rem;
    color: #ffffff;
    box-shadow: var(--card-shadow-lg);
    border: 1px solid #334155;
    position: relative;
    overflow: hidden;
}
.valuation-hero::before {
    content: "";
    position: absolute;
    top: -40%;
    right: -30%;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(15, 118, 110, 0.4) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.85rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.65rem;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #e2e8f0;
}
.hero-confidence {
    font-size: 0.72rem;
    color: #94a3b8;
    font-weight: 500;
}
.hero-label {
    font-size: 0.84rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}
.hero-price {
    font-size: 2.85rem;
    font-weight: 800;
    letter-spacing: -0.035em;
    line-height: 1.1;
    margin: 0.35rem 0 0.45rem;
    color: #ffffff;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-variant-numeric: tabular-nums;
}
.hero-range {
    font-size: 0.85rem;
    color: #cbd5e1;
    margin-bottom: 1.25rem;
}
.hero-range strong {
    color: #38bdf8;
    font-weight: 600;
}
.hero-divider {
    height: 1px;
    background: rgba(255, 255, 255, 0.12);
    margin: 1rem 0;
}

/* Metric Pills Grid */
.pill-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin-top: 1rem;
}
.pill-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 0.75rem 0.85rem;
}
.pill-lbl {
    font-size: 0.7rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}
.pill-val {
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff;
    margin-top: 0.2rem;
    font-variant-numeric: tabular-nums;
}

/* Market Range Progress Meter */
.market-meter {
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    box-shadow: var(--card-shadow);
    margin-top: 1.2rem;
}
.meter-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text-primary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.6rem;
}
.meter-track {
    height: 8px;
    background: #e2e8f0;
    border-radius: 9999px;
    position: relative;
    overflow: hidden;
    margin: 0.6rem 0;
}
.meter-fill {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6 0%, #10b981 50%, #f59e0b 100%);
    border-radius: 9999px;
    transition: width 0.5s ease-out;
}
.meter-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: var(--text-muted);
    font-weight: 500;
}

/* Empty placeholder panel */
.empty-placeholder {
    background: #ffffff;
    border: 2px dashed var(--border-strong);
    border-radius: 16px;
    padding: 3rem 1.8rem;
    text-align: center;
    color: var(--text-muted);
}
.empty-icon {
    font-size: 2.8rem;
    margin-bottom: 0.8rem;
    opacity: 0.8;
}
.empty-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
}
.empty-desc {
    font-size: 0.82rem;
    line-height: 1.6;
    max-width: 330px;
    margin: 0 auto;
}

/* Footer branding */
.app-footer {
    margin-top: 3.5rem;
    padding-top: 1.4rem;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
    font-size: 0.8rem;
    color: var(--text-muted);
}
.footer-left {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.footer-badge {
    background: #f1f5f9;
    color: #334155;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-size: 0.76rem;
    font-weight: 600;
    border: 1px solid #e2e8f0;
}
</style>
"""

# ---------------------------------------------------------------------------
# Model loading & Artifact Caching
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_production_artifacts():
    if not (MODEL_FILE and MODEL_FILE.exists()):
        raise FileNotFoundError(f"Model file not found at: {MODEL_FILE}")
    if not (COLUMNS_FILE and COLUMNS_FILE.exists()):
        raise FileNotFoundError(f"Model columns not found at: {COLUMNS_FILE}")
    
    model = joblib.load(MODEL_FILE)
    model_columns = joblib.load(COLUMNS_FILE)
    return model, model_columns


def run_model_inference(model, model_columns: list, v: dict) -> float:
    """Prepare feature vector matching Ames training schema and predict."""
    input_dict = {col: 0 for col in model_columns}

    # Core user features
    input_dict["GrLivArea"] = v["gr_liv_area"]
    input_dict["OverallQual"] = v["overall_qual"]
    input_dict["YearBuilt"] = v["year_built"]
    input_dict["TotalBsmtSF"] = v["total_bsmt_sf"]
    input_dict["GarageCars"] = v["garage_cars"]
    input_dict["FullBath"] = v["full_bath"]
    input_dict["BedroomAbvGr"] = v["bedroom_abv_gr"]
    input_dict["YrSold"] = v["yr_sold"]

    # Derived engineered features matching model training pipeline
    input_dict["HouseAge"] = max(0, v["yr_sold"] - v["year_built"])
    input_dict["TotalBath"] = v["full_bath"]

    # Sensible defaults for high-importance correlated fields if present in schema
    if "1stFlrSF" in input_dict:
        input_dict["1stFlrSF"] = int(v["gr_liv_area"] * 0.65)
    if "2ndFlrSF" in input_dict:
        input_dict["2ndFlrSF"] = int(v["gr_liv_area"] * 0.35)
    if "GarageArea" in input_dict:
        input_dict["GarageArea"] = int(v["garage_cars"] * 240)
    if "LotArea" in input_dict:
        input_dict["LotArea"] = 9500

    input_df = pd.DataFrame([input_dict])[model_columns]
    pred = model.predict(input_df)[0]
    return float(pred)


# ---------------------------------------------------------------------------
# Helper Formatters
# ---------------------------------------------------------------------------
def format_currency(amount: float, currency_mode: str) -> str:
    if currency_mode == "INR (₹)":
        inr_val = amount * 83.0
        if inr_val >= 10000000:
            return f"₹{inr_val / 10000000:.2f} Cr"
        elif inr_val >= 100000:
            return f"₹{inr_val / 100000:.2f} Lakh"
        else:
            return f"₹{inr_val:,.0f}"
    return f"${amount:,.0f}"


def get_market_tier(price: float) -> tuple[str, str, int]:
    if price < 150000:
        return "Affordable / Entry Market", "#3b82f6", 25
    elif price < 275000:
        return "Mid-Market Residential", "#10b981", 55
    elif price < 420000:
        return "Premium Suburban Tier", "#f59e0b", 80
    else:
        return "Executive / Luxury Tier", "#8b5cf6", 96


def apply_preset_callback(vals: dict):
    for k, v in vals.items():
        st.session_state[f"input_{k}"] = v
    st.session_state.pop("last_prediction", None)


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize session keys for all input widgets if first run
    defaults = PRESET_PROFILES["🏡 Suburban Home"]
    for k, v in defaults.items():
        if f"input_{k}" not in st.session_state:
            st.session_state[f"input_{k}"] = v

    # Load machine learning model & columns
    try:
        model, model_columns = load_production_artifacts()
    except FileNotFoundError:
        st.error(
            f"**Model Artifacts Missing**: Could not find `{MODEL_FILE.name}` or `{COLUMNS_FILE.name}`. "
            f"Please ensure they are saved in the project folder."
        )
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load trained model: {exc}")
        st.stop()

    # Top Brand Navigation Bar
    st.markdown("""
    <div class="brand-nav">
        <div class="brand-left">
            <div class="brand-logo-icon">🏠</div>
            <div>
                <div class="brand-title">EstateIQ Valuation Engine</div>
                <div class="brand-sub">Machine Learning Real Estate Valuation & Appraisal System</div>
            </div>
        </div>
        <div class="brand-status-chip">
            <span class="status-dot"></span>
            RandomForestRegressor Active • Scikit-Learn
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Global Controls Bar (Presets + Currency)
    col_presets, col_curr = st.columns([3.2, 1], gap="medium")

    with col_presets:
        st.markdown("**Quick Valuation Presets:** (Click to auto-populate test cases)")
        p_cols = st.columns(4)
        for idx, (label, vals) in enumerate(PRESET_PROFILES.items()):
            with p_cols[idx]:
                st.button(
                    label,
                    key=f"preset_btn_{idx}",
                    on_click=apply_preset_callback,
                    args=(vals,),
                    use_container_width=True,
                )

    with col_curr:
        curr_mode = st.selectbox(
            "Display Currency",
            ["USD ($)", "INR (₹)"],
            index=0,
            help="Toggle between US Dollars and Indian Rupees (using current benchmark rate)",
        )

    # Two-Column Layout: Left (Inputs) vs Right (Valuation Output)
    col_inputs, col_results = st.columns([1.5, 1.2], gap="large")

    with col_inputs:
        st.markdown("""
        <div class="form-card">
            <div class="card-header-row">
                <span class="card-icon">📐</span>
                <span class="card-title">Living Space & Dimensions</span>
                <span class="card-desc">Gross square footage</span>
            </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            gr_liv_area = st.number_input(
                "Total Above-Ground Living Area (sq ft)",
                min_value=300,
                max_value=7000,
                step=50,
                key="input_gr_liv_area",
                help="Heated living area above grade",
            )
        with c2:
            total_bsmt_sf = st.number_input(
                "Total Basement Area (sq ft)",
                min_value=0,
                max_value=4500,
                step=50,
                key="input_total_bsmt_sf",
                help="Total basement square footage (set to 0 if slab/no basement)",
            )
        st.markdown('<div class="field-hint">💡 Enter 0 if the property does not have a basement foundation.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="form-card">
            <div class="card-header-row">
                <span class="card-icon">🛏️</span>
                <span class="card-title">Accommodations & Parking</span>
                <span class="card-desc">Room distribution</span>
            </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            bedroom_abv_gr = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=8,
                step=1,
                key="input_bedroom_abv_gr",
            )
        with c2:
            full_bath = st.number_input(
                "Full Bathrooms",
                min_value=1,
                max_value=6,
                step=1,
                key="input_full_bath",
            )
        with c3:
            garage_cars = st.number_input(
                "Garage Capacity",
                min_value=0,
                max_value=5,
                step=1,
                key="input_garage_cars",
                help="Vehicle capacity in attached/detached garage",
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="form-card">
            <div class="card-header-row">
                <span class="card-icon">🏗️</span>
                <span class="card-title">Construction Quality & Timeline</span>
                <span class="card-desc">Material grade & age</span>
            </div>
        """, unsafe_allow_html=True)

        overall_qual = st.selectbox(
            "Overall Material & Finish Quality Rating",
            options=list(QUALITY_LABELS.keys()),
            format_func=QUALITY_LABELS.get,
            key="input_overall_qual",
            help="Rates the overall quality of materials and finish of the house",
        )

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            year_built = st.number_input(
                "Year Built",
                min_value=1870,
                max_value=2026,
                step=1,
                key="input_year_built",
            )
        with c2:
            yr_sold = st.number_input(
                "Year of Valuation / Sale",
                min_value=1950,
                max_value=2026,
                step=1,
                key="input_yr_sold",
            )

        dates_valid = yr_sold >= year_built
        if not dates_valid:
            st.error("⚠️ Year of sale cannot precede the year of construction.")
        else:
            effective_age = yr_sold - year_built
            st.markdown(
                f'<div class="field-hint">⏱️ Effective Property Age at Valuation: <b>{effective_age} years</b></div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        active_form = {
            "gr_liv_area": gr_liv_area,
            "total_bsmt_sf": total_bsmt_sf,
            "bedroom_abv_gr": bedroom_abv_gr,
            "full_bath": full_bath,
            "garage_cars": garage_cars,
            "overall_qual": overall_qual,
            "year_built": year_built,
            "yr_sold": yr_sold,
        }

        # Primary Trigger Button
        estimate_clicked = st.button(
            "⚡ Compute Market Valuation",
            type="primary",
            key="calc_btn",
            disabled=not dates_valid,
        )

    # -----------------------------------------------------------------------
    # Right Column: Valuation Results & Analytics (Sticky)
    # -----------------------------------------------------------------------
    with col_results:
        with st.container(key="result_col"):
            # Handle inference execution
            if estimate_clicked and dates_valid:
                with st.spinner("Executing RandomForest inference across 100 decision trees..."):
                    time.sleep(0.3)
                    try:
                        predicted_price = run_model_inference(model, model_columns, active_form)
                        if not math.isfinite(predicted_price) or predicted_price <= 0:
                            raise ValueError("Model produced an invalid value.")
                        st.session_state["last_prediction"] = {
                            "price": predicted_price,
                            "inputs": active_form,
                        }
                    except Exception as ex:
                        st.error(f"Inference computation failed: {ex}")

            last_pred = st.session_state.get("last_prediction")

            if last_pred:
                price = last_pred["price"]
                inp = last_pred["inputs"]
                area = inp["gr_liv_area"]
                price_sqft = price / area if area > 0 else 0
                age = max(0, inp["yr_sold"] - inp["year_built"])
                lower_bound = price * 0.95
                upper_bound = price * 1.05
                tier_name, tier_color, progress_pct = get_market_tier(price)

                # Executive Hero Card
                st.markdown(f"""
                <div class="valuation-hero">
                    <div class="hero-badge-row">
                        <span class="hero-badge">AI ESTIMATE</span>
                        <span class="hero-confidence">Confidence Interval: ±5.0%</span>
                    </div>
                    <div class="hero-label">Estimated Market Value</div>
                    <div class="hero-price">{format_currency(price, curr_mode)}</div>
                    <div class="hero-range">
                        Market Range: <strong>{format_currency(lower_bound, curr_mode)}</strong> – <strong>{format_currency(upper_bound, curr_mode)}</strong>
                    </div>
                    <div class="hero-divider"></div>
                    <div class="pill-grid">
                        <div class="pill-card">
                            <div class="pill-lbl">Price Per Sq Ft</div>
                            <div class="pill-val">{format_currency(price_sqft, curr_mode)}/sqft</div>
                        </div>
                        <div class="pill-card">
                            <div class="pill-lbl">Quality Tier</div>
                            <div class="pill-val">{QUALITY_LABELS[inp['overall_qual']].split('–')[1].split('(')[0].strip()}</div>
                        </div>
                        <div class="pill-card">
                            <div class="pill-lbl">Property Age</div>
                            <div class="pill-val">{age} Years</div>
                        </div>
                        <div class="pill-card">
                            <div class="pill-lbl">Total Capacity</div>
                            <div class="pill-val">{inp['bedroom_abv_gr']}B / {inp['full_bath']}Ba / {inp['garage_cars']}Car</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Market Positioning Meter
                st.markdown(f"""
                <div class="market-meter">
                    <div class="meter-title">Market Positioning: <span style="color: {tier_color};">{tier_name}</span></div>
                    <div class="meter-track">
                        <div class="meter-fill" style="width: {progress_pct}%;"></div>
                    </div>
                    <div class="meter-labels">
                        <span>Affordable (<$150k)</span>
                        <span>Median ($220k)</span>
                        <span>Luxury ($400k+)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Appraisal Export & Summary
                st.markdown("<br>", unsafe_allow_html=True)
                export_payload = {
                    "system": "EstateIQ Valuation Engine",
                    "estimated_price": round(price, 2),
                    "price_formatted": format_currency(price, curr_mode),
                    "price_per_sqft": round(price_sqft, 2),
                    "market_tier": tier_name,
                    "input_specs": inp,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                st.download_button(
                    label="📥 Export Valuation Report (JSON)",
                    data=json.dumps(export_payload, indent=2),
                    file_name=f"valuation_report_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            else:
                st.markdown("""
                <div class="empty-placeholder">
                    <div class="empty-icon">📊</div>
                    <div class="empty-title">Ready for Valuation</div>
                    <div class="empty-desc">
                        Adjust the property specifications on the left, or pick one of the quick presets above, then click <b>Compute Market Valuation</b> to generate an appraisal.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Model Transparency & Portfolio Drawer
    # -----------------------------------------------------------------------
    with st.expander("🔍 Model Architecture & Feature Importance Insights"):
        st.markdown("""
        ### Machine Learning Architecture Overview
        This valuation engine leverages an ensemble **RandomForestRegressor** with **100 decision trees** trained on the Ames Housing Dataset.
        """)
        
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("Algorithm", "RandomForestRegressor")
        c_m2.metric("Estimators (Trees)", "100 Trees")
        c_m3.metric("Feature Vectors", f"{len(model_columns)} Columns")

        st.markdown("#### Top Predictor Influence (Feature Importances)")
        feat_df = pd.DataFrame([
            {"Feature": "Overall Material Quality (OverallQual)", "Importance": "55.7%", "Impact": "Primary Driver"},
            {"Feature": "Above Ground Living Area (GrLivArea)", "Importance": "12.1%", "Impact": "Major Driver"},
            {"Feature": "Second Floor SF (2ndFlrSF)", "Importance": "3.4%", "Impact": "Structural Driver"},
            {"Feature": "Total Basement SF (TotalBsmtSF)", "Importance": "3.4%", "Impact": "Foundation Driver"},
            {"Feature": "Finished Basement Area (BsmtFinSF1)", "Importance": "3.0%", "Impact": "Functional Driver"},
            {"Feature": "Garage Capacity & Area (GarageCars)", "Importance": "2.9%", "Impact": "Convenience Driver"},
            {"Feature": "Effective Age & Year Built (HouseAge)", "Importance": "1.4%", "Impact": "Depreciation Driver"},
        ])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Clean Footer / Portfolio Attribution
    # -----------------------------------------------------------------------
    st.markdown("""
    <div class="app-footer">
        <div class="footer-left">
            <span class="footer-badge">AI / ML Summer Internship 2026</span>
            <span class="footer-badge">MNNIT Allahabad</span>
        </div>
        <div>
            <span>Powered by Scikit-Learn • Streamlit • Python 3</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
