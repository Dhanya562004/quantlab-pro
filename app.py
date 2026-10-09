"""
QuantLab Pro: Multi-Agent Quantitative ML Research Platform
Streamlit Main Interactive Application.

Ultra-Premium Institutional Quantitative Terminal Design System.
Aligned to Tower Research Capital's Intern - AI/ML Job Description.
"""

import json
import time
from typing import Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Import QuantLab Pro Modules
from quantlab.data.synthetic import generate_synthetic_ohlcv, compute_dataset_fingerprint
from quantlab.data.loader import load_csv_dataset, load_yfinance_dataset
from quantlab.data.validation import validate_ohlcv_data, ValidationReport
from quantlab.features.builder import FeatureConfig, build_features_and_target, create_chronological_splits
from quantlab.models.trainer import train_model
from quantlab.evaluation.metrics import compute_classification_metrics
from quantlab.evaluation.backtest import run_research_backtest
from quantlab.orchestration.engine import MultiAgentOrchestrator
from quantlab.tools.registry import ToolRegistry
from quantlab.storage.db import ExperimentStorage

# --- Page Configuration ---
st.set_page_config(
    page_title="QuantLab Pro | Institutional ML Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Ultra-Premium Dark Purple & Obsidian Aesthetic CSS ---
ADVANCED_STYLING_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global Body & Background */
    .stApp {
        background: #040208;
        background-image: 
            radial-gradient(at 15% 15%, rgba(88, 28, 135, 0.25) 0px, transparent 50%),
            radial-gradient(at 85% 20%, rgba(168, 85, 247, 0.15) 0px, transparent 50%),
            radial-gradient(at 50% 80%, rgba(14, 116, 144, 0.15) 0px, transparent 50%);
        color: #f3e8ff;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #090412;
        border-right: 1px solid rgba(168, 85, 247, 0.2);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.6);
    }
    
    /* Neon Gradient Text Header */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #e9d5ff 40%, #c084fc 70%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    
    .hero-subtitle {
        color: #a855f7;
        font-size: 1.15rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        margin-bottom: 24px;
    }

    /* Premium Glassmorphism Container */
    .glass-panel {
        background: rgba(14, 6, 28, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(168, 85, 247, 0.25);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.45);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-panel:hover {
        border-color: rgba(192, 132, 252, 0.5);
        box-shadow: 0 16px 48px 0 rgba(168, 85, 247, 0.25);
        transform: translateY(-2px);
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(23, 10, 46, 0.8) 0%, rgba(12, 5, 24, 0.9) 100%);
        border: 1px solid rgba(168, 85, 247, 0.25);
        border-radius: 14px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #a855f7, #38bdf8);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #c084fc;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Agent Execution Trace Node Styling */
    .agent-trace-box {
        background: rgba(18, 9, 36, 0.8);
        border-left: 4px solid #a855f7;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
        border-top: 1px solid rgba(168, 85, 247, 0.15);
        border-right: 1px solid rgba(168, 85, 247, 0.15);
        border-bottom: 1px solid rgba(168, 85, 247, 0.15);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .agent-trace-pass {
        border-left-color: #10b981;
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.08) 0%, rgba(18, 9, 36, 0.8) 100%);
    }
    .agent-trace-fail {
        border-left-color: #f43f5e;
        background: linear-gradient(90deg, rgba(244, 63, 94, 0.08) 0%, rgba(18, 9, 36, 0.8) 100%);
    }
    
    .status-badge-pass {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .status-badge-fail {
        background: rgba(244, 63, 94, 0.2);
        color: #fb7185;
        border: 1px solid #f43f5e;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    /* Live Pulse Status Dot */
    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981;
        margin-right: 8px;
        animation: pulse-animation 2s infinite;
    }
    @keyframes pulse-animation {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #9333ea 0%, #c084fc 100%);
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 12px 28px;
        box-shadow: 0 4px 15px rgba(147, 51, 234, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #a855f7 0%, #d8b4fe 100%);
        box-shadow: 0 0 25px rgba(168, 85, 247, 0.7);
        transform: translateY(-2px);
    }
    
    /* Code and Pre */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #0c0617 !important;
        border: 1px solid rgba(168, 85, 247, 0.2) !important;
        border-radius: 8px !important;
        color: #e9d5ff !important;
    }
</style>
"""
st.markdown(ADVANCED_STYLING_CSS, unsafe_allow_html=True)


# --- Session State Initialization ---
if "current_df" not in st.session_state:
    df_init, meta_init = generate_synthetic_ohlcv(symbol="SYNTH_BTC", n_bars=400, seed=42)
    st.session_state["current_df"] = df_init
    st.session_state["current_meta"] = meta_init

if "latest_workflow_state" not in st.session_state:
    st.session_state["latest_workflow_state"] = None


# --- Sidebar Branding & Navigation ---
st.sidebar.markdown(
    """
    <div style='padding: 10px 0;'>
        <div style='display:flex; align-items:center; gap:12px;'>
            <div style='background: linear-gradient(135deg, #a855f7, #38bdf8); width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; box-shadow: 0 0 15px rgba(168,85,247,0.5);'>
                <span style='font-size:20px;'>⚡</span>
            </div>
            <div>
                <h3 style='margin:0; font-size:1.3rem; color:#ffffff; font-weight:800;'>QuantLab Pro</h3>
                <p style='margin:0; font-size:0.75rem; color:#c084fc; font-weight:500;'>TOWER QUANT ENGINE v1.0</p>
            </div>
        </div>
        <div style='margin-top:14px; background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); padding:6px 12px; border-radius:20px; font-size:0.72rem; color:#34d399; display:inline-flex; align-items:center;'>
            <span class='pulse-dot'></span> SYSTEM ONLINE
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

nav_option = st.sidebar.radio(
    "NAVIGATION TERMINAL",
    [
        "📊 Overview",
        "🔍 Data Lab",
        "🧪 Experiment Lab",
        "🤖 Agent Trace",
        "📈 Results & Risk",
        "📜 Experiment History",
        "🛠️ MCP Tools",
        "ℹ️ Developer Guide & About",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='background:rgba(14,6,28,0.7); border:1px solid rgba(168,85,247,0.2); padding:12px; border-radius:10px; font-size:0.75rem; color:#a855f7;'>
        <b>🔐 Research Disclaimer</b><br/>
        Educational platform for quantitative ML research. Not financial advice or live trading.
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 1. OVERVIEW SECTION
# ==========================================
if nav_option == "📊 Overview":
    st.markdown("<div class='hero-title'>⚡ QuantLab Pro Terminal</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Multi-Agent Orchestration & Leakage-Resistant Quantitative ML Architecture</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Active Symbol</div>
                <div class='kpi-value'>{st.session_state['current_meta'].get('symbol', 'N/A')}</div>
                <div class='kpi-sub'>Primary Asset Ticker</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Sample Size</div>
                <div class='kpi-value'>{len(st.session_state['current_df'])} <span style='font-size:1rem; font-weight:400; color:#a855f7;'>bars</span></div>
                <div class='kpi-sub'>Historical Observations</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Provenance Source</div>
                <div class='kpi-value'>{st.session_state['current_meta'].get('source', 'synthetic').upper()}</div>
                <div class='kpi-sub'>Verified SHA-256 Origin</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        db = ExperimentStorage()
        exp_count = len(db.list_experiments())
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Recorded Experiments</div>
                <div class='kpi-value'>{exp_count}</div>
                <div class='kpi-sub'>SQLite Persistent Store</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### 🏛️ Core Platform Competencies")
    
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown(
            """
            <div class='glass-panel'>
                <div style='display:flex; align-items:center; gap:12px; margin-bottom:12px;'>
                    <div style='background:rgba(168,85,247,0.2); padding:10px; border-radius:10px; font-size:22px;'>🤖</div>
                    <h3 style='margin:0; font-size:1.2rem; color:#e9d5ff;'>Multi-Agent Orchestration</h3>
                </div>
                <p style='color:#cbd5e1; font-size:0.92rem; line-height:1.6;'>
                    Executes deterministic workflows using 6 specialized quantitative agents:
                    <b style='color:#c084fc;'>Data Quality</b>, <b style='color:#c084fc;'>Quant Research Planner</b>, 
                    <b style='color:#c084fc;'>Feature Engineering</b>, <b style='color:#c084fc;'>Model Training</b>, 
                    <b style='color:#c084fc;'>Evaluation & Risk</b>, and <b style='color:#c084fc;'>Experiment Auditor</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class='glass-panel'>
                <div style='display:flex; align-items:center; gap:12px; margin-bottom:12px;'>
                    <div style='background:rgba(56,189,248,0.2); padding:10px; border-radius:10px; font-size:22px;'>🛡️</div>
                    <h3 style='margin:0; font-size:1.2rem; color:#e9d5ff;'>Leakage-Resistant ML Pipelines</h3>
                </div>
                <p style='color:#cbd5e1; font-size:0.92rem; line-height:1.6;'>
                    Prevents look-ahead bias with strict target shifting, lagged technical indicators, 
                    and chronological train/validation/test splits. Preprocessing scalers are fit 
                    <b style='color:#38bdf8;'>strictly on training split data only</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_b:
        st.markdown(
            """
            <div class='glass-panel'>
                <div style='display:flex; align-items:center; gap:12px; margin-bottom:12px;'>
                    <div style='background:rgba(236,72,153,0.2); padding:10px; border-radius:10px; font-size:22px;'>🛠️</div>
                    <h3 style='margin:0; font-size:1.2rem; color:#e9d5ff;'>Official MCP Python SDK & Tool Registry</h3>
                </div>
                <p style='color:#cbd5e1; font-size:0.92rem; line-height:1.6;'>
                    Exposes allowlisted domain tools (validation, features, training, metrics) via 
                    the official Model Context Protocol (MCP) server for integration with Cursor, Claude Code, or external clients.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class='glass-panel'>
                <div style='display:flex; align-items:center; gap:12px; margin-bottom:12px;'>
                    <div style='background:rgba(16,185,129,0.2); padding:10px; border-radius:10px; font-size:22px;'>🔬</div>
                    <h3 style='margin:0; font-size:1.2rem; color:#e9d5ff;'>MLOps, Tracking & FastAPI Service</h3>
                </div>
                <p style='color:#cbd5e1; font-size:0.92rem; line-height:1.6;'>
                    Provides SQLite persistence, JSON reproducibility manifests, optional MLflow integration, 
                    and a local FastAPI REST service for real-time model inference and monitoring.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🔄 Interactive Multi-Agent Pipeline Topology")
    st.markdown(
        """
        <div class='glass-panel' style='text-align:center;'>
            <div style='display:flex; flex-wrap:wrap; justify-content:center; align-items:center; gap:12px; padding:10px 0;'>
                <div style='background:#170a2c; border:1px solid #a855f7; border-radius:10px; padding:12px 18px; font-weight:600; color:#e9d5ff;'>
                    📊 1. Data Quality Agent
                </div>
                <div style='color:#a855f7; font-size:20px;'>➔</div>
                <div style='background:#170a2c; border:1px solid #a855f7; border-radius:10px; padding:12px 18px; font-weight:600; color:#e9d5ff;'>
                    📐 2. Research Planner Agent
                </div>
                <div style='color:#a855f7; font-size:20px;'>➔</div>
                <div style='background:#170a2c; border:1px solid #a855f7; border-radius:10px; padding:12px 18px; font-weight:600; color:#e9d5ff;'>
                    ⚙️ 3. Feature Agent
                </div>
                <div style='color:#a855f7; font-size:20px;'>➔</div>
                <div style='background:#170a2c; border:1px solid #38bdf8; border-radius:10px; padding:12px 18px; font-weight:600; color:#e9d5ff;'>
                    🧠 4. Model Training Agent
                </div>
                <div style='color:#38bdf8; font-size:20px;'>➔</div>
                <div style='background:#170a2c; border:1px solid #38bdf8; border-radius:10px; padding:12px 18px; font-weight:600; color:#e9d5ff;'>
                    📈 5. Evaluation & Risk Agent
                </div>
                <div style='color:#38bdf8; font-size:20px;'>➔</div>
                <div style='background:#170a2c; border:1px solid #10b981; border-radius:10px; padding:12px 18px; font-weight:600; color:#e9d5ff;'>
                    📝 6. Auditor Agent
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# 2. DATA LAB SECTION
# ==========================================
elif nav_option == "🔍 Data Lab":
    st.markdown("<div class='hero-title'>🔍 Data Lab & Validation Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Load market datasets, inspect SHA-256 provenance, and run quantitative quality audits.</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    data_source_mode = st.radio("Select Provenance Source Mode", ["Synthetic Generator", "Upload CSV File", "yfinance Download"], horizontal=True)
    
    if data_source_mode == "Synthetic Generator":
        c1, c2, c3 = st.columns(3)
        with c1:
            sym = st.text_input("Asset Symbol", value="SYNTH_BTC")
        with c2:
            n_bars = st.slider("Number of Bars", min_value=150, max_value=2000, value=400, step=50)
        with c3:
            seed = st.number_input("Deterministic Seed", value=42, step=1)
            
        if st.button("Generate Synthetic Dataset"):
            df_gen, meta_gen = generate_synthetic_ohlcv(symbol=sym, n_bars=n_bars, seed=seed)
            st.session_state["current_df"] = df_gen
            st.session_state["current_meta"] = meta_gen
            st.success(f"Generated synthetic dataset '{sym}' ({n_bars} bars) with SHA-256 fingerprint `{meta_gen['fingerprint'][:16]}...`")

    elif data_source_mode == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload Market OHLCV CSV File", type=["csv"])
        custom_sym = st.text_input("Custom Ticker Label", value="UPLOADED_ASSET")
        if uploaded_file is not None and st.button("Load and Validate CSV"):
            try:
                df_csv, meta_csv = load_csv_dataset(uploaded_file, symbol=custom_sym)
                st.session_state["current_df"] = df_csv
                st.session_state["current_meta"] = meta_csv
                st.success(f"Successfully loaded CSV '{custom_sym}' ({len(df_csv)} bars).")
            except Exception as ex:
                st.error(f"Error loading CSV file: {str(ex)}")

    elif data_source_mode == "yfinance Download":
        c1, c2, c3 = st.columns(3)
        with c1:
            yf_ticker = st.text_input("Ticker Symbol", value="AAPL")
        with c2:
            s_date = st.text_input("Start Date", value="2023-01-01")
        with c3:
            e_date = st.text_input("End Date", value="2024-01-01")
            
        if st.button("Download Historical Market Data"):
            with st.spinner("Downloading market data from yfinance..."):
                try:
                    df_yf, meta_yf = load_yfinance_dataset(symbol=yf_ticker, start_date=s_date, end_date=e_date)
                    st.session_state["current_df"] = df_yf
                    st.session_state["current_meta"] = meta_yf
                    st.success(f"Successfully downloaded {yf_ticker} data ({len(df_yf)} bars).")
                except Exception as ex:
                    st.error(f"yfinance Download Error: {str(ex)}. Please select another source or verify internet connectivity.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Active Dataset Overview & Validation
    df_curr = st.session_state["current_df"]
    meta_curr = st.session_state["current_meta"]
    
    st.markdown("### 📄 Provenance & Metadata Signature")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.markdown(f"<div class='kpi-card'><div class='kpi-label'>SYMBOL</div><div class='kpi-value'>{meta_curr.get('symbol')}</div></div>", unsafe_allow_html=True)
    m_col2.markdown(f"<div class='kpi-card'><div class='kpi-label'>PROVENANCE</div><div class='kpi-value'>{meta_curr.get('source').upper()}</div></div>", unsafe_allow_html=True)
    m_col3.markdown(f"<div class='kpi-card'><div class='kpi-label'>BAR COUNT</div><div class='kpi-value'>{len(df_curr)}</div></div>", unsafe_allow_html=True)
    m_col4.markdown(f"<div class='kpi-card'><div class='kpi-label'>FINGERPRINT</div><div class='kpi-value' style='font-size:1.1rem;'>{meta_curr.get('fingerprint')[:14]}...</div></div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    # Run Data Validation Audit
    val_report = validate_ohlcv_data(df_curr)
    
    st.markdown("### 🛡️ Quantitative Validation Audit Report")
    if val_report.is_valid:
        st.markdown(
            f"""
            <div style='background:rgba(16,185,129,0.12); border:1px solid #10b981; border-radius:12px; padding:16px; margin-bottom:20px;'>
                <b style='color:#34d399; font-size:1.05rem;'>✅ Data Quality Audit Passed</b>
                <p style='color:#e2e8f0; margin-top:4px; font-size:0.9rem;'>All critical price boundary, monotonicity, and schema integrity checks passed cleanly. Zero blocking errors.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='background:rgba(244,63,94,0.12); border:1px solid #f43f5e; border-radius:12px; padding:16px; margin-bottom:20px;'>
                <b style='color:#fb7185; font-size:1.05rem;'>❌ Data Quality Audit Failed ({len(val_report.errors)} blocking errors)</b>
            </div>
            """,
            unsafe_allow_html=True
        )
        for err in val_report.errors:
            st.error(f"🔴 **Error**: {err}")
            
    if val_report.warnings:
        for warn in val_report.warnings:
            st.warning(f"⚠️ **Warning**: {warn}")

    with st.expander("Inspection: View Audit Step Log & Raw Market Data Table", expanded=False):
        st.dataframe(pd.DataFrame(val_report.audit_trail), use_container_width=True)
        st.dataframe(df_curr.head(50), use_container_width=True)

    # Plotly Candlestick & Volume Chart
    st.markdown("### 📈 Interactive Candlestick Market Chart")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_curr["Date"],
        open=df_curr["Open"],
        high=df_curr["High"],
        low=df_curr["Low"],
        close=df_curr["Close"],
        name="OHLC Price",
        increasing_line_color="#c084fc",
        increasing_fillcolor="rgba(192, 132, 252, 0.4)",
        decreasing_line_color="#f472b6",
        decreasing_fillcolor="rgba(244, 114, 182, 0.4)",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#05020a",
        plot_bgcolor="#0e061a",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_rangeslider_visible=False,
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 3. EXPERIMENT LAB SECTION
# ==========================================
elif nav_option == "🧪 Experiment Lab":
    st.markdown("<div class='hero-title'>🧪 Quantitative Experiment Lab</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Configure model architecture, prediction target horizon, and launch the multi-agent execution pipeline.</div>", unsafe_allow_html=True)
    
    df_curr = st.session_state["current_df"]
    meta_curr = st.session_state["current_meta"]
    
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        model_choice = st.selectbox(
            "Select Model Architecture",
            ["logistic_regression", "pytorch_mlp", "majority_class"],
            index=0,
            help="Choose baseline or PyTorch MLP classifier for directional return prediction."
        )
        target_h = st.slider("Forecast Target Horizon (h bars forward)", min_value=1, max_value=5, value=1)
    with c2:
        exp_seed = st.number_input("Deterministic Random Seed", value=42, step=1)
        tx_cost = st.number_input("Backtest Transaction Cost (bps)", value=10.0, step=1.0)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("🚀 Launch Multi-Agent Workflow Execution"):
        with st.spinner("Multi-Agent Orchestrator executing sequential agent steps..."):
            orchestrator = MultiAgentOrchestrator()
            state = orchestrator.run_pipeline(
                df_raw=df_curr,
                raw_metadata=meta_curr,
                model_choice=model_choice,
                seed=exp_seed,
                target_horizon=target_h,
                transaction_cost_bps=tx_cost,
            )
            st.session_state["latest_workflow_state"] = state
            
        if state.is_completed:
            st.success(f"🎉 Workflow Execution Completed! Generated Experiment ID: `{state.experiment_id}`")
            st.info("Navigate to **🤖 Agent Trace** or **📈 Results & Risk** tabs to review execution details and held-out test evaluation.")
        else:
            st.error(f"❌ Workflow Execution Halted: {state.error_message}")


# ==========================================
# 4. AGENT TRACE SECTION
# ==========================================
elif nav_option == "🤖 Agent Trace":
    st.markdown("<div class='hero-title'>🤖 Multi-Agent Execution Trace</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Step-by-step state transition logs, step duration, decision rationales, and agent artifacts.</div>", unsafe_allow_html=True)
    
    state = st.session_state.get("latest_workflow_state")
    if state is None:
        st.warning("No workflow execution recorded in current session. Please run an experiment in the **🧪 Experiment Lab** first.")
    else:
        st.markdown(
            f"""
            <div class='glass-panel'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <b style='color:#c084fc; font-size:0.9rem;'>EXPERIMENT RUN ID</b>
                        <h2 style='margin:0; font-family:"JetBrains Mono"; color:#ffffff;'>{state.experiment_id}</h2>
                    </div>
                    <div>
                        <span class='{"status-badge-pass" if state.is_completed else "status-badge-fail"}'>
                            {'✅ WORKFLOW COMPLETED' if state.is_completed else '❌ WORKFLOW HALTED'}
                        </span>
                    </div>
                </div>
                <hr style='border-color:rgba(168,85,247,0.2); margin:16px 0;'/>
                <p style='color:#e2e8f0; margin:0;'><b>Research Rationale</b>: <i>{state.research_rationale}</i></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("### Agent Execution Timeline")
        for trace in state.execution_trace:
            is_pass = trace.status == "PASS"
            box_class = "agent-trace-pass" if is_pass else "agent-trace-fail"
            badge_class = "status-badge-pass" if is_pass else "status-badge-fail"
            
            st.markdown(
                f"""
                <div class='agent-trace-box {box_class}'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <div style='display:flex; align-items:center; gap:12px;'>
                            <div style='background:rgba(168,85,247,0.2); width:32px; height:32px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-weight:bold; color:#c084fc;'>
                                {trace.step_index}
                            </div>
                            <h4 style='margin:0; color:#ffffff;'>{trace.agent_name}</h4>
                        </div>
                        <div>
                            <span class='{badge_class}'>{trace.status}</span>
                            <span style='margin-left:12px; font-family:"JetBrains Mono"; color:#c084fc; font-size:0.85rem;'>{trace.duration_ms:.1f} ms</span>
                        </div>
                    </div>
                    <p style='color:#94a3b8; font-size:0.85rem; margin:8px 0 4px 0;'><b>Role</b>: {trace.role}</p>
                    <p style='color:#f3e8ff; margin:0; font-weight:500;'>{trace.summary}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            with st.expander(f"Inspect Agent {trace.step_index} ({trace.agent_name}) State Details"):
                st.json(trace.details)


# ==========================================
# 5. RESULTS & RISK SECTION
# ==========================================
elif nav_option == "📈 Results & Risk":
    st.markdown("<div class='hero-title'>📈 Held-Out Evaluation & Risk Audit</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Held-out test split classification accuracy, macro F1 score, backtest equity curve, and data leakage flags.</div>", unsafe_allow_html=True)
    
    state = st.session_state.get("latest_workflow_state")
    if state is None or state.test_metrics is None:
        st.warning("No experiment results available. Run an experiment in the **🧪 Experiment Lab** first.")
    else:
        m = state.test_metrics
        b = state.backtest_result
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='kpi-card'><div class='kpi-label'>HELD-OUT TEST ACCURACY</div><div class='kpi-value'>{m.accuracy * 100:.1f}%</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi-card'><div class='kpi-label'>MACRO F1-SCORE</div><div class='kpi-value'>{m.f1_score:.2f}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi-card'><div class='kpi-label'>SHARPE RATIO</div><div class='kpi-value'>{b.sharpe_ratio:.2f}</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='kpi-card'><div class='kpi-label'>MAX DRAWDOWN</div><div class='kpi-value'>{b.max_drawdown * 100:.1f}%</div></div>", unsafe_allow_html=True)
        
        st.markdown("<br/>", unsafe_allow_html=True)
        
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown("### 📊 Classification Confusion Matrix")
            cm_df = pd.DataFrame(
                m.confusion_matrix,
                index=["Actual Non-Pos (0)", "Actual Pos (1)"],
                columns=["Pred Non-Pos (0)", "Pred Pos (1)"]
            )
            st.table(cm_df)
            
            st.markdown("### 📋 Per-Class Performance Breakdown")
            st.json(m.per_class_report)
            
        with r_col2:
            st.markdown("### 📈 Cumulative Strategy vs Benchmark Return")
            if b:
                fig_ret = go.Figure()
                fig_ret.add_trace(go.Scatter(
                    x=b.dates,
                    y=b.strategy_curve,
                    mode="lines",
                    name="Strategy Net Return",
                    line=dict(color="#c084fc", width=2.5)
                ))
                fig_ret.add_trace(go.Scatter(
                    x=b.dates,
                    y=b.benchmark_curve,
                    mode="lines",
                    name="Benchmark (Buy & Hold)",
                    line=dict(color="#38bdf8", width=1.5, dash="dash")
                ))
                fig_ret.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#05020a",
                    plot_bgcolor="#0e061a",
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=380,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_ret, use_container_width=True)

        st.markdown("### ⚠️ Risk Auditing & Data Leakage Warnings")
        if state.risk_warnings:
            for rw in state.risk_warnings:
                st.warning(f"⚠️ {rw}")
        else:
            st.success("✅ Zero critical risk or data leakage flags detected.")


# ==========================================
# 6. EXPERIMENT HISTORY SECTION
# ==========================================
elif nav_option == "📜 Experiment History":
    st.markdown("<div class='hero-title'>📜 Experiment Persistence Store</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>SQLite database experiment runs, manifest viewer, export, and history reset.</div>", unsafe_allow_html=True)
    
    storage = ExperimentStorage()
    runs = storage.list_experiments()
    
    if not runs:
        st.info("No recorded experiments found in SQLite database.")
    else:
        df_runs = pd.DataFrame(runs)
        st.dataframe(df_runs, use_container_width=True)
        
        st.markdown("<br/>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            selected_exp_id = st.selectbox("Select Experiment ID to Inspect Manifest", df_runs["experiment_id"].tolist())
            if selected_exp_id:
                rec = storage.get_experiment(selected_exp_id)
                if rec:
                    st.markdown(f"#### Reproducibility Manifest (`{selected_exp_id}`)")
                    st.json(rec.manifest_json)
                    manifest_bytes = json.dumps(rec.manifest_json, indent=2).encode("utf-8")
                    st.download_button(
                        label="📥 Export Manifest JSON",
                        data=manifest_bytes,
                        file_name=f"{selected_exp_id}_manifest.json",
                        mime="application/json"
                    )
        with c2:
            st.markdown("#### 🗑️ Reset SQLite Database History")
            st.caption("Permanently clear all recorded experiment runs from database.")
            confirm_reset = st.checkbox("I confirm I want to clear all experiment history.")
            if st.button("Clear All Experiments") and confirm_reset:
                storage.clear_all_experiments()
                st.success("Database history cleared successfully.")
                st.rerun()


# ==========================================
# 7. MCP TOOLS SECTION
# ==========================================
elif nav_option == "🛠️ MCP Tools":
    st.markdown("<div class='hero-title'>🛠️ Model Context Protocol (MCP) Registry</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Inspect allowlisted tools, execute tool calls directly, and view Pydantic JSON schemas.</div>", unsafe_allow_html=True)
    
    registry = ToolRegistry()
    tools = registry.list_tools()
    
    st.markdown("### Allowlisted Domain Tools")
    for t in tools:
        with st.expander(f"Tool: `{t['name']}` - {t['description']}"):
            st.markdown("**Parameters Pydantic Schema**:")
            st.json(t["parameters_schema"])

    st.markdown("---")
    st.markdown("### 🧪 Direct Tool Invocator")
    
    tool_choice = st.selectbox("Select Tool to Invoke", [t["name"] for t in tools])
    
    if tool_choice == "dataset_summary":
        sym = st.text_input("symbol", "MCP_BTC")
        n = st.number_input("n_bars", 200)
        s = st.number_input("seed", 42)
        if st.button("Invoke dataset_summary"):
            res = registry.execute("dataset_summary", {"symbol": sym, "n_bars": n, "seed": s})
            st.json(res.model_dump())
            
    elif tool_choice == "run_experiment":
        sym = st.text_input("symbol", "MCP_BTC")
        m_type = st.selectbox("model_type", ["logistic_regression", "pytorch_mlp", "majority_class"])
        if st.button("Invoke run_experiment"):
            res = registry.execute("run_experiment", {"symbol": sym, "model_type": m_type})
            st.json(res.model_dump())


# ==========================================
# 8. DEVELOPER GUIDE & ABOUT SECTION
# ==========================================
elif nav_option == "ℹ️ Developer Guide & About":
    st.markdown("<div class='hero-title'>ℹ️ Developer Guide & Project Architecture</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Tower Research Capital alignment, agentic coding workflows, and local service execution.</div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class='glass-panel'>
            <h3 style='color:#c084fc; margin-top:0;'>Tower Research Capital Competencies</h3>
            <p style='color:#cbd5e1;'>
                This platform demonstrates core quantitative development and AI/ML competencies:
            </p>
            <ul style='color:#e2e8f0; line-height:1.8;'>
                <li><b>Multi-Agent Orchestration</b>: Sequential execution of 6 specialized agents with typed state passing.</li>
                <li><b>Statistical & Quantitative ML</b>: Leakage-resistant feature engineering, chronological splits, PyTorch MLP, and backtesting.</li>
                <li><b>MCP Implementation</b>: Official Python MCP SDK exposing allowlisted tools over stdio transport.</li>
                <li><b>MLOps & Reproducibility</b>: SQLite experiment store, SHA-256 dataset fingerprinting, and FastAPI REST endpoints.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("### 🤖 Agentic Coding Tool Workflow Guide")
    st.markdown(
        """
        When using agentic assistants like **Claude Code**, **Cursor**, or **Codex** on this codebase:
        1. **Refactoring Features**: Prompt the agent to inspect `quantlab/features/builder.py` and enforce strict chronological scaling.
        2. **Adding Model Architecture**: Add model wrappers in `quantlab/models/` implementing `.fit()` and `.predict()`.
        3. **Adding Unit Tests**: Run `pytest -v` to ensure zero regression failures.
        """
    )
    
    st.markdown("### ⚡ Local Services Commands")
    st.markdown(
        """
        - **Launch Streamlit Terminal**: `streamlit run app.py`
        - **Run Test Suite**: `pytest -v`
        - **Run MCP Server**: `python -m mcp_server.server`
        - **Run FastAPI Service**: `uvicorn api.main:app --reload --port 8000`
        """
    )
