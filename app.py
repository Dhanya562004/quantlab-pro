"""
QuantLab Pro: Multi-Agent Quantitative ML Research Platform
Streamlit Main Interactive Application.

Designed for educational research in quantitative ML, agentic orchestration, and model reproducibility.
Aligned to Tower Research Capital's Intern - AI/ML Job Description competencies.
"""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab.data.loader import load_csv_dataset, load_yfinance_dataset

# Import QuantLab Pro Modules
from quantlab.data.synthetic import (
    generate_synthetic_ohlcv,
)
from quantlab.data.validation import validate_ohlcv_data
from quantlab.orchestration.engine import MultiAgentOrchestrator
from quantlab.storage.db import ExperimentStorage
from quantlab.tools.registry import ToolRegistry

# --- Page Configuration ---
st.set_page_config(
    page_title="QuantLab Pro | Quantitative ML Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Dark Purple & Black Cyberpunk Aesthetic CSS ---
STYLING_CSS = """
<style>
    /* Global Page Styling */
    .stApp {
        background-color: #0b0614;
        color: #f3e8ff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #120824;
        border-right: 1px solid #2e1554;
    }
    
    /* Cards and Glassmorphism Containers */
    .glass-card {
        background: linear-gradient(135deg, rgba(23, 10, 44, 0.8) 0%, rgba(14, 6, 28, 0.9) 100%);
        border: 1px solid #3d1b70;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    }
    
    .agent-card-pass {
        background: linear-gradient(135deg, rgba(20, 10, 44, 0.9) 0%, rgba(10, 24, 28, 0.9) 100%);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .agent-card-fail {
        background: linear-gradient(135deg, rgba(44, 10, 20, 0.9) 0%, rgba(28, 6, 10, 0.9) 100%);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* Metric Cards */
    .metric-container {
        background: #170a2c;
        border: 1px solid #381963;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #c084fc;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Purple Neon Headers */
    h1, h2, h3 {
        color: #e9d5ff !important;
        font-weight: 700 !important;
    }
    
    .neon-text {
        color: #c084fc;
        text-shadow: 0 0 10px rgba(192, 132, 252, 0.5);
    }
    
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #9333ea 0%, #c084fc 100%);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #a855f7 0%, #d8b4fe 100%);
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.6);
        transform: translateY(-1px);
    }
</style>
"""
st.markdown(STYLING_CSS, unsafe_allow_html=True)


# --- Session State Initialization ---
if "current_df" not in st.session_state:
    df_init, meta_init = generate_synthetic_ohlcv(symbol="SYNTH_BTC", n_bars=400, seed=42)
    st.session_state["current_df"] = df_init
    st.session_state["current_meta"] = meta_init

if "latest_workflow_state" not in st.session_state:
    st.session_state["latest_workflow_state"] = None


# --- Sidebar Navigation ---
st.sidebar.markdown("<h2 class='neon-text'>⚡ QuantLab Pro</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#a855f7; font-size:0.85rem;'>Multi-Agent Quantitative ML Platform</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

nav_option = st.sidebar.radio(
    "Navigation Menu",
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
st.sidebar.caption("🔒 Educational Research Platform Only")
st.sidebar.caption("Not Financial or Investment Advice")


# ==========================================
# 1. OVERVIEW SECTION
# ==========================================
if nav_option == "📊 Overview":
    st.markdown("<h1>⚡ QuantLab Pro Platform Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#d8b4fe; font-size:1.1rem;'>Multi-Agent Orchestration & Leakage-Resistant Quantitative Machine Learning Architecture</p>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class='metric-container'>
                <div class='metric-label'>Active Dataset</div>
                <div class='metric-value'>{st.session_state['current_meta'].get('symbol', 'N/A')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class='metric-container'>
                <div class='metric-label'>Sample Size</div>
                <div class='metric-value'>{len(st.session_state['current_df'])} bars</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class='metric-container'>
                <div class='metric-label'>Provenance Source</div>
                <div class='metric-value'>{st.session_state['current_meta'].get('source', 'synthetic').upper()}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        db = ExperimentStorage()
        exp_count = len(db.list_experiments())
        st.markdown(
            f"""
            <div class='metric-container'>
                <div class='metric-label'>Recorded Experiments</div>
                <div class='metric-value'>{exp_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("### Core Platform Capabilities")
    
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown(
            """
            <div class='glass-card'>
                <h4 style='color:#c084fc;'>🤖 Multi-Agent Orchestration</h4>
                <p style='color:#e9d5ff;'>
                    Executes deterministic workflows using 6 specialized quantitative agents:
                    <b>Data Quality</b>, <b>Quant Research Planner</b>, <b>Feature Engineering</b>, 
                    <b>Model Training</b>, <b>Evaluation & Risk</b>, and <b>Experiment Auditor</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class='glass-card'>
                <h4 style='color:#c084fc;'>🛡️ Leakage-Resistant ML Pipelines</h4>
                <p style='color:#e9d5ff;'>
                    Prevents look-ahead bias with strict target shifting, lagged technical indicators, 
                    and chronological train/validation/test splits. Preprocessing scalers are fit 
                    <b>strictly</b> on training split data.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_b:
        st.markdown(
            """
            <div class='glass-card'>
                <h4 style='color:#c084fc;'>🛠️ Official MCP SDK & Safe Tool Calling</h4>
                <p style='color:#e9d5ff;'>
                    Exposes allowlisted domain tools (validation, features, training, metrics) via 
                    the official Model Context Protocol (MCP) server for integration with Cursor, Claude Code, or external clients.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class='glass-card'>
                <h4 style='color:#c084fc;'>🔬 MLOps, Tracking & Local Inference</h4>
                <p style='color:#e9d5ff;'>
                    Provides SQLite persistence, JSON reproducibility manifests, optional MLflow integration, 
                    and a local FastAPI REST service for real-time model inference.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🏛️ Multi-Agent Architecture Workflow")
    st.markdown(
        """
        ```mermaid
        graph LR
            A[Raw Dataset] --> B(Data Quality Agent)
            B --> C(Quant Research Planner Agent)
            C --> D(Feature Engineering Agent)
            D --> E(Model Training Agent)
            E --> F(Evaluation & Risk Agent)
            F --> G(Experiment Auditor Agent)
            G --> H[(SQLite Database)]
        ```
        """,
        unsafe_allow_html=True
    )


# ==========================================
# 2. DATA LAB SECTION
# ==========================================
elif nav_option == "🔍 Data Lab":
    st.markdown("<h1>🔍 Data Lab & Quantitative Validation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#d8b4fe;'>Load market data, inspect provenance metadata, and execute quantitative validation audits.</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    data_source_mode = st.radio("Select Data Source Mode", ["Synthetic Generator", "Upload CSV File", "yfinance Download"], horizontal=True)
    
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
            st.success(f"Generated synthetic dataset '{sym}' ({n_bars} bars) with fingerprint {meta_gen['fingerprint'][:16]}...")

    elif data_source_mode == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload Market OHLCV CSV", type=["csv"])
        custom_sym = st.text_input("Custom Ticker Label", value="UPLOADED_ASSET")
        if uploaded_file is not None and st.button("Load and Validate CSV"):
            try:
                df_csv, meta_csv = load_csv_dataset(uploaded_file, symbol=custom_sym)
                st.session_state["current_df"] = df_csv
                st.session_state["current_meta"] = meta_csv
                st.success(f"Successfully loaded CSV '{custom_sym}' ({len(df_csv)} bars).")
            except Exception as ex:
                st.error(f"Error loading CSV file: {ex!s}")

    elif data_source_mode == "yfinance Download":
        c1, c2, c3 = st.columns(3)
        with c1:
            yf_ticker = st.text_input("Ticker Symbol", value="AAPL")
        with c2:
            s_date = st.text_input("Start Date", value="2023-01-01")
        with c3:
            e_date = st.text_input("End Date", value="2024-01-01")
            
        if st.button("Download Historical Data"):
            with st.spinner("Downloading market data from yfinance..."):
                try:
                    df_yf, meta_yf = load_yfinance_dataset(symbol=yf_ticker, start_date=s_date, end_date=e_date)
                    st.session_state["current_df"] = df_yf
                    st.session_state["current_meta"] = meta_yf
                    st.success(f"Successfully downloaded {yf_ticker} data ({len(df_yf)} bars).")
                except Exception as ex:
                    st.error(f"yfinance Download Error: {ex!s}. Please select another source or verify your internet connection.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Active Dataset Overview & Validation
    df_curr = st.session_state["current_df"]
    meta_curr = st.session_state["current_meta"]
    
    st.markdown("### 📄 Active Dataset Provenance & Summary")
    meta_cols = st.columns(4)
    meta_cols[0].info(f"**Symbol**: {meta_curr.get('symbol')}")
    meta_cols[1].info(f"**Provenance**: {meta_curr.get('source').upper()}")
    meta_cols[2].info(f"**Sample Count**: {len(df_curr)} bars")
    meta_cols[3].info(f"**Fingerprint**: `{meta_curr.get('fingerprint')[:16]}...`")

    # Run Data Validation Audit
    val_report = validate_ohlcv_data(df_curr)
    
    st.markdown("### 🛡️ Quantitative Data Validation Audit")
    if val_report.is_valid:
        st.success(f"✅ Data Quality Passed Validation Audit (0 blocking errors, {len(val_report.warnings)} warnings)")
    else:
        st.error(f"❌ Data Quality Failed Audit ({len(val_report.errors)} blocking errors)")
        for err in val_report.errors:
            st.write(f"- 🔴 **Error**: {err}")
            
    if val_report.warnings:
        for warn in val_report.warnings:
            st.warning(f"⚠️ **Warning**: {warn}")

    with st.expander("Inspection: View Audit Step Log & Raw Data Table", expanded=False):
        st.dataframe(pd.DataFrame(val_report.audit_trail), use_container_width=True)
        st.dataframe(df_curr.head(50), use_container_width=True)

    # Plotly Price Chart
    st.markdown("### 📈 Interactive Market Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_curr["Date"],
        open=df_curr["Open"],
        high=df_curr["High"],
        low=df_curr["Low"],
        close=df_curr["Close"],
        name="OHLC",
        increasing_line_color="#c084fc",
        decreasing_line_color="#f472b6",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0614",
        plot_bgcolor="#160b28",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 3. EXPERIMENT LAB SECTION
# ==========================================
elif nav_option == "🧪 Experiment Lab":
    st.markdown("<h1>🧪 Experiment Lab</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#d8b4fe;'>Configure ML experiment parameters and launch the Multi-Agent Quantitative Workflow.</p>", unsafe_allow_html=True)
    
    df_curr = st.session_state["current_df"]
    meta_curr = st.session_state["current_meta"]
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        model_choice = st.selectbox(
            "Select Model Architecture",
            ["logistic_regression", "pytorch_mlp", "majority_class"],
            index=0,
            help="Choose model algorithm for directional return prediction."
        )
        target_h = st.slider("Prediction Horizon (h bars forward)", min_value=1, max_value=5, value=1)
    with c2:
        exp_seed = st.number_input("Random Seed", value=42, step=1)
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
            st.info("Navigate to the **🤖 Agent Trace** or **📈 Results & Risk** tab to review the agent state sequence and backtest evaluation.")
        else:
            st.error(f"❌ Workflow Execution Halted: {state.error_message}")


# ==========================================
# 4. AGENT TRACE SECTION
# ==========================================
elif nav_option == "🤖 Agent Trace":
    st.markdown("<h1>🤖 Multi-Agent Execution Trace</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#d8b4fe;'>Real-time step-by-step state transition log, duration, and decision rationales across all 6 specialized agents.</p>", unsafe_allow_html=True)
    
    state = st.session_state.get("latest_workflow_state")
    if state is None:
        st.warning("No workflow execution recorded in current session. Please run an experiment in the **🧪 Experiment Lab** first.")
    else:
        st.markdown(f"**Experiment ID**: `{state.experiment_id}` | **Status**: {'✅ COMPLETED' if state.is_completed else '❌ HALTED'}")
        st.markdown(f"**Research Rationale**: _{state.research_rationale}_")
        
        st.markdown("---")
        for trace in state.execution_trace:
            card_class = "agent-card-pass" if trace.status == "PASS" else ("agent-card-fail" if trace.status == "FAIL" else "glass-card")
            status_icon = "✅ PASS" if trace.status == "PASS" else ("❌ FAIL" if trace.status == "FAIL" else "⏸️ SKIPPED")
            
            st.markdown(
                f"""
                <div class='{card_class}'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <h4 style='margin:0; color:#e9d5ff;'>Step {trace.step_index}: {trace.agent_name}</h4>
                        <span style='font-weight:bold;'>{status_icon} ({trace.duration_ms} ms)</span>
                    </div>
                    <p style='color:#c084fc; font-size:0.9rem; margin-top:4px;'><b>Role</b>: {trace.role}</p>
                    <p style='color:#ffffff;'><b>Outcome</b>: {trace.summary}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            with st.expander(f"Inspect Agent {trace.step_index} State Snapshot & Details"):
                st.json(trace.details)


# ==========================================
# 5. RESULTS & RISK SECTION
# ==========================================
elif nav_option == "📈 Results & Risk":
    st.markdown("<h1>📈 Experiment Results & Risk Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#d8b4fe;'>Held-out test set classification performance, confusion matrix, backtest returns curve, and risk auditing.</p>", unsafe_allow_html=True)
    
    state = st.session_state.get("latest_workflow_state")
    if state is None or state.test_metrics is None:
        st.warning("No experiment results available. Run an experiment in the **🧪 Experiment Lab** first.")
    else:
        m = state.test_metrics
        b = state.backtest_result
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Held-Out Test Accuracy", f"{m.accuracy * 100:.1f}%")
        c2.metric("Macro F1-Score", f"{m.f1_score:.2f}")
        c3.metric("Annualized Sharpe Ratio", f"{b.sharpe_ratio:.2f}" if b else "N/A")
        c4.metric("Max Drawdown", f"{b.max_drawdown * 100:.1f}%" if b else "N/A")
        
        st.markdown("---")
        
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown("### 📊 Classification Confusion Matrix")
            cm_df = pd.DataFrame(
                m.confusion_matrix,
                index=["Actual Non-Pos (0)", "Actual Pos (1)"],
                columns=["Pred Non-Pos (0)", "Pred Pos (1)"]
            )
            st.table(cm_df)
            
            st.markdown("### 📋 Per-Class Classification Metrics")
            st.json(m.per_class_report)
            
        with r_col2:
            st.markdown("### 📈 Cumulative Strategy vs Benchmark Return")
            if b:
                fig_ret = go.Figure()
                fig_ret.add_trace(go.Scatter(
                    x=b.dates,
                    y=b.strategy_curve,
                    mode="lines",
                    name="Strategy (Net Cost)",
                    line=dict(color="#c084fc", width=2)
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
                    paper_bgcolor="#0b0614",
                    plot_bgcolor="#160b28",
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_ret, use_container_width=True)

        st.markdown("### ⚠️ Risk & Data Leakage Audit Warnings")
        if state.risk_warnings:
            for rw in state.risk_warnings:
                st.warning(f"⚠️ {rw}")
        else:
            st.success("✅ Zero critical risk or data leakage flags detected.")


# ==========================================
# 6. EXPERIMENT HISTORY SECTION
# ==========================================
elif nav_option == "📜 Experiment History":
    st.markdown("<h1>📜 Experiment History & Reproducibility Database</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#d8b4fe;'>Persistent SQLite experiment store, run comparison, manifest export, and history reset.</p>", unsafe_allow_html=True)
    
    storage = ExperimentStorage()
    runs = storage.list_experiments()
    
    if not runs:
        st.info("No recorded experiments found in SQLite database.")
    else:
        df_runs = pd.DataFrame(runs)
        st.dataframe(df_runs, use_container_width=True)
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            selected_exp_id = st.selectbox("Select Experiment ID to Inspect Manifest", df_runs["experiment_id"].tolist())
            if selected_exp_id:
                rec = storage.get_experiment(selected_exp_id)
                if rec:
                    st.markdown(f"#### Reproducibility Manifest for `{selected_exp_id}`")
                    st.json(rec.manifest_json)
                    manifest_bytes = json.dumps(rec.manifest_json, indent=2).encode("utf-8")
                    st.download_button(
                        label="📥 Export Manifest JSON",
                        data=manifest_bytes,
                        file_name=f"{selected_exp_id}_manifest.json",
                        mime="application/json"
                    )
        with c2:
            st.markdown("#### 🗑️ Reset Database History")
            st.caption("Permanently clear all recorded experiment runs from SQLite database.")
            confirm_reset = st.checkbox("I confirm I want to clear all experiment history.")
            if st.button("Clear All Experiments") and confirm_reset:
                storage.clear_all_experiments()
                st.success("Database history cleared successfully.")
                st.rerun()


# ==========================================
# 7. MCP TOOLS SECTION
# ==========================================
elif nav_option == "🛠️ MCP Tools":
    st.markdown("<h1>🛠️ Model Context Protocol (MCP) Tool Registry</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#d8b4fe;'>Browse allowlisted tools, execute tool calls directly, and view JSON schemas.</p>", unsafe_allow_html=True)
    
    registry = ToolRegistry()
    tools = registry.list_tools()
    
    st.markdown("### Allowlisted Tools")
    for t in tools:
        with st.expander(f"Tool: `{t['name']}` - {t['description']}"):
            st.markdown("**Parameters Schema (Pydantic)**:")
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
    st.markdown("<h1>ℹ️ Developer Guide & Project Architecture</h1>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class='glass-card'>
            <h3 style='color:#c084fc;'>Tower Research Capital Alignment</h3>
            <p style='color:#e9d5ff;'>
                This platform demonstrates core quantitative development and AI/ML competencies:
            </p>
            <ul>
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
    
    st.markdown("### ⚡ Local Services Execution")
    st.markdown(
        """
        - **Launch Streamlit Dashboard**: `streamlit run app.py`
        - **Run Test Suite**: `pytest -v`
        - **Run MCP Server**: `python -m mcp_server.server`
        - **Run FastAPI Service**: `uvicorn api.main:app --reload --port 8000`
        """
    )
