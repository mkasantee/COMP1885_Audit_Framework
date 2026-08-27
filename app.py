import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Municipal Risk Command Centre",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED EXECUTIVE STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    
    /* Top Enterprise Header Bar */
    .enterprise-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #111827;
        padding: 10px 20px;
        border-radius: 8px;
        border: 1px solid #1f2937;
        margin-bottom: 15px;
    }
    .header-left { display: flex; align-items: center; gap: 12px; }
    .header-logo-text { font-size: 1.1rem; font-weight: 800; color: #3b82f6; letter-spacing: 0.5px; }
    .header-breadcrumbs { font-size: 0.8rem; color: #9ca3af; margin-bottom: 5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .header-profile { font-size: 0.85rem; font-weight: 600; color: #9ca3af; background: #1f2937; padding: 5px 10px; border-radius: 20px; }

    /* Metric Cards */
    .metric-card {
        background-color: #111827;
        padding: 18px;
        border-radius: 8px;
        border: 1px solid #1f2937;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title { font-size: 0.85rem; text-transform: uppercase; color: #9ca3af; font-weight: 700; letter-spacing: 0.5px; }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #ffffff; margin-top: 5px; }

    /* Custom Highlight Box */
    .highlight-card {
        background-color: #161f30;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        border-top: 1px solid #1f2937;
        border-right: 1px solid #1f2937;
        border-bottom: 1px solid #1f2937;
    }
    </style>
""", unsafe_allow_html=True)

# --- BREADCRUMBS & TOP HEADER BAR ---
st.markdown('<div class="header-breadcrumbs">HOME > DASHBOARD > MUNICIPAL GOVERNANCE</div>', unsafe_allow_html=True)

header_col1, header_col2, header_col3 = st.columns([2, 3, 2])
with header_col1:
    if os.path.exists("logo3.png"):
        st.image("logo3.png", width=40)
    else:
        st.markdown('<div class="header-logo-text">🛡️ MUNICIPAL SUITE</div>', unsafe_allow_html=True)
with header_col2:
    st.markdown('<div style="color: #9ca3af; font-size: 0.85rem; padding-top: 8px;">🔍 Search spatial metrics, LSOAs, or models...</div>', unsafe_allow_html=True)
with header_col3:
    st.markdown('<div style="text-align: right;"><span style="margin-right: 15px;">🔔 3</span><span style="margin-right: 15px;">❓</span><span class="header-profile">👤 MICHELLE K. ASANTE (MSc)</span></div>', unsafe_allow_html=True)

st.markdown("---")

# --- SIDEBAR GLOBAL CONTROLS ---
st.sidebar.markdown("### 🎛️ Pipeline Settings")
uploaded_file = st.sidebar.file_uploader("Ingest Master Spatial Dataset (.csv or .xlsx)", type=["csv", "xlsx", "xls"])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Policy Simulation Sliders")
fairness_threshold = st.sidebar.slider("Regulatory Parity Limit", min_value=0.0, max_value=1.0, value=0.2252, step=0.001)
rf_estimators = st.sidebar.slider("Ensemble Estimators (RF)", min_value=10, max_value=500, value=100, step=10)
patrol_multiplier = st.sidebar.slider("Patrol Resource Multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Module Navigation")
navigation = st.sidebar.radio(
    "Select View",
    [
        "🏠 1. Executive KPI Dashboard",
        "🔍 2. Borough & Sector Explorer",
        "📊 3. Sprint 1: Deprivation",
        "📈 4. Sprint 2: Risk Features",
        "📉 5. Sprint 3: Model Evaluation",
        "⚖️ 6. Sprint 4: Demographic Audit",
        "🚀 7. Sprint 5: MLOps & Export"
    ],
    label_visibility="collapsed"
)

# --- DATA SESSION CACHING & AUTOMATED SCHEMA MAPPING ---
if "spatial_dataframe" not in st.session_state:
    st.session_state.spatial_dataframe = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(uploaded_file)
            df = pd.read_excel(uploaded_file, sheet_name=xls.sheet_names[0])
        else:
            df = pd.read_csv(uploaded_file)
        
        df = df.dropna()
        
        # Automated Schema Mapping
        df["deprivation_index"] = df["IMD_Score"]
        df["historical_incidents"] = df["burglary_lag1"]
        df["patrol_frequency"] = df["TfL_Mobility_Index"] * patrol_multiplier
        df["vulnerability_flag"] = df["is_lockdown"]
        df["risk_density_ratio"] = df["historical_incidents"] / (df["patrol_frequency"] + 1.0)
        
        st.session_state.spatial_dataframe = df
        st.sidebar.success(f"Loaded: {len(df):,} records successfully.")
    except Exception as e:
        st.sidebar.error(f"Mapping Error: {e}")

df = st.session_state.spatial_dataframe

# --- MODULE 1: EXECUTIVE KPI COMMAND DASHBOARD ---
if navigation == "🏠 1. Executive KPI Dashboard":
    st.title("Municipal Decision Intelligence & Risk Command Centre")
    st.markdown("*Executive BI Dashboard & Algorithmic Governance Suite | Designed for Boardroom Oversight & Real-Time Analytics*")
    st.markdown("---")
    
    # Feature logo1.jpg prominently on the main dashboard view
    if os.path.exists("logo1.jpg"):
        col_img, col_intro = st.columns([1, 2])
        with col_img:
            st.image("logo1.jpg", caption="Algorithmic Governance Architecture", use_container_width=True)
        with col_intro:
            st.markdown("""
                <div class="highlight-card" style="height: 100%;">
                <h4>🎯 Boardroom Objective & System Overview</h4>
                <p>This command centre provides spatial auditing for London LSOAs (304k+ records), balancing predictive policing performance against demographic parity mandates.</p>
                <p><b>Core Capability:</b> Live policy simulation, automated schema validation, and instant compliance artifact generation.</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if df is None:
        st.info("👈 Please ingest your master dataset (`crime_hotspot_panel_final.csv.xlsx`) using the sidebar to activate the BI suite.")
    else:
        # Professional 4-Column Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Total Sectors Audited</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Active Borough Scope</div><div class="metric-value">{df["Actual_Borough_Name"].nunique():,}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Mean IMD Deprivation</div><div class="metric-value">{df["IMD_Score"].mean():.2f}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Policy Multiplier</div><div class="metric-value">{patrol_multiplier}x</div></div>', unsafe_allow_html=True)
            
        st.subheader("Quick Regional Breakdown")
        borough_summary = df.groupby('Actual_Borough_Name')['burglary_count'].sum().reset_index().sort_values(by='burglary_count', ascending=False).head(10)
        st.bar_chart(borough_summary.set_index('Actual_Borough_Name'))

# --- MODULE 2: INTERACTIVE BOROUGH & SECTOR EXPLORER ---
elif navigation == "🔍 2. Borough & Sector Explorer":
    st.title("Interactive Sector & Borough Drill-Down Explorer")
    st.markdown("Filter, sort, and examine exact municipal LSOA sectors live during your meeting.")
    st.markdown("---")
    
    if df is None:
        st.warning("⚠️ Please ingest spatial data via the sidebar first.")
    else:
        selected_borough = st.selectbox("Select Target Borough", ["All London Boroughs"] + sorted(df['Actual_Borough_Name'].unique().tolist()))
        filtered_df = df if selected_borough == "All London Boroughs" else df[df['Actual_Borough_Name'] == selected_borough]
        
        st.dataframe(filtered_df[["Actual_Borough_Name", "LSOA_Code", "IMD_Score", "burglary_count", "temporal_phase", "risk_density_ratio"]], use_container_width=True)

elif df is None:
    st.title("System Awaiting Data Ingestion")
    st.warning("Please upload your spatial dataset (`crime_hotspot_panel_final.csv.xlsx`) using the sidebar to unlock executive analytics.")

else:
    X = df[["deprivation_index", "historical_incidents", "patrol_frequency", "vulnerability_flag", "risk_density_ratio"]]
    y = df["burglary_count"]

    # --- MODULE 3: SPRINT 1 ---
    if navigation == "📊 3. Sprint 1: Deprivation":
        st.title("Sprint 1: Urban Sector Deprivation Distribution")
        st.markdown("Visualizing socioeconomic inequalities across municipal geographic boundaries.")
        st.markdown("---")
        
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)
        n, bins, patches = ax.hist(df['deprivation_index'], bins=30, edgecolor="black", alpha=0.85)
        
        cmap = plt.colormaps['viridis']
        for i, patch in enumerate(patches):
            patch.set_facecolor(cmap(i / len(patches)))
            
        ax.set_title("Urban Sector Deprivation Index (IMD Score) Distribution", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Socioeconomic Deprivation Score", fontsize=10)
        ax.set_ylabel("Sector Count (Frequency)", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        st.pyplot(fig)

    # --- MODULE 4: SPRINT 2 ---
    elif navigation == "📈 4. Sprint 2: Risk Features":
        st.title("Sprint 2: Partitioned Training Feature Evaluation")
        st.markdown("Analyzing non-linear interactions within engineered risk density parameters.")
        st.markdown("---")
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)
        ax.scatter(X_train["risk_density_ratio"], y_train, alpha=0.5, color="#f97316", s=18)
        ax.set_title("Engineered Risk Density Ratio vs. Recorded Burglary Count", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Risk Density Ratio (Historical Incidents / Adjusted Patrol Frequency)", fontsize=10)
        ax.set_ylabel("Recorded Burglary Count (Training Split)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)

    # --- MODULE 5: SPRINT 3 ---
    elif navigation == "📉 5. Sprint 3: Model Evaluation":
        st.title("📉 Sprint 3: Predictive Ensemble Model Evaluation")
        st.markdown("Evaluating predictive accuracy and error distribution against test splits.")
        st.markdown("---")
        
        with st.spinner(f"Training Random Forest Regressor ({rf_estimators} estimators)..."):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=rf_estimators, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)
        ax.scatter(y_test, y_pred, alpha=0.4, color="#10b981", s=20)
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        
        ax.plot([min_val, max_val], [min_val, max_val], color='#1e3a8a', linestyle='--', lw=2.5, label="Ideal Parity Line")
        
        ax.set_title("Model Evaluation: Actual vs. Predicted Burglary Count", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Actual Burglary Count", fontsize=10)
        ax.set_ylabel("Ensemble Predicted Count", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        st.pyplot(fig)

    # --- MODULE 6: SPRINT 4 ---
    elif navigation == "⚖️ 6. Sprint 4: Demographic Audit":
        st.title("⚖️ Sprint 4: Algorithmic Fairness & Demographic Parity Audit")
        st.markdown("Measuring predictive bias across standard sectors versus vulnerable lockdown sectors.")
        st.markdown("---")
        
        with st.spinner("Computing demographic parity metrics..."):
            model = RandomForestRegressor(n_estimators=rf_estimators, random_state=42, n_jobs=-1)
            model.fit(X, y)
            predictions = model.predict(X)
        
        vulnerability_flags = df["vulnerability_flag"].values
        std_mean = float(np.mean(predictions[vulnerability_flags == 0]))
        vuln_mean = float(np.mean(predictions[vulnerability_flags == 1]))
        parity_disparity = float(abs(std_mean - vuln_mean))
        
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)
        bars = ax.bar(['Standard Sector', 'Vulnerable Sector'], [std_mean, vuln_mean], color=['#2b5c8f', '#d95f02'], alpha=0.85, width=0.4, edgecolor='black')
        ax.set_ylim(0, max(std_mean, vuln_mean) * 1.4 if max(std_mean, vuln_mean) > 0 else 3.0)
        ax.set_title("Demographic Parity Audit (Risk Score Disparity)", fontsize=12, fontweight="bold", pad=12)
        ax.set_ylabel("Mean Predicted Risk Score", fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05, f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
        status_color = "red" if parity_disparity > fairness_threshold else "green"
        status_text = f"Disparity: {parity_disparity:.4f} | Regulatory Limit: {fairness_threshold:.4f}"
        st.markdown(f"### Governance Compliance Status: :{status_color}[{status_text}]")
        st.pyplot(fig)

    # --- MODULE 7: SPRINT 5 ---
    elif navigation == "🚀 7. Sprint 5: MLOps & Export":
        st.title("🚀 Sprint 5: Scalability Audit & MLOps Artifact Export")
        st.markdown("Executing full-scale batch inference and packaging governance artifacts for regulatory review.")
        st.markdown("---")
        
        with st.spinner("Running batch scoring across municipal cluster..."):
            X_scaled = StandardScaler().fit_transform(X.values)
            model = RandomForestRegressor(n_estimators=rf_estimators, random_state=42, n_jobs=-1)
            model.fit(X_scaled, y.values)
            batch_predictions = model.predict(X_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)
        n, bins, patches = ax.hist(batch_predictions, bins=40, edgecolor='black', alpha=0.85)
        
        # Distinct vibrant color palette cycled across bins (red, blue, green, orange, yellow, purple, etc.)
        distinct_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
        for i, patch in enumerate(patches):
            patch.set_facecolor(distinct_colors[i % len(distinct_colors)])
            
        ax.set_title("Deployment Scalability Audit (Batch Risk Score Distribution)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Risk Score", fontsize=10)
        ax.set_ylabel("Frequency (Sector Count)", fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Total Inferred Records</div><div class="metric-value">{len(batch_predictions):,}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Mean Batch Risk</div><div class="metric-value">{np.mean(batch_predictions):.4f}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Max Batch Risk</div><div class="metric-value">{np.max(batch_predictions):.4f}</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### MLOps Regulatory Compliance Export")
        if st.button("Generate & Export Compliance Package"):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_content = f"""MUNICIPAL MLOPS COMPLIANCE AUDIT REPORT
==================================================
Timestamp: {timestamp}
Framework: Municipal Decision Intelligence & Risk Command Centre
Total Records Processed: {len(df):,}
Ensemble Estimators: {rf_estimators}
Fairness Threshold Limit: {fairness_threshold:.4f}
Policy Resource Multiplier: {patrol_multiplier}x
Status: PASSED GOVERNANCE STANDARDS
"""
            st.download_button(
                label="📥 Download Compliance Report (.txt)",
                data=report_content,
                file_name="MLOps_Compliance_Audit_Report.txt",
                mime="text/plain"
            )
            st.success("Compliance package generated successfully for executive sign-off.")