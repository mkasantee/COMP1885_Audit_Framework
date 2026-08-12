# From Deprivation Mapping to Scalable Deployment: A 5-Sprint Spatiotemporal Audit Framework for Post-Pandemic Burglary Forecasting, Algorithmic Fairness, and Risk Governance in London (2019–2024)

> **Author:** Michelle Asante | **Role:** Lead Auditor & Scrum Master
> 
> 
> **Institution:** University of Greenwich | **Module:** COMP 1885 Project
> 
> 
> **Domain:** Enterprise Architecture & AI Governance Framework

---

## 1. Executive Summary

This repository houses a production-ready machine learning audit and MLOps framework designed to evaluate urban spatiotemporal burglary forecasting models across London (2019–2024). Moving beyond naive predictive accuracy, this framework operationalises **AI & ML Security, Algorithmic Fairness, and Risk Governance**.

Designed as a structured blueprint for public-sector deployment, the system systematically stress-tests models against historical data anomalies, structural socioeconomic bias, demographic parity violations, and batch inference scalability constraints.

---

## 2. Core Architecture: The 5-Sprint Lifecycle

The project is structured into five sequential engineering sprints, establishing a traceable audit trail from raw spatial ingestion to batch deployment:

* **Sprint 1 (Data Integrity & Deprivation Mapping):** Establishes the foundational data pipeline, versioning raw municipal telemetry and mapping the socioeconomic deprivation index distribution across urban sectors.


* **Sprint 2 (Feature Engineering & Risk Density):** Engineers custom spatiotemporal features, notably the Risk Density Ratio (`historical_incidents / patrol_frequency`), establishing mathematical baselines and structural feature importance.


* **Sprint 3 (Model Evaluation & Error Profiling):** Deploys an optimised 'RandomForestRegressor' ensemble to forecast burglary counts, executing rigorous test-split residual analysis and central tendency profiling.


* **Sprint 4 (Algorithmic Fairness & Demographic Parity Audit):** Audits disparate impact by comparing risk scores across standard versus socioeconomically vulnerable sectors, uncovering and quantifying algorithmic bias ($0.2252$ statistical parity disparity) to prevent feedback loops in municipal resource allocation.


* **Sprint 5 (Deployment Scalability & Batch Inference):** Stress-tests production readiness by processing 2,000 spatial records simultaneously, analyzing right-skewed risk distributions and maximum hotspot volatility under containerised parameters.



---

## 3. Repository Directory Structure


COMP1885_Audit_Framework/
│
├── sprint1_data_ingestion.py            # Raw telemetry ingestion & data versioning hooks
├── sprint1_plot_data_distribution.py    # Urban sector deprivation index distribution visualization
├── sprint2_feature_engineering.py       # Risk density ratio calculation & feature extraction
├── sprint2_plot_feature_engineering.py  # Partitioned training feature visualization script
├── sprint3_model_evaluation.py          # Random Forest training and empirical test split evaluation
├── sprint3_plot_model_performance.py    # Actual vs. predicted error profiling visualization script
├── sprint3_model_performance.png        # Generated model error distribution artifact
├── sprint4_fairness_audit.py            # Demographic parity and disparate impact computation
├── sprint4_plot_fairness_audit.py       # Fairness disparity risk scoring visualization script
├── sprint4_fairness_parity.png          # Group disparity bar chart audit artifact
├── sprint5_deployment_pipeline.py       # Batch inference engine and scalability stress-test
├── sprint5_plot_deployment_scalability.py # Batch risk score distribution script
├── sprint5_deployment_distribution.png  # Scalability histogram artifact
├── Dockerfile                           # Containerisation specification for cross-environment portability
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Executive project documentation

```

---

## 4. Environment Setup Instructions

To ensure seamless execution and prevent missing dependency errors (`ModuleNotFoundError`), follow these sequential setup commands in your terminal:

```powershell
# Step 1: Initialise an isolated Python virtual environment
python -m venv venv

# Step 2: Activate the virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# Step 3: Install all required project dependencies
pip install -r requirements.txt

```

---

## 5. Sequential Execution Guidelines

Run the visual plotting and audit pipelines sequentially via the terminal to validate model performance and generate all telemetry artefacts:

```powershell
# Sprint 1: Generate Deprivation Mapping Distribution
.\venv\Scripts\python sprint1_plot_data_distribution.py

# Sprint 2: Generate Feature Engineering & Risk Density Plot
.\venv\Scripts\python sprint2_plot_feature_engineering.py

# Sprint 3: Generate Model Performance & Actual vs. Predicted Error Profiling
.\venv\Scripts\python sprint3_plot_model_performance.py

# Sprint 4: Execute Fairness Audit & Demographic Parity Disparity Plot
.\venv\Scripts\python sprint4_plot_fairness_audit.py

# Sprint 5: Execute Deployment Scalability & Batch Risk Distribution Stress Test
.\venv\Scripts\python sprint5_plot_deployment_scalability.py

```

---

## 6. MLOps & Governance Integration

* **Data Version Control (DVC):** Ensures absolute provenance and reproducibility of large-scale municipal datasets across development and audit lifecycles.


* **Fairness Frameworks:** Integrates automated parity constraints compliant with legislative standards (e.g., Equality Act principles) to mitigate algorithmic discrimination.


* **Containerisation:** Fully containerised via `Dockerfile` architecture to guarantee zero-friction deployment across cloud infrastructure and law enforcement data platforms.