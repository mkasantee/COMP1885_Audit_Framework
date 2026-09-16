# Algorithmic Fairness Data Quality Auditing and Spatial Risk Prediction: Bias Mitigation and Scalable Burglary Forecasting in Post Pandemic London (2019-2024)

## Executive Summary
This repository houses an enterprise-grade machine learning evaluation, algorithmic fairness auditing, and batch deployment pipeline designed to analyse spatiotemporal burglary forecasting across urban municipal sectors in London. Moving beyond naive predictive modelling, this framework systematically addresses institutional data bias, quantifies disparate impact across socioeconomically vulnerable communities, and stress-tests batch inference scalability for secure municipal deployment.

---

## Table of Contents
1. System Architecture & Pipeline Flow
2. Industry-Ready Machine Learning Skills & Core Tooling
3. Sprint Engineering Deep Dive
4. Empirical Findings & Visual Artefacts
5. Data Access and Recreation
6. Academic Report & Reference Corpus
7. Repository Structure
8. System Installation & Execution Guide

---

## 1. System Architecture & Pipeline Flow
The framework is engineered as a modular, sequential pipeline spanning five distinct operational sprints:
* Sprint 1 (Data Ingestion): Ingestion of spatial telemetry and creation of baseline municipal sector grids mapped against deprivation indices.
* Sprint 2 (Feature Engineering): Construction of domain-specific variables, notably the Risk Density Ratio (historical_incidents / [patrol_frequency + 1]) to quantify police resource strain.
* Sprint 3 (Predictive Modelling): Training of an optimised RandomForestRegressor ensemble to forecast burglary frequencies with rigorous test-split evaluation.
* Sprint 4 (Fairness Audit): Execution of a demographic parity audit measuring systemic risk score inflation between standard and vulnerable sectors.
* Sprint 5 (Deployment Scalability): Batch inference stress testing processing 2,000 spatial records simultaneously to evaluate distribution bounds and hotspot outliers.

---

## 2. Industry-Ready Machine Learning Skills & Core Tooling
This architecture demonstrates mastery across key technical domains required for advanced machine learning engineering:
* Advanced Python Programming: Clean, modular code utilising advanced data structures and vectorised array operations.
* Supervised Machine Learning & Ensemble Modelling: Building, tuning, and evaluating non-linear predictive architectures using scikit-learn (RandomForestRegressor).
* Algorithmic Auditing & Responsible AI: Implementing demographic parity compliance frameworks, quantifying statistical disparities, and unmasking disparate impact to mitigate feedback loops.
* Enterprise Visualisation: Producing publication-grade, stakeholder-ready visual telemetry assets using Matplotlib, Altair, and Streamlit.

---

## 3. Sprint Engineering Deep Dive
* Why Random Forest? Non-linear spatial relationships and complex feature interactions require a robust ensemble method capable of handling tabular municipal telemetry without overfitting to local grid anomalies.
* The Risk Density Feature: By evaluating historical incident pressure relative to active patrol frequencies, the model isolates areas of under-resourcing versus true crime density.
* Algorithmic Disparity Auditing: The framework calculates statistical parity disparity to unmask hidden feedback loops where historical data biases risk scoring against disadvantaged areas.

---

## 4. Empirical Findings & Visual Artefacts

### Model Evaluation (Sprint 3)
* Insight: The model displays conservative central tendency behaviour, accurately tracking baseline risk while under-estimating extreme upper-tier hotspots (6+ burglaries).

### Demographic Parity Audit (Sprint 4)
* Standard Sector Mean Risk: 2.0963
* Vulnerable Sector Mean Risk: 2.3215
* Statistical Parity Disparity: 0.2252 (High Disparate Impact Flag)

### Batch Deployment Scalability (Sprint 5)
* Total Records Processed: 2000
* Mean Predicted Risk Score: 2.0730
* Max Risk Score (Outlier Hotspot): 13.3690

---

## 5. Data Access and Recreation (COMP1885 Requirement)
The primary dataset comprises post-pandemic London burglary records, OpenStreetMap data, and London Lower Layer Super Output Area shapefiles. 

Due to strict Moodle file size limits, a representative sample dataset comprising 2,000 geospatial records (crime_hotspot_panel_final.csv.xlsx) is included directly within this directory to demonstrate functionality. To recreate the full master dataset (304,268 LSOA-month panel observations across 4,988 spatial units), the ingestion pipeline automatically downloads the required spatial boundary shapefiles and compiles the SQLite relational schema upon full enterprise execution.

---

## 6. Academic Report & Reference Corpus
* Project Dissertation: COMP1885 Individual Project Report.pdf
* Key Academic References & Literature:
  - Barocas, S., Hardt, M., & Narayanan, A. (2023). Fairness and Machine Learning: Limitations and Opportunities.
  - Richardson, R., Schultz, J. M., & Crawford, K. (2019). Dirty Data, Bad Predictions: How Civil Rights Violations Impact Police Data, Predictive Policing Systems, and Justice.
  - Brayne, S. (2017). Big Data Surveillance: The Case of Policing.

---

## 7. Repository Structure
COMP1885_Audit_Framework/
|-- app.py                               # Main Streamlit executive dashboard application
|-- extract_master_features.py           # Extracts clean panel data for academic reporting
|-- sprint1_data_ingestion.py            # Executes raw spatiotemporal data validation
|-- sprint1_plot_data_distribution.py    # Generates deprivation distribution visualisations
|-- sprint2_feature_engineering.py       # Engineers risk interaction terms and partitions data
|-- sprint2_plot_feature_engineering.py  # Generates training split relationship visualisations
|-- sprint3_model_evaluation.py          # Evaluates predictive MSE and R-Squared metrics
|-- sprint3_plot_model_performance.py    # Generates model evaluation comparison visualisations
|-- sprint4_fairness_audit.py            # Computes statistical parity disparity between sectors
|-- sprint4_plot_fairness_audit.py       # Generates demographic parity disparity visualisations
|-- sprint5_deployment_pipeline.py       # Initialises production-grade batch inference scaling
|-- sprint5_plot_deployment_scalability.py # Generates batch risk score distribution visualisations
|-- crime_hotspot_panel_final.csv.xlsx   # Representative sample dataset
|-- README_1885.txt                      # Executive project documentation

---

## 8. System Installation & Execution Guide

### Required Package Dependencies
To successfully execute the VeriForce command centre and all associated sprint scripts, the following Python packages must be downloaded and installed in your environment:
* streamlit (Web application framework)
* pandas (Data manipulation and analysis)
* numpy (Numerical computing operations)
* scikit-learn (Machine learning ensemble architecture)
* matplotlib (Static publication-grade plotting)
* altair (Interactive statistical visualisations)
* openpyxl (Engine required to read .xlsx sample data files)
* requests (Webhook API transmission protocols)

### Step 1: Open Project in Visual Studio Code
Extract the submitted zip folder. Open Visual Studio Code, select "File > Open Folder", and open the root folder containing the project files.

### Step 2: Install Required Packages
In Visual Studio Code, open a new terminal window (Terminal > New Terminal). Copy and paste the following command into the terminal to install all required dependencies systematically:

pip install streamlit pandas numpy scikit-learn matplotlib altair openpyxl requests

### Step 3: Execute the Application
Once the package installation is complete, launch the VeriForce Municipal Risk Command Centre by copying and pasting the following two lines directly into the terminal:

cd COMP1885_Audit_Framework
py -m streamlit run app.py

### Live Cloud Deployment (Alternative Access)
Alternatively, the full interactive application has been deployed to the cloud for immediate executive stakeholder access and examiner review. The live production environment requires no local installation and can be accessed directly via the following URL:
https://mkasantee-comp1885-audit-framework-app-qxxcki.streamlit.app