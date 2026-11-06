# 🛰️ Solar PV Digital Twin & SCADA Anomaly Detection System

A predictive analytics platform for solar photovoltaic power plants that combines classical ML, deep learning, generative models, and graph neural networks to monitor inverter health, forecast power output, and detect operational anomalies in real time.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models & Techniques](#models--techniques)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Notebooks](#running-the-notebooks)
- [Running the Dashboard](#running-the-dashboard)
- [Running Tests](#running-tests)
- [Results](#results)

---

## Overview

This project implements a **Solar Digital Twin** — a software replica of a real-world solar PV plant — that tracks, predicts, and cross-validates inverter performance using multiple AI models. The system ingests 15-minute interval generation and weather sensor data from two solar plants and provides:

- **Power Forecasting** using XGBoost (baseline) and a Bidirectional LSTM
- **Unsupervised Anomaly Detection** using TadGAN (Time-series Anomaly Detection with Generative Adversarial Networks)
- **Fleet Spatial Cross-Validation** using a Graph Attention Network (GAT) across all inverters
- **Peer-Relative Ranking** to flag persistent underperformers
- **Mathematically rigorous anomaly thresholding** via Extreme Value Theory (EVT), Gaussian Mixture Models (GMM), and Bayesian cost-optimal calibration — replacing naive percentile cutoffs
- **Interactive SCADA Dashboard** built with Streamlit and Plotly for live telemetry exploration, fault injection simulation, and decision-support

---

## Project Structure

```
project/
├── dataset/
│   ├── Plant_1_Generation_Data.csv      # Plant 1 inverter-level AC/DC power readings
│   ├── Plant_1_Weather_Sensor_Data.csv  # Plant 1 irradiation, ambient & module temperature
│   ├── Plant_2_Generation_Data.csv      # Plant 2 inverter-level AC/DC power readings
│   └── Plant_2_Weather_Sensor_Data.csv  # Plant 2 weather sensor data
│
├── notebooks/
│   ├── 01_EDA_and_Clustering.ipynb               # Exploratory analysis & DTW time-series clustering
│   ├── 02_XGBoost_Model_Training_and_Evaluation.ipynb  # Baseline XGBoost training & evaluation
│   ├── 03_BiLSTM_Model_Training_and_Evaluation.ipynb   # Bidirectional LSTM training & evaluation
│   ├── 04_TadGAN_Anomaly_Detection.ipynb               # TadGAN training & anomaly scoring
│   └── 05_Grid_Threshold_Graph_Analysis.ipynb           # Environmental grid, GAT, and threshold analysis
│
├── src/
│   ├── models/
│   │   ├── baseline.py        # XGBoost training, evaluation, and prediction utilities
│   │   ├── sequence.py        # BiLSTM model architecture and training loop
│   │   ├── generative.py      # TadGAN: Encoder, Generator, CriticX, CriticZ, WGAN-GP training
│   │   └── graph_network.py   # InverterGAT: Graph Attention Network for fleet cross-validation
│   ├── clustering.py          # DTW-based time-series K-Means clustering of inverter daily profiles
│   ├── dashboard.py           # Streamlit SCADA dashboard (main app entry point)
│   ├── data_fusion.py         # Data loading and feature engineering pipelines
│   ├── grid_optimization.py   # 2D environmental grid (irradiance × temperature) residual normalization
│   ├── optimization.py        # Dispatch optimization utilities
│   ├── peer_ranking.py        # Percentile-based peer ranking and underperformer flagging
│   ├── simulation.py          # FleetSimulator and fault injection engine
│   └── threshold_model.py     # EVT (GPD), GMM, and cost-optimal anomaly threshold fitting
│
├── tests/
│   ├── conftest.py
│   ├── test_graph_network.py
│   ├── test_grid_optimization.py
│   ├── test_peer_ranking.py
│   ├── test_simulation.py
│   └── test_threshold_model.py
│
├── results/                   # Generated model outputs, predictions CSVs, saved model weights
│   ├── saved_models/          # Serialized model files (.pt, .joblib)
│   ├── plant1_xgb/            # XGBoost result plots for Plant 1
│   └── plant2_xgb/            # XGBoost result plots for Plant 2
│
└── requirements.txt
```

---

## Dataset

The dataset contains operational data from **two solar power plants** (Plant IDs: 4135001 and 4136001) sampled at **15-minute intervals**.

| File | Description |
|---|---|
| `Plant_X_Generation_Data.csv` | Per-inverter DC power, AC power, daily yield, total yield |
| `Plant_X_Weather_Sensor_Data.csv` | Ambient temperature, module temperature, irradiation |

**Key features used for modelling:**

| Feature | Description |
|---|---|
| `AMBIENT_TEMPERATURE` | Atmospheric temperature (°C) |
| `MODULE_TEMPERATURE` | PV module surface temperature (°C) |
| `IRRADIATION` | Solar irradiance (kW/m²) |
| `TIME_OF_DAY` | Encoded time slot within the day |
| `AC_POWER` | Target — inverter AC output (kW) |

---

## Models & Techniques

### 1. XGBoost (Baseline)
- Gradient boosted regression on tabular weather features
- Chronological 70 / 15 / 15 train-val-test split
- Metrics: MAE, R²

### 2. Bidirectional LSTM (BiLSTM)
- Sequence model capturing temporal dependencies in both forward and backward directions
- Sequence length auto-tuned by cross-validated MAE across `[8, 16, 48, 96]` steps
- Architecture: 2-layer BiLSTM → Linear head
- Trained with MAE loss and Adam optimizer

### 3. TadGAN (Anomaly Detection)
- Generative Adversarial Network adapted for unsupervised time-series anomaly detection
- Components: **Encoder** (BiLSTM) + **Generator** (LSTM decoder) + **CriticX** + **CriticZ**
- Training objective: WGAN-GP (Wasserstein GAN with Gradient Penalty)
- Anomaly score = weighted combination of **reconstruction error** and **critic score**

### 4. Graph Attention Network (GAT)
- Fleet-wide spatial cross-validation across all inverters
- Adjacency matrix built from pairwise Pearson correlation (threshold ≥ 0.70)
- 2-layer GAT with multi-head attention + peer-consensus residual skip connection
- Flags inverters whose output deviates from their graph-neighbourhood prediction

### 5. DTW Time-Series Clustering
- K-Means clustering of daily AC power profiles using Dynamic Time Warping (DTW) distance
- Identifies distinct operational behaviour patterns across the inverter fleet

### 6. Mathematical Anomaly Thresholding
Replaces the naive 98th-percentile cutoff with statistically principled methods:

| Method | Description |
|---|---|
| **EVT (GPD)** | Generalized Pareto Distribution fitted to score tail exceedances |
| **GMM** | Gaussian Mixture posterior crossing point |
| **Cost-Optimal** | Bayesian threshold minimising false-alarm and missed-detection costs |

### 7. 2D Environmental Grid
- Bins operating conditions by irradiance × module temperature quantiles
- Computes per-cell median power and MAD for residual normalization
- Prevents false alarms during benign cloud-edge or thermal-derating transients

---

## Requirements

- **Python**: 3.10+
- **PyTorch**: 2.6.0 (CUDA 12.4 build — see note below)

**Full dependency list (`requirements.txt`):**

```
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.6.0
pandas
scikit-learn
tslearn
xgboost
jupyter
matplotlib
seaborn
tqdm
scipy
networkx
pytest
plotly
streamlit
python-docx
python-pptx
```

> **Note on PyTorch:** The `requirements.txt` pulls the CUDA 12.4 build of PyTorch. If you do not have an NVIDIA GPU or have a different CUDA version, replace the index URL or install the CPU-only build:
> ```
> pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Installation

### 1. Clone the repository

```bash
git clone <https://github.com/AthJa/predictive-project-final>
cd "project"
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Notebooks

Notebooks must be executed **in order** — each notebook saves intermediate results (CSVs, model weights) that the subsequent ones and the dashboard depend on.

```bash
# Start Jupyter
jupyter notebook
```

Then open and run the notebooks in this sequence:

| # | Notebook | Purpose |
|---|---|---|
| 1 | `01_EDA_and_Clustering.ipynb` | Data exploration, preprocessing, DTW clustering |
| 2 | `02_XGBoost_Model_Training_and_Evaluation.ipynb` | Train & evaluate XGBoost; saves `results/xgboost_predictions.csv` |
| 3 | `03_BiLSTM_Model_Training_and_Evaluation.ipynb` | Train & evaluate BiLSTM; saves `results/bilstm_predictions.csv` |
| 4 | `04_TadGAN_Anomaly_Detection.ipynb` | Train TadGAN; saves `results/tadgan_plant*_anomalies.csv` |
| 5 | `05_Grid_Threshold_Graph_Analysis.ipynb` | Environmental grid, GAT analysis, threshold comparison |

---

## Running the Dashboard

The Streamlit dashboard requires the prediction CSVs produced by notebooks 2–4 to be present in the `results/` directory.

```bash
# From the project root directory
streamlit run src/dashboard.py
```

The dashboard will open at **http://localhost:8501** and provides:

- **Fleet Overview** — Aggregate plant output vs. digital-twin baseline, peer ranking heatmap
- **Inverter Deep Dive** — Per-inverter time series, residual track, TadGAN anomaly score overlay
- **Advanced Analytics** — Interactive threshold explorer (EVT / GMM / Cost-Optimal), 2D environmental grid, GAT fleet topology graph
- **Simulation & Demo Lab** — Live fault injection scenarios (soiling, bypass diode, cloud transient, panel shading) with side-by-side naive vs. dual-signal detection verdicts

---

## Running Tests

```bash
# Run the full test suite from the project root
pytest tests/ -v
```

Individual test modules:

```bash
pytest tests/test_threshold_model.py -v
pytest tests/test_graph_network.py -v
pytest tests/test_peer_ranking.py -v
pytest tests/test_grid_optimization.py -v
pytest tests/test_simulation.py -v
```

---

## Results

Trained model artefacts are saved to `results/saved_models/`:

| File | Description |
|---|---|
| `xgboost_model.joblib` | Trained XGBoost regressor |
| `bilstm_model.pt` | Trained BiLSTM weights |
| `bilstm_feature_scaler.joblib` | Feature scaler for BiLSTM inputs |
| `bilstm_target_scaler.joblib` | Target scaler for BiLSTM outputs |
| `tadgan_encoder.pt` | TadGAN encoder weights |
| `tadgan_generator.pt` | TadGAN generator weights |
| `tadgan_cx.pt` | TadGAN CriticX weights |
| `tadgan_cz.pt` | TadGAN CriticZ weights |

Prediction CSVs and diagnostic plots are stored in `results/`.

