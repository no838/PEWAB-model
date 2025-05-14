
# PEWAB Model for Algal Bloom Forecasting

This repository contains the complete implementation of the PEWAB (Precipitation–Environment–Water–Algal–Bloom) modeling framework, designed to predict chlorophyll-a concentration in inland lakes and rivers using environmental monitoring data and causal inference.

## 🧠 Model Overview

The model integrates:
- **Feature engineering** (including lag variables and rolling cumulative indicators)
- **Causal structure learning** via PCMCI (ParCorr)
- **Feature selection** using Random Forest
- **Four deep learning models**:
  - LSTM (Long Short-Term Memory)
  - GCN (Graph Convolutional Network) for causal-graph-informed learning (PEWAB core)
  - CNN (1D Convolutional Neural Network)
  - BNN (Bayesian Neural Network)
- **Multi-lead forecasting (1–7 days)**
- **Ensemble performance comparison**
- **Per-section evaluation across multiple monitoring sites**

## 📁 Repository Structure

- `main.py`: Main modeling script (provided above)
- `final_daily_CHaonly_means_with_meteorological_with_air_quality_data.csv`: Input dataset
- `/output`: Model results and figures
- `/output/figures`: Plots of feature importances and prediction results

## 📊 Input Data

This dataset contains high-frequency monitoring data (4-hour interval) from national control sections  in China, including:
- Water quality (TN, TP, DO, turbidity, chlorophyll-a, etc.)
- Meteorological data (temperature, precipitation, wind speed)

**Chlorophyll-a** was measured following the national standard:
> HJ 897-2017: Water quality — Determination of chlorophyll-a — Spectrophotometric method.

## 🔧 How to Run

1. Place your dataset in the same directory as `main.py`
2. Adjust file names if needed in `data_file = 'final_daily_...csv'`
3. Run the script using Python 3.8+ with the following dependencies:

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn networkx tigramite torch-geometric
```

4. Check `output/` and `output/figures/` for results

## 📜 License

This project is shared under the MIT License.

## 📬 Contact

For questions, please contact [yourname@domain.com] or raise an issue.
