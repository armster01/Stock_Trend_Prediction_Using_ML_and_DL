# Stock Trend Prediction Using ML & DL

Predict stock price direction and short-term trends using classical machine learning and deep learning models. This repository contains data-preparation code, model training scripts, notebooks, and utilities used to explore and evaluate models for stock trend prediction.

## Table of Contents
- [Project Overview](#project-overview)
- [Highlights](#highlights)
- [Repo Structure](#repo-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Install](#install)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Exploratory Notebooks](#exploratory-notebooks)
  - [Training Models](#training-models)
  - [Inference / Predicting](#inference--predicting)
- [Evaluation](#evaluation)
- [Results & Visualizations](#results--visualizations)
- [Reproducibility & Tips](#reproducibility--tips)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview
The goal is to predict the short-term trend (e.g., up/down/neutral) of stock prices using both traditional ML (e.g., RandomForest, XGBoost) and Deep Learning (LSTM, Transformer, 1D-CNN) approaches. Focus areas include:
- Feature engineering from OHLCV data (technical indicators, rolling stats)
- Time-series model training with appropriate train/validation/test splits
- Backtesting simple strategies and reporting common metrics

## Highlights
- End-to-end pipeline: data collection → preprocessing → training → evaluation → inference
- Examples in Jupyter notebooks for experiments and visualization
- Config-driven training and modular code for adding new models or features

## Repo Structure
- data/                - raw and processed datasets (gitignored)
- notebooks/           - EDA and experiment notebooks
- src/
  - data/              - data loaders & preprocessing
  - features/          - feature engineering functions
  - models/            - model definitions and training loops
  - utils/             - helper scripts (logging, metrics, plotting)
- models/              - saved model checkpoints
- requirements.txt
- README.md

(adapt paths as needed for your repo)

## Getting Started

### Prerequisites
- Python 3.8+ (recommend 3.9 or 3.10)
- pip or conda
- Optional: GPU for deep learning experiments (CUDA + PyTorch/TensorFlow)

### Install
1. Clone the repository:
   git clone https://github.com/armster01/Stock_Trend_Prediction_Using_ML_and_DL.git
   cd Stock_Trend_Prediction_Using_ML_and_DL

2. Create environment and install:
   python -m venv .venv
   source .venv/bin/activate   # Linux / macOS
   .venv\Scripts\activate      # Windows
   pip install --upgrade pip
   pip install -r requirements.txt

3. (Optional) If you prefer conda:
   conda create -n stockml python=3.9
   conda activate stockml
   pip install -r requirements.txt

## Dataset
- This project expects OHLCV data (Open, High, Low, Close, Volume).
- You can download data using yfinance or your preferred provider (AlphaVantage, Kaggle, CSV).
- Example using yfinance:
  ```python
  import yfinance as yf
  df = yf.download("AAPL", start="2018-01-01", end="2024-01-01", interval="1d")
  df.to_csv("data/raw/AAPL.csv")
  ```

- Place raw CSVs in `data/raw/`. Preprocessed files will be saved to `data/processed/`.

## Usage

### Exploratory Notebooks
- Open notebooks in `notebooks/` to run EDA, feature checks, and visualization.
- Recommended: use JupyterLab or VSCode Notebook.

### Training Models
- Example script pattern (adapt to your actual scripts):
  python src/models/train.py --config configs/experiment.yaml

- If you use notebooks for training, copy the config cells and run sequentially.

- Tips:
  - Use a walk-forward or time-based split; avoid random cross-validation for time series.
  - Keep a holdout test set representing the latest time window for final evaluation.

### Inference / Predicting
- Example usage:
  python src/models/predict.py --model models/best_model.pth --ticker AAPL --start 2024-01-01 --end 2024-10-01

- Or use the notebook `notebooks/inference.ipynb` for visualization and backtesting.

## Evaluation
Recommended metrics for classification-style trend prediction:
- Accuracy
- Precision / Recall / F1
- Confusion matrix
- ROC-AUC (if using probability outputs)
- Directional accuracy and simple backtest metrics (cumulative returns, Sharpe ratio)

Include baseline comparisons:
- Naive (previous day direction)
- Moving average crossover
- Random classifier

## Results & Visualizations
- Save plots to `reports/figures/` and summary tables to `reports/metrics/`.
- Provide sample plots: prediction vs actual price, prediction distribution, confusion matrix, and cumulative returns from a simple strategy.

## Reproducibility & Tips
- Set random seeds across numpy, torch, random for reproducibility.
- Use deterministic flags where applicable (note: can impact performance).
- Save model configs, training logs, and the exact dataset hash (e.g., commit CSV or store date/time and query params).
- Consider using Docker for full reproducibility:
  - Provide a Dockerfile or environment.yml for conda.

## Common Pitfalls to Watch For
- Data leakage: ensure features don't use future information.
- Look-ahead bias: use strictly past data for feature calculations.
- Non-stationarity: validate models across multiple time periods and tickers.
- Overfitting: prefer simpler models and regularization; cross-validate across time windows.

## How to Improve This Project
- Add more advanced features (rolling vol, higher timeframes, sentiment features).
- Use hyperparameter tuning (Optuna, Ray Tune).
- Implement robust backtesting with transaction costs and slippage (Backtrader, Zipline, or custom).
- Multi-task models that predict both next-step return and direction.
- Ensembling of models (stacking or voting).

## Contributing
Contributions are welcome. Please:
1. Open an issue describing the feature or bug.
2. Create a branch: git checkout -b feat/your-feature
3. Make changes, add tests or example notebooks, and open a PR.

## License
Specify your license (e.g., MIT). If you don't have one yet, add a LICENSE file.

## Contact
- Author: armster01
- Email: armstersaha8@gmail.com
- GitHub: https://github.com/armster01
  
## References / Citations
- Papers and libraries you used (e.g., LSTM papers, Transformer for time series, yfinance docs, TA-Lib).
