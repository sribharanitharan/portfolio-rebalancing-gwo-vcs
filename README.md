# ML-Guided Hybrid GWO-VCS Portfolio Rebalancing Framework

An advanced, machine learning-guided investment portfolio rebalancing and optimization system. This project combines state-of-the-art metaheuristic optimization algorithms—**Grey Wolf Optimization (GWO)** and **Virus Colony Search (VCS)**—with **Machine Learning models** for intelligent search space reduction, enabling optimal asset allocation under real-world financial constraints.

---

## 📈 System Architecture & Features

This framework contains a modular pipeline built to automate data processing, predictive asset selection, and metaheuristic weight allocation:

1. **Robust Data Pipeline**:
   - Downloads historical stock data from Yahoo Finance.
   - Computes logarithmic returns, covariance matrices, and expected returns.
   - Engineering features (e.g., momentum, moving averages, volatility indicators) to feed the ML models.
   
2. **Machine Learning-Guided Search Space Reduction**:
   - Train predictive models (Random Forest, MLP, or Gradient Boosting) to score assets based on future performance indicators.
   - Filters the candidate universe to the top-k highest-scoring assets.
   - Significantly reduces the optimization dimensionality, accelerating convergence and improving solution quality.

3. **Hybrid Metaheuristic Optimization (GWO + VCS)**:
   - **Grey Wolf Optimizer (GWO)**: Models the social hierarchy and hunting behavior of grey wolves to perform exploration and exploitation of the search space.
   - **Virus Colony Search (VCS)**: Models the infection and replication behavior of viruses to find local/global optimal solutions.
   - **Hybrid GWO-VCS**: Combines the rapid convergence of GWO with the mutation and diffusion steps of VCS to avoid local minima.

4. **Real-world Portfolio Constraints**:
   - Enforces cardinality limits (minimum/maximum number of active assets).
   - Dynamic weight boundaries (upper and lower bounds per asset).
   - Portfolio turnover limits to minimize unnecessary rebalancing.
   - Configurable transaction costs.
   - Supports multiple objective functions: Sharpe Ratio, Sortino Ratio, and Minimum Variance.

5. **Interactive Dashboard**:
   - A fully-featured **Streamlit app** to run backtests, compare models, visualize portfolio weights, and analyze metrics in real-time.

---

## 📂 Project Structure

```text
├── data/
│   ├── raw/                  # Downloaded raw stock price datasets
│   └── processed/            # Computed returns, covariance, and ML features
├── models/                   # Serialized ML models (.pkl)
├── results/                  # Generated portfolio weights and historical backtests
├── src/
│   ├── algorithms/           # Implementations of GWO, VCS, and Hybrid GWO-VCS
│   ├── data_pipeline/        # Data loading, preprocessing, and Gym-like Environment
│   ├── ml_predictor/         # ML feature extractor, region predictor, and search space reducer
│   ├── portfolio/            # Portfolio objective functions, constraints, and rebalancing logic
│   └── evaluation/           # Backtester and risk metric calculators (Sharpe, Sortino, etc.)
├── requirements.txt          # Python dependencies
├── app.py                    # Streamlit Dashboard UI
└── main.py                   # Command-line entry point
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sribharanitharan/portfolio-rebalancing-gwo-vcs.git
   cd portfolio-rebalancing-gwo-vcs
   ```

2. **Set up a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the System

### 1. Command-Line Execution
Run the full backtest pipeline via the terminal:
```bash
python main.py
```
This runs the entire flow: downloads data, runs the ML space reduction, optimizes asset weights across each rebalancing period, saves results, and prints the performance metrics.

### 2. Run the Interactive Dashboard
Launch the Streamlit web interface to run live scenarios and visualize results dynamically:
```bash
streamlit run app.py
```

---

## 📊 Evaluation Metrics Analyzed
- **Cumulative Return**
- **Sharpe Ratio & Sortino Ratio**
- **Maximum Drawdown (MDD)**
- **Turnover Rate**
- **Total Transaction Costs Incurred**
