# PyPairs: Statistical Arbitrage Trading System

**Course:** FE 520 - Final Project (Fall 2025)

## 1. Project Overview
**PyPairs** is a Python package designed for **Statistical Arbitrage** (Pairs Trading). Unlike traditional portfolios that rely on market direction (Beta), PyPairs identifies two assets that historically move together, calculates the spread between them, and executes market-neutral trades when they diverge (Mean Reversion).

This package was built to demonstrate:
* **Data Mining:** Finding statistical relationships in the auto/tech sectors.
* **Mathematical Modeling:** Using Linear Regression to calculate Hedge Ratios and Z-Scores.
* **Simulation:** Backtesting trading strategies over historical data.
* **Optimization:** Algorithmic tuning of entry thresholds.
* **Risk Management:** Calculating Institutional metrics like Sharpe Ratio and Max Drawdown.

---

## 2. Key Features & Modules

The package is modularized into five core components:

* **`DataEngine` (Module B):** Handles data ingestion from Yahoo Finance (`yfinance`) and performs correlation analysis to identify the "best fit" stock pair. It calculates the spread Z-Score using linear regression.
* **`Backtester` (Module C):** The simulation engine. It iterates through historical data, opening Long/Short positions based on Z-Score signals, and tracks the daily Profit & Loss (PnL).
* **`RiskMetrics` (Module D):** A financial calculator that assesses the safety of the strategy. It computes the **Sharpe Ratio** (risk-adjusted return) and **Maximum Drawdown** (worst-case loss).
* **`StrategyOptimizer` (Module E):** An optimization loop that tests multiple entry thresholds (e.g., Z=1.0 to 3.0) to scientifically determine the most profitable trade trigger.
* **`Visualizer` (Module A):** Generates a professional 3-panel financial dashboard displaying the Trading Signals, Equity Curve, and Optimization Grid.

---

## 3. Installation Instructions

### Prerequisites
* Python 3.12 or higher
* Internet connection (for fetching stock data)

### Option A: Using `pip` (Standard)
1.  Clone or unzip the project repository.
2.  Navigate to the project root directory in your terminal.
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Alternatively, install the package in editable mode:*
    ```bash
    pip install -e .
    ```

### Option B: Using `uv` (Modern/Fast)
If you have `uv` installed (recommended for this project):
```bash
uv sync
````

-----

## 4\. Usage Instructions

### Running the Full Pipeline

The easiest way to run the project is via the `main.py` script, which orchestrates the entire workflow from data download to visualization.

1.  **Run the script:**

    ```bash
    python main.py
    ```

    *(Or if using uv: `uv run main.py`)*

2.  **What to Expect:**

      * The system will download 1 year of data for **Ford (F), GM, Toyota (TM), and Honda (HMC)**.
      * It will automatically select the most correlated pair (e.g., GM & F).
      * It will run an optimization loop to find the best trading threshold.
      * It will print a **Summary Report** to the console (Profit, Sharpe Ratio, etc.).
      * Finally, a **Dashboard Window** will appear with charts.

### Using as a Library

You can also import individual modules into your own scripts:

```python
from pypairs.data_engine import DataEngine
from pypairs.backtester import Backtester

# 1. Initialize Engine
engine = DataEngine(verbose=True)

# 2. Get Z-Score Data
df, pair_info = engine.run_full_pipeline(['AAPL', 'MSFT', 'GOOG'], period='1y')

# 3. Run a Backtest
bt = Backtester()
results = bt.run_backtest(df, entry_threshold=2.0)

print(f"Final Profit: ${results['Cumulative PnL'].iloc[-1]:.2f}")
```

-----

## 5\. Design Decisions & Error Handling

  * **No "Black Box" Libraries:** We strictly avoided `scikit-learn` or `scipy` for core logic. All statistical calculations (Regression, Z-Score, Sharpe Ratio) were implemented manually using `numpy` to demonstrate mathematical understanding.
  * **Robust Data Ingestion:** The `DataEngine` includes `try-except` blocks to handle network failures or invalid tickers. It explicitly checks for `NaN` values and aligns time-series data before processing.
  * **Vectorization:** Where possible, `pandas` and `numpy` vectorization was used instead of Python loops for performance efficiency.

-----

## 6\. Credits

  * **Pradhyumna:** System Architect, Visualization, & Integration (`main.py`, `visualizer.py`)
  * **Thejas:** Data Mining & Math Engine (`data_engine.py`)
  * **Pallavi:** Trading Simulation Logic (`backtester.py`)
  * **Rishika:** Risk Management Analysis (`risk_engine.py`)
  * **Sirideep:** Parameter Optimization (`optimizer.py`)
