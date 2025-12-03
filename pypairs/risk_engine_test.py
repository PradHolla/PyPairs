import pandas as pd
import numpy as np
from risk_engine import RiskMetrics

#Dummy Cumulative PnL Data

np.random.seed(42)

# Simulate a noisy upward-sloping equity curve
daily_changes = np.random.normal(loc=0.5, scale=2.0, size=200)   # random PnL changes
cumulative_pnl = pd.Series(daily_changes).cumsum()

print("Sample of Cumulative PnL:")
print(cumulative_pnl.head())


risk = RiskMetrics()


metrics = risk.calculate_metrics(cumulative_pnl)

print("\n=== RISK METRICS ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
