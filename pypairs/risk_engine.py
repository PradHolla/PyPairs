# pypairs/risk_engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
import pandas as pd


NumberLike = Union[pd.Series, pd.DataFrame, np.ndarray, list]


@dataclass
class RiskMetrics:
    """
    Risk analysis utilities for the PyPairs strategy.

    Expected input:
        - A 1D series representing the *equity curve* of the strategy
          (e.g., 'Cumulative PnL' from the Backtester), OR
        - A 1D series of daily PnL values.

    Assumptions:
        - Max Drawdown is computed on the provided series as if it were
          the equity curve.
        - Daily returns for Sharpe are approximated as the first difference
          of that series (i.e., day-to-day PnL changes).
    """

    trading_days_per_year: int = 252

    def calculate_metrics(self, pnl_series: NumberLike) -> Dict[str, float]:
        """
        Calculate key risk metrics: Max Drawdown and Sharpe Ratio.

        Parameters
        ----------
        pnl_series : array-like or pd.Series
            Typically the 'Cumulative PnL' column from the Backtester
            (or any equity curve). Can also be a series of daily PnL.

        Returns
        -------
        Dict[str, float]
            {
                'Max Drawdown': float (negative number, e.g. -0.15),
                'Sharpe Ratio': float
            }
        """
        # Convert to a clean pandas Series
        pnl = pd.Series(pnl_series).dropna()

        if len(pnl) < 2:
            # Not enough data to compute meaningful metrics
            return {"Max Drawdown": 0.0, "Sharpe Ratio": 0.0}

        # ---------- Max Drawdown ----------
        # Treat `pnl` as an equity curve (e.g., cumulative PnL)
        equity_curve = pnl.astype(float)

        running_max = equity_curve.cummax()
        # Avoid division by zero – if running_max is zero, set drawdown to 0
        drawdown = (equity_curve - running_max) / running_max.replace(0, np.nan)
        drawdown = drawdown.fillna(0.0)

        max_drawdown = float(drawdown.min())  # This will be <= 0

        # ---------- Sharpe Ratio ----------
        # Approximate "daily returns" as day-to-day PnL changes
        daily_changes = equity_curve.diff().fillna(0.0)

        mean_daily = daily_changes.mean()
        std_daily = daily_changes.std(ddof=1)

        if std_daily == 0 or np.isnan(std_daily):
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = float(
                (mean_daily / std_daily) * np.sqrt(self.trading_days_per_year)
            )

        return {
            "Max Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe_ratio,
        }
