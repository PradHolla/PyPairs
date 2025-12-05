"""
Parameter optimizer for PyPairs (Person E)

Implements a simple threshold search over entry Z-Score values by
running the provided Backtester class on the DataEngine output.

Author: Person E (Parameter Optimizer)
"""
from typing import List

import numpy as np
import pandas as pd


class StrategyOptimizer:
    """Search entry thresholds and record final profit.

    Methods
    -------
    optimize_thresholds(backtester_class, data)
        Runs backtester for thresholds 1.0 -> 3.0 step 0.2 and returns a
        DataFrame with 'Threshold' and 'Final Profit'.
    """

    def __init__(self):
        pass

    def optimize_thresholds(self, backtester_class, data: pd.DataFrame, exit_threshold: float = 0.0) -> pd.DataFrame:
        """
        Run backtests for entry thresholds from 1.0 to 3.0 (step 0.2).

        Args:
            backtester_class: A Backtester class or an instance with method `run_backtest(data, entry_threshold, exit_threshold)`.
            data: DataFrame produced by DataEngine.calculate_zscore (must contain 'Z-Score').
            exit_threshold: Exit threshold forwarded to backtester.

        Returns:
            pd.DataFrame: Two columns: 'Threshold' and 'Final Profit'
        """

        # Build threshold list 1.0, 1.2, ..., 3.0
        thresholds = list(np.round(np.arange(1.0, 3.0 + 1e-8, 0.2), 2))

        results: List[dict] = []

        for t in thresholds:
            # Instantiate backtester if a class was provided
            if isinstance(backtester_class, type):
                bt = backtester_class()
            else:
                bt = backtester_class

            # Run backtest; allow errors to bubble up
            backtest_df = bt.run_backtest(data.copy(), entry_threshold=float(t), exit_threshold=float(exit_threshold))

            # Extract final cumulative PnL
            if 'Cumulative PnL' in backtest_df.columns and len(backtest_df) > 0:
                final_profit = float(backtest_df['Cumulative PnL'].iloc[-1])
            else:
                # If field missing, attempt to derive from Daily PnL
                if 'Daily PnL' in backtest_df.columns:
                    final_profit = float(backtest_df['Daily PnL'].cumsum().iloc[-1])
                else:
                    final_profit = 0.0

            results.append({'Threshold': t, 'Final Profit': final_profit})

        df = pd.DataFrame(results)
        return df


if __name__ == '__main__':
    # Simple demo that runs the full pipeline (DataEngine -> Backtester -> Optimizer)
    try:
        from pypairs.data_engine import DataEngine
        from pypairs.backtester import Backtester

        engine = DataEngine(verbose=False)
        demo_tickers = ['F', 'GM', 'AAPL', 'MSFT']
        print('Downloading data and computing Z-Score for demo tickers...')
        result_df, pair_info = engine.run_full_pipeline(demo_tickers, period='1y')

        print('Running optimizer over thresholds 1.0 -> 3.0...')
        opt = StrategyOptimizer()
        grid = opt.optimize_thresholds(Backtester, result_df)
        print('\nOptimizer results:')
        print(grid.to_string(index=False))

    except Exception as exc:
        print('Demo failed:', exc)
        raise
