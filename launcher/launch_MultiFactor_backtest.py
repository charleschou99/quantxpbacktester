#!/usr/bin/env python3

import argparse
import pandas as pd

# Import your modules (adjust paths as needed)
from quantxpbacktester.signals.MultiFactorStrategy import generate_signals_for_symbol, create_orders
from quantxpbacktester.backtester.backtester import Backtester, RiskManager


def main():
    parser = argparse.ArgumentParser(
        description="Launch a Multi-Factor Backtest and return the result file name."
    )
    parser.add_argument("--symbol", type=str, default="AAPL", help="Ticker symbol to backtest.")
    parser.add_argument("--start_date", type=str, default="2022-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default="2022-12-31", help="End date (YYYY-MM-DD).")
    parser.add_argument("--timeframe", type=str, default="1Day", help="Bar timeframe (e.g., 1Day, 1Min).")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital in USD.")
    parser.add_argument("--freq", type=str, default="D", help="Frequency for performance metrics (D, W, M, 1Min...).")
    # parser.add_argument("--file_name", type=str, default="backtest_results.pkl",
    #                     help="File name in the data_cache folder to store results.")

    args = parser.parse_args()

    # 1) Fetch & generate signals
    df = generate_signals_for_symbol(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # 2) Create orders from the continuous signal
    df = create_orders(df, signal_col='signal')

    # 3) Instantiate a backtester
    risk_mgr = RiskManager(max_notional=1e6, max_position_percentage=0.2)

    # Build the backtester
    # We'll assume your InstitutionalBacktester supports a 'file_name' property to store results
    backtester = Backtester(
        data=df,
        symbol=args.symbol,
        frequency=args.timeframe,
        initial_capital=args.capital,
        risk_manager=risk_mgr
    )

    # 4) Run the backtest
    backtester.run_backtest()

    # If your backtester has a member variable self.file_name, set it here
    # so it knows which file to write to in data_cache:
    # backtester.file_name = args.file_name

    # 5) Print the file name (which the system can capture)
    # This is how we "return" data from a script to an external caller.
    # Make sure your backtester writes to data_cache/<file_name> in the run_backtest() method.
    # e.g., "data_cache/backtest_results.pkl"
    print(backtester.file_name)


if __name__ == "__main__":
    main()
