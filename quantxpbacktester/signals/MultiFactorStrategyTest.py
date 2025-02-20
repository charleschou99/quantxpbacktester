import pandas as pd
from quantxpbacktester.data.Alpaca import AlpacaDataClient
from quantxpbacktester.signals.MultiFactorStrategy import generate_signals_for_symbol, create_orders
from quantxpbacktester.backtester.backtester import *

def main():
    # Parameters for data fetch
    symbol = "AAPL"
    period = 30
    timeframe = "1D"
    start_date = "2020-01-01"
    end_date = "2024-12-31"

    print(f"Fetching Alpaca data for {symbol} from {start_date} to {end_date}...")
    df_signals = generate_signals_for_symbol(
        symbol=symbol,
        period=period,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        extended_hours=False
    )

    df_signals = df_signals.dropna()
    df_signals = create_orders(df_signals, use_kelly=True, p_win=0.6, base_risk=0.05)

    print("Data fetched and signal generated. Head of DataFrame:")
    print(df_signals.head())

    # Initialize a simple risk manager
    risk_mgr = RiskManager(max_notional=1e6, max_position_percentage=0.2)

    # Instantiate our institutional-grade backtester
    backtester = Backtester(
        data=df_signals,
        frequency=timeframe,
        symbol=symbol,
        initial_capital=1e5,
        risk_manager=risk_mgr
    )

    print("Running backtest...")
    backtester.run_backtest()

    # Compute advanced performance metrics
    metrics = backtester.risk_metrics
    print("\n=== Performance Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Placeholder calls to show future expansions (not yet implemented):
    fig = backtester.plot_equity_curve()
    fig.show()
    # backtester.run_sensitivity_analysis()
    # backtester.export_results()
    return backtester

if __name__ == "__main__":
    import plotly.io as pio
    pio.renderers.default = "browser"
    backtester = main()
