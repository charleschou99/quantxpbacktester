from quantxpbacktester.data.Alpaca import AlpacaDataClient
from quantxpbacktester.signals.ValueCaptureStrategy import *
from quantxpbacktester.backtester.backtester import *

# Fetch data
data_client = AlpacaDataClient()
risk_params = {
    'target_vol': 0.15,  # 15% annualized volatility target
    'max_position': 0.1,  # 10% of portfolio max
    'stop_loss': '2_atr'
}

# Fetch required data
historical_data = data_client.fetch_bars(
    symbol='AAPL',
    timeframe='1Hour',
    start_date='2020-01-01',
    end_date='2023-12-31',
    adjustment='all'
)

corporate_actions = data_client.fetch_corporate_actions(
    symbol='AAPL',
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Initialize components
data_handler = DataHandler(historical_data, corporate_actions)
strategy = QuantValueCaptureStrategy(risk_params)
engine = BacktestEngine(data_handler, strategy)

# Run backtest
engine.run()

# Generate report
metrics = AdvancedMetrics(engine.portfolio)
print("Backtest Results:")
print(pd.Series(metrics.full_report()))
