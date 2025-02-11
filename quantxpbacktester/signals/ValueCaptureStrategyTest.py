from quantxpbacktester.data.Alpaca import AlpacaDataClient
from quantxpbacktester.signals.ValueCaptureStrategy import *
from quantxpbacktester.backtester.backtester import *
# Initialize components
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

# Initialize strategy and backtester
strategy = QuantValueCaptureStrategy(risk_params)
engine = BacktestEngine(
    data_handler=DataHandler(historical_data),
    strategy=strategy,
    risk_params=risk_params
)

# Run backtest
engine.run()

# Analyze results
metrics = AdvancedMetrics(engine.portfolio, benchmark=historical_data['close'])
report = metrics.full_report()
