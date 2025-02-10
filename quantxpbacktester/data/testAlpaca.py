
from quantxpbacktester.data.Alpaca import AlpacaDataClient


client = AlpacaDataClient()

# Fetch minute bars with caching
minute_bars = client.fetch_bars(
    symbol='AAPL',
    timeframe='1Min',
    start_date='2023-01-01',
    end_date='2023-01-31',
    cache_dir='./data_cache',
    extended_hours=False
)

# Get corporate actions
corporate_actions = client.fetch_corporate_actions(
    symbol='AAPL',
    start_date='2020-01-01',
    end_date='2023-12-31',
    ca_types=['dividend', 'split']
)

print(f"Fetched {len(minute_bars)} bars with microstructure features:")
print(minute_bars[['open', 'high', 'low', 'close', 'volume', 'spread']].tail())

print("\nCorporate Actions:")
print(corporate_actions.tail())
