import requests
import pandas as pd
from datetime import datetime, timedelta, time
import pytz
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

API_KEY_ID = os.environ.get("ALPACA_KEY")
API_SECRET_KEY = os.environ.get("ALPACA_SECRET")


class AlpacaDataClient:
    """Institutional-grade Alpaca data client with enhanced features"""

    def __init__(self):
        self.session = self._create_session()
        self.eastern = pytz.timezone('America/New_York')
        self.utc = pytz.utc

    def _create_session(self) -> requests.Session:
        """Create resilient session with retries and timeout"""
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(['GET'])
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        adjustment: str = 'all',
        feed: str = 'sip',
        extended_hours: bool = False,
        cache_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Enhanced market data fetcher with:
        - Intelligent pagination
        - Data validation
        - Caching layer
        - Microstructure features
        """
        if cache_dir:
            cache_path = self._get_cache_path(symbol, timeframe, start_date, end_date, cache_dir)
            if os.path.exists(cache_path):
                return pd.read_parquet(cache_path)

        url = 'https://data.alpaca.markets/v2/stocks/bars'
        headers = {
            'APCA-API-KEY-ID': API_KEY_ID,
            'APCA-API-SECRET-KEY': API_SECRET_KEY
        }

        params = {
            'symbols': symbol,
            'timeframe': timeframe,
            'start': self._parse_date(start_date),
            'end': self._parse_date(end_date),
            'limit': 10000,
            'adjustment': adjustment,
            'feed': feed
        }

        data = []
        page_token = None

        while True:
            if page_token:
                params['page_token'] = page_token

            try:
                response = self.session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=10
                )
                response.raise_for_status()

                json_data = response.json()
                page_token = json_data.get('next_page_token')

                df_page = self._process_bars(
                    json_data['bars'][symbol],
                    timeframe,
                    extended_hours
                )
                data.append(df_page)

                if not page_token:
                    break

            except requests.exceptions.HTTPError as e:
                if response.status_code == 422:
                    print(f"Invalid parameters: {response.text}")
                break
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                break

        if data:
            final_df = pd.concat(data).sort_index()
            if cache_dir:
                final_df.to_parquet(cache_path)
            return final_df
        return pd.DataFrame()

    def fetch_multiple_symbols(self, symbols: List[str], **kwargs):
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.fetch_bars, sym, **kwargs)
                       for sym in symbols]
            return {f.result().index[0].symbol: f.result() for f in futures}

    def _process_bars(self, bars: List[Dict], timeframe: str, extended_hours: bool) -> pd.DataFrame:
        """Process raw bars with microstructure features"""
        processed = []

        for bar in bars:
            utc_time = datetime.fromisoformat(bar['t'].rstrip('Z')).replace(tzinfo=self.utc)
            eastern_time = utc_time.astimezone(self.eastern)

            if not extended_hours and self._is_intraday(timeframe):
                if not self._is_market_hours(eastern_time):
                    continue

            processed.append({
                'timestamp': eastern_time,
                'open': bar['o'],
                'high': bar['h'],
                'low': bar['l'],
                'close': bar['c'],
                'volume': bar['v'],
                'trade_count': bar.get('n', 0),
                'vwap': bar.get('vw', bar['c']),
                'spread': self._calculate_spread(bar)
            })

        df = pd.DataFrame(processed)
        df.set_index('timestamp', inplace=True)
        return self._clean_data(df)

    def _calculate_spread(self, bar: Dict) -> float:
        """Calculate bid-ask spread proxy"""
        if 'h' in bar and 'l' in bar:
            return bar['h'] - bar['l']
        return 0.0

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data validation and cleaning"""
        # Remove zero-volume bars
        df = df[df['volume'] > 0]

        # Validate price movements
        df = df[(df['high'] >= df['low']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open'])]

        # Forward fill missing values
        df = df.ffill().bfill()
        return df

    def fetch_corporate_actions(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        ca_types: List[str] = ['dividend', 'split']
    ) -> pd.DataFrame:
        """Enhanced corporate actions fetcher with multiple types"""
        url = "https://paper-api.alpaca.markets/v2/corporate_actions/announcements"
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": API_KEY_ID,
            "APCA-API-SECRET-KEY": API_SECRET_KEY
        }

        results = []
        current_start = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        while current_start <= end_date:
            current_end = min(current_start + timedelta(days=89), end_date)

            params = {
                'since': current_start.strftime("%Y-%m-%d"),
                'until': current_end.strftime("%Y-%m-%d"),
                'symbol': symbol,
                'ca_types': ','.join(ca_types)
            }

            try:
                response = self.session.get(url, headers=headers, params=params)
                response.raise_for_status()
                results.extend(response.json())
            except requests.exceptions.HTTPError as e:
                print(f"Failed to fetch CA data: {response.text}")

            current_start = current_end + timedelta(days=1)

        return self._process_corporate_actions(results)

    def _process_corporate_actions(self, actions: List[Dict]) -> pd.DataFrame:
        """Process corporate actions into tradable format"""
        processed = []

        for action in actions:
            if action['ca_type'] == 'dividend':
                processed.append({
                    'date': datetime.strptime(action['ex_date'], "%Y-%m-%d"),
                    'type': 'dividend',
                    'value': action['cash'],
                    'ratio': 1.0
                })
            elif action['ca_type'] == 'split':
                processed.append({
                    'date': datetime.strptime(action['ex_date'], "%Y-%m-%d"),
                    'type': 'split',
                    'value': action['from_factor'] / action['to_factor'],
                    'ratio': action['from_factor'] / action['to_factor']
                })

        return pd.DataFrame(processed).set_index('date')

    # Helper methods
    def _parse_date(self, date_str: str) -> str:
        return datetime.strptime(date_str, "%Y-%m-%d").isoformat() + 'Z'

    def _is_market_hours(self, dt: datetime) -> bool:
        return time(9, 30) <= dt.time() <= time(15, 59)

    def _is_intraday(self, timeframe: str) -> bool:
        return any(c in timeframe for c in ['Min', 'Hour'])

    def _get_cache_path(self, symbol: str, timeframe: str, start: str, end: str, cache_dir: str) -> str:
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(
            cache_dir,
            f"{symbol}_{timeframe}_{start}_{end}.parquet"
        )

