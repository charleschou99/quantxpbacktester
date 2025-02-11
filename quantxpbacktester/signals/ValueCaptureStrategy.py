import pandas as pd
import numpy as np
from scipy.stats import zscore


class QuantValueCaptureStrategy:
    """
    Institutional Multi-Factor Strategy combining:
    1. Volume-Weighted Momentum (Technical)
    2. Dividend Seasonality (Fundamental)
    3. Multi-Timeframe Convergence
    4. Volatility Targeting
    """

    def __init__(self, risk_params: dict):
        self.risk_params = risk_params
        self.dividend_lookahead = 5  # Days ahead to consider dividends

    def generate_signals(self, historical_data: pd.DataFrame,
                         corporate_actions: pd.DataFrame) -> dict[str, float]:
        """
        Generate signals using integrated analysis
        Returns: Dictionary of {asset: target_position}
        """
        # Feature Engineering
        df = self._calculate_features(historical_data)
        ca_df = self._process_corporate_actions(corporate_actions)

        merged = df.join(ca_df, how='left')

        # Signal Components
        signals = self._technical_signals(merged)
        fundamental = self._fundamental_signals(merged)
        risk_adjusted = self._risk_management(merged, signals)

        return risk_adjusted

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create multi-timeframe features"""
        # Volume-Weighted Price Trends
        data['vwap_20'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
        data['vwap_50'] = (data['close'] * data['volume']).rolling(50).sum() / data['volume'].rolling(50).sum()

        # Volatility Measures
        data['atr_14'] = self._average_true_range(data, 14)
        data['volatility_30'] = data['close'].pct_change().rolling(30).std()

        # Volume Dynamics
        data['volume_z'] = zscore(data['volume'], nan_policy='omit')
        data['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(50).mean()

        # Multi-Timeframe Momentum
        data['hourly_macd'] = self._macd(data['close'], 12, 26, 9)
        data['daily_macd'] = self._macd(data['close'].resample('D').last().ffill(), 12, 26, 9)

        return data.dropna()

    def _process_corporate_actions(self, ca_data: pd.DataFrame) -> pd.DataFrame:
        """Enrich data with corporate action events"""
        ca_features = pd.DataFrame(index=ca_data.index)

        # Dividend proximity features
        ca_features['dividend_pending'] = ca_data['type'].eq('dividend').shift(-self.dividend_lookahead).fillna(0)
        ca_features['dividend_yield'] = ca_data['value'] / ca_data['close']

        # Split adjustment factors
        ca_features['split_ratio'] = ca_data['ratio'].fillna(1.0)

        return ca_features

    def _technical_signals(self, data: pd.DataFrame) -> pd.Series:
        """Technical trading signals"""
        # Volume-Weighted Trend
        vwap_signal = np.where(data['vwap_20'] > data['vwap_50'], 1, -1)

        # MACD Convergence
        macd_signal = np.where((data['hourly_macd'] > 0) & (data['daily_macd'] > 0), 1, 0)

        # Breakout Detection
        volatility_adjusted = data['close'] / data['atr_14']
        breakout = zscore(volatility_adjusted) > 1.5

        return 0.4 * vwap_signal + 0.3 * macd_signal + 0.3 * breakout

    def _fundamental_signals(self, data: pd.DataFrame) -> pd.Series:
        """Fundamental factors"""
        # Dividend Capture Signal
        dividend_signal = data['dividend_pending'] * (1 + data['dividend_yield'])

        # Valuation Signal (P/E from Alpaca)
        pe_signal = np.where(data['pe_ratio'] < 15, 1,
                             np.where(data['pe_ratio'] > 25, -1, 0))

        return 0.6 * dividend_signal + 0.4 * pe_signal

    def _risk_management(self, data: pd.Series, raw_signal: pd.Series) -> dict[str, float]:
        """Volatility-targeted position sizing"""
        # Target volatility scaling
        position_size = self.risk_params['target_vol'] / data['volatility_30']
        position_size = position_size.clip(upper=self.risk_params['max_position'])

        # Stop-loss levels
        stop_loss = data['close'] - 2 * data['atr_14']

        # Final signal composition
        final_signal = raw_signal * position_size
        final_signal = np.where(data['close'] < stop_loss, 0, final_signal)

        return {'equity': final_signal.iloc[-1]}  # Single-asset example

    # Technical Indicators --------------------------------
    def _average_true_range(self, data: pd.DataFrame, window: int) -> pd.Series:
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    def _macd(self, series: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line
