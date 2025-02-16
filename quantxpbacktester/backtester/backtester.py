import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict, Optional, List
import datetime

class DataHandler:
    """Handles pre-adjusted data from Alpaca with corporate action metadata"""

    def __init__(self, ohlcv: pd.DataFrame, corporate_actions: pd.DataFrame):
        self.data = self._validate_data(ohlcv)
        self.corporate_actions = self._process_corporate_actions(corporate_actions)

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return data.dropna()

    def _process_corporate_actions(self, ca: pd.DataFrame) -> pd.DataFrame:
        """Align corporate actions with price data index"""
        return ca.reindex(self.data.index).ffill().bfill()


class Portfolio:
    """Tracks positions and value using pre-adjusted data"""

    def __init__(self, initial_capital: float):
        self.capital = initial_capital
        self.positions: Dict[str, float] = {}
        self.history = pd.DataFrame(columns=['total'])

    def update(self, timestamp: datetime, trades: Dict[str, float], prices: pd.Series):
        """Update portfolio with new trades"""
        for asset, delta in trades.items():
            self.positions[asset] = self.positions.get(asset, 0) + delta
            self.capital -= delta * prices[asset]

        # Record portfolio value
        position_value = sum(qty * prices.get(asset, 0)
                             for asset, qty in self.positions.items())
        self.history.loc[timestamp] = self.capital + position_value


class BacktestEngine:
    """Main backtesting execution"""

    def __init__(self, data_handler: DataHandler, strategy):
        self.data = data_handler.data
        self.strategy = strategy
        self.portfolio = Portfolio(1_000_000)
        self.ca = data_handler.corporate_actions

    def run(self):
        """Core backtest loop"""
        for idx, row in self.data.iterrows():
            # Get historical data up to current time
            history = self.data.loc[:idx]

            # Generate signals (using corporate actions data)
            signals = self.strategy.generate_signals(
                historical_data=history,
                corporate_actions=self.ca.loc[:idx]
            )

            # Execute trades at current prices
            self.portfolio.update(
                timestamp=idx,
                trades=signals,
                prices=row[['open', 'high', 'low', 'close']].mean()  # VWAP approximation
            )


class AdvancedMetrics:
    """Performance analytics with type-safe calculations"""

    def __init__(self, portfolio: Portfolio):
        self.returns = self._calculate_returns(portfolio)

    def _calculate_returns(self, portfolio: Portfolio) -> pd.Series:
        """Calculate daily returns from portfolio history"""
        return portfolio.history['total'].pct_change().dropna()

    def full_report(self) -> Dict[str, float]:
        """Generate complete performance report"""
        return {
            'sharpe': self._annualized_sharpe(),
            'sortino': self._sortino_ratio(),
            'max_dd': self._max_drawdown(),
            'skewness': float(skew(self.returns)),
            'kurtosis': float(kurtosis(self.returns))
        }

    def _annualized_sharpe(self) -> float:
        excess_returns = self.returns - 0.02 / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def _sortino_ratio(self) -> float:
        downside_returns = self.returns[self.returns < 0]
        return self.returns.mean() / downside_returns.std() * np.sqrt(252)

    def _max_drawdown(self) -> float:
        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        return (cumulative / peak - 1).min()
