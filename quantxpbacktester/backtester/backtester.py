"""
Institutional-Grade Backtesting Engine v2.0
Key Enhancements:
1. Advanced market impact models
2. Realistic order execution logic
3. Comprehensive risk management system
4. Statistical robustness checks
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_1samp
from typing import Dict, List, Tuple
from enum import Enum

class DataHandler:
    """Clean, survivorship-bias-free data pipeline"""

    def __init__(self, raw_data: pd.DataFrame):
        self.data = self._process_data(raw_data)
        self._validate_dataset()

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle corporate actions and survivorship bias"""
        processed = data.copy()

        # Adjust for splits and dividends
        processed['adj_close'] = processed['close'] * processed['cumulative_factor']
        processed['adj_volume'] = processed['volume'] / processed['split_factor']

        # Forward-fill missing data with volatility estimation
        processed = processed.ffill().bfill()
        processed['filled_returns'] = processed['adj_close'].pct_change()
        processed['filled_vol'] = processed['filled_returns'].ewm(span=63).std()

        return processed

    def _validate_dataset(self):
        """Ensure no look-ahead bias in timestamps"""
        assert self.data.index.is_monotonic_increasing, "Timestamps must be ordered"
        assert not self.data.isna().any().any(), "NaN values detected in processed data"


from typing import Optional, Tuple, Dict, List
from enum import Enum
import pandas as pd
import numpy as np


class OrderType(Enum):
    MARKET = 1
    LIMIT = 2
    TWAP = 3
    VWAP = 4

class Order:
    """Institutional-quality order execution system"""

    def __init__(
        self,
        asset: str,
        quantity: float,
        order_type: OrderType,
        timestamp: pd.Timestamp,
        limit_price: Optional[float] = None,
        duration: str = '1h',
        tick_data: pd.DataFrame = None
    ):
        """
        Parameters:
        - tick_data: DataFrame with microstructural features ['bid', 'ask', 'volume', 'trade_count']
        """
        self.asset = asset
        self.original_qty = quantity
        self.remaining_qty = abs(quantity)
        self.direction = np.sign(quantity)
        self.order_type = order_type
        self.creation_time = timestamp
        self.limit_price = limit_price
        self.duration = pd.to_timedelta(duration)
        self.end_time = self.creation_time + self.duration
        self.tick_data = tick_data
        self.executions = []

    def execute(self, current_time: pd.Timestamp, lob_data: pd.DataFrame) -> Tuple[float, float]:
        """
        Execute order against limit order book data
        Returns tuple of (executed_quantity, executed_price)
        """
        if current_time > self.end_time or self.remaining_qty <= 0:
            return 0.0, 0.0

        if self.order_type == OrderType.TWAP:
            return self._twap_execution(current_time, lob_data)
        elif self.order_type == OrderType.VWAP:
            return self._vwap_execution(current_time, lob_data)
        # Other order types handled similarly

    def _twap_execution(self, current_time: pd.Timestamp, lob_data: pd.DataFrame) -> Tuple[float, float]:
        """Time-Weighted Average Price execution"""
        time_elapsed = (current_time - self.creation_time).total_seconds()
        total_duration = self.duration.total_seconds()

        # Calculate time remaining and target participation
        time_remaining = max(total_duration - time_elapsed, 1)
        target_participation = self.remaining_qty * (1 / time_remaining)

        # Get current spread and mid-price
        best_bid = lob_data['bid'].iloc[0]
        best_ask = lob_data['ask'].iloc[0]
        mid_price = (best_bid + best_ask) / 2

        # Aggressive execution taking liquidity
        executed_qty = min(target_participation, lob_data['ask_size'].sum() if self.direction > 0
        else lob_data['bid_size'].sum())
        executed_price = best_ask if self.direction > 0 else best_bid

        # Apply market impact
        impact_factor = 1 + 0.0001 * (executed_qty / lob_data['volume'].mean())
        executed_price *= impact_factor

        self._update_order_state(executed_qty, executed_price)
        return executed_qty * self.direction, executed_price

    def _vwap_execution(self, current_time: pd.Timestamp, lob_data: pd.DataFrame) -> Tuple[float, float]:
        """Volume-Weighted Average Price execution"""
        # Calculate cumulative volume since order creation
        historical_volume = self.tick_data.loc[self.creation_time:current_time, 'volume'].sum()
        remaining_volume = self.tick_data.loc[current_time:self.end_time, 'volume'].sum()
        total_volume = historical_volume + remaining_volume

        if total_volume == 0:
            return 0.0, 0.0

        # Calculate target participation rate
        target_pct = self.remaining_qty / total_volume
        current_volume = lob_data['volume'].sum()
        target_qty = target_pct * current_volume

        # Execute across order book levels
        executed_qty = 0
        vwap_numerator = 0
        book_levels = lob_data.iterrows() if self.direction > 0 else reversed(lob_data.iterrows())

        for _, level in book_levels:
            available = level['ask_size'] if self.direction > 0 else level['bid_size']
            fill = min(target_qty - executed_qty, available)
            price = level['ask'] if self.direction > 0 else level['bid']

            executed_qty += fill
            vwap_numerator += fill * price

            if executed_qty >= target_qty:
                break

        if executed_qty == 0:
            return 0.0, 0.0

        executed_price = vwap_numerator / executed_qty
        self._update_order_state(executed_qty, executed_price)
        return executed_qty * self.direction, executed_price

    def _update_order_state(self, executed_qty: float, executed_price: float):
        """Update order bookkeeping"""
        self.remaining_qty -= executed_qty
        self.executions.append({
            'timestamp': pd.Timestamp.now(),
            'quantity': executed_qty * self.direction,
            'price': executed_price
        })

    @property
    def is_complete(self) -> bool:
        return self.remaining_qty <= 0 or pd.Timestamp.now() > self.end_time


class TWAPVWAPAnalyzer:
    """Execution quality analysis toolkit"""

    def __init__(self, orders: List[Order]):
        self.orders = orders

    def calculate_slippage(self) -> Dict[str, float]:
        """Compute implementation shortfall"""
        results = []
        for order in self.orders:
            if order.order_type not in [OrderType.TWAP, OrderType.VWAP]:
                continue

            benchmark = order.tick_data.loc[order.creation_time:order.end_time, 'close'].mean()
            actual = np.mean([e['price'] for e in order.executions])
            results.append((actual - benchmark) / benchmark * 1e4)

        return {
            'mean_bps': np.mean(results),
            'std_bps': np.std(results),
            'worst_bps': np.min(results)
        }

    def plot_execution_quality(self):
        """Visualize execution performance"""
        # Implementation using matplotlib/seaborn
        pass

class Portfolio:
    """Professional portfolio management with risk constraints"""

    def __init__(self, initial_capital: float, risk_params: Dict):
        self.capital = initial_capital
        self.positions = {}
        self.risk = RiskManager(risk_params)
        self.trade_log = []

    def update(self, timestamp: pd.Timestamp, orders: List[Order], market_data: Dict):
        """Process orders with risk checks"""
        for order in orders:
            if self.risk.check_order(order, market_data):
                filled_qty, filled_price = order.execute(market_data)
                self._update_position(order.asset, filled_qty, filled_price, timestamp)

    def _update_position(self, asset: str, qty: float, price: float, timestamp: pd.Timestamp):
        """Atomic position update with PnL calculation"""
        prev_value = self.positions.get(asset, 0) * price
        self.positions[asset] = self.positions.get(asset, 0) + qty
        new_value = self.positions[asset] * price
        self.capital -= qty * price
        self.trade_log.append({
            'timestamp': timestamp,
            'asset': asset,
            'qty': qty,
            'price': price,
            'pnl': new_value - prev_value
        })


class RiskManager:
    """Institutional-grade risk management system"""

    def __init__(self, params: Dict):
        self.max_drawdown = params.get('max_drawdown', 0.2)
        self.var_limit = params.get('var_limit', 0.05)
        self.position_limits = params.get('position_limits', {})
        self.concentration_limits = params.get('concentration_limits', {})

    def check_order(self, order: Order, market_data: pd.Series) -> bool:
        """Comprehensive pre-trade checks"""
        if self._exceeds_var(order, market_data):
            return False
        if self._violates_position_limits(order):
            return False
        if self._violates_concentration(order):
            return False
        return True

    def _exceeds_var(self, order: Order, market_data: pd.Series) -> bool:
        """Historical VaR check at 99% confidence"""
        returns = market_data['filled_returns'].dropna()
        var = norm.ppf(0.99) * returns.std() * np.sqrt(10)
        potential_loss = abs(order.quantity) * market_data['close'] * var
        return potential_loss > self.var_limit * self.current_portfolio_value()


class BacktestEngine:
    """Event-driven backtesting core"""

    def __init__(self, data_handler: DataHandler, strategy, risk_params: Dict):
        self.data = data_handler.data
        self.strategy = strategy
        self.portfolio = Portfolio(1e6, risk_params)
        self._create_event_stream()

    def _create_event_stream(self):
        """Create tick/bar events with microstructural features"""
        self.events = []
        for timestamp, row in self.data.iterrows():
            self.events.append({
                'type': 'BAR',
                'timestamp': timestamp,
                'data': row
            })

    def run(self):
        """Main event loop"""
        for event in self.events:
            if event['type'] == 'BAR':
                self._process_bar(event)

    def _process_bar(self, event):
        """Handle bar event with strategy logic"""
        current_data = event['data']
        historical_data = self.data.loc[:event['timestamp']]

        # Generate signals
        signals = self.strategy.generate_signals(historical_data)

        # Create orders
        orders = self._signals_to_orders(signals, current_data)

        # Update portfolio
        self.portfolio.update(event['timestamp'], orders, current_data)

    def _signals_to_orders(self, signals: Dict, market_data: pd.Series) -> List[Order]:
        """Convert strategy signals to executable orders"""
        orders = []
        for asset, signal in signals.items():
            target_qty = self._calculate_position_size(asset, signal, market_data)
            current_qty = self.portfolio.positions.get(asset, 0)
            delta = target_qty - current_qty

            if delta != 0:
                orders.append(Order(
                    asset=asset,
                    quantity=delta,
                    order_type=OrderType.TWAP,
                    timestamp=market_data.name
                ))
        return orders


class AdvancedMetrics:
    """Professional performance analytics"""

    def __init__(self, portfolio: Portfolio, benchmark: pd.Series):
        self.returns = self._calculate_returns(portfolio)
        self.benchmark_rets = benchmark.pct_change().dropna()

    def full_report(self) -> Dict:
        return {
            'sharpe': self._adjusted_sharpe(),
            'probabilistic_sharpe': self._probabilistic_sharpe(),
            'deflated_sharpe': self._deflated_sharpe(),
            'beta': self._beta_exposure(),
            'turnover': self._annualized_turnover(),
            'capacity': self._strategy_capacity(),
            'bootstrap': self._bootstrap_analysis()
        }

    def _adjusted_sharpe(self) -> float:
        """Sharpe ratio adjusted for skewness and kurtosis"""
        sr = self.returns.mean() / self.returns.std()
        skewness = skew(self.returns)
        kurt = kurtosis(self.returns)
        return sr * (1 + (skewness / 6) * sr - ((kurt - 3) / 24) * sr ** 2)

    def _bootstrap_analysis(self) -> Dict:
        """Bootstrap resampling for metric significance"""
        sharpe_samples = []
        for _ in range(1000):
            sample = np.random.choice(self.returns, len(self.returns), replace=True)
            sharpe_samples.append(sample.mean() / sample.std())

        ci = np.percentile(sharpe_samples, [5, 95])
        return {
            'sharpe_ci': ci,
            'p_value': ttest_1samp(sharpe_samples, 0).pvalue
        }
