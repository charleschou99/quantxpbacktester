import numpy as np
import pandas as pd
import datetime as dt
import os
import json

STOREPATH = os.path.join(r'C:\Users\charl\BacktestData', 'MultiFactor')

class Order:
    """
    Institutional-style Order class
    - For now, we assume all fills occur at VWAP for the bar in which the order is submitted.
    - We track the side (buy/sell), quantity, symbol, and fill details.
    """

    def __init__(self, symbol: str, side: str, quantity: float, timestamp: pd.Timestamp):
        """
        :param symbol: The symbol being traded (e.g., 'AAPL').
        :param side: 'BUY' or 'SELL'.
        :param quantity: The quantity of shares to trade.
        :param timestamp: The bar timestamp at which the order is placed.
        """
        self.symbol = symbol
        self.side = side.upper()
        self.quantity = quantity
        self.timestamp = timestamp
        self.fill_price = None
        self.is_filled = False

    def fill_order(self, fill_price: float):
        """
        Fill the order at the provided fill price (in this backtester, we use VWAP).
        """
        self.fill_price = fill_price
        self.is_filled = True

    def to_dict(self):
        """
        Convert the order object to a dictionary for serialization, logging, etc.
        """
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "fill_price": self.fill_price,
            "is_filled": self.is_filled
        }


class RiskManager:
    """
    Basic Risk Manager that can be expanded to include advanced checks, e.g.:
    - Sector exposures
    - Factor exposures
    - Dollar VaR / Beta constraints
    - etc.
    """

    def __init__(self, max_notional: float = 1e7, max_position_percentage: float = 0.1):
        """
        :param max_notional: Maximum total notional portfolio value allowed.
        :param max_position_percentage: Maximum fraction of total portfolio per position.
        """
        self.max_notional = max_notional
        self.max_position_percentage = max_position_percentage

    def check_risk(self, current_positions: dict, proposed_trade: Order, current_price: float) -> bool:
        """
        Example risk check to ensure we are not exceeding max notional or position-based constraints.
        :param current_positions: A dict of {symbol: (quantity, average_cost)} describing the current holdings.
        :param proposed_trade: The Order object representing the trade we want to execute.
        :param current_price: The current price (or vwap) for that symbol.
        :return: True if the trade passes risk checks, False otherwise.
        """
        # 1) Check total notional
        #    Evaluate total portfolio notional + proposed trade notional
        total_portfolio_value = 0
        for sym, (qty, avg_px) in current_positions.items():
            total_portfolio_value += qty * avg_px

        if proposed_trade.side == 'BUY':
            proposed_notional = proposed_trade.quantity * current_price
        else:
            proposed_notional = -proposed_trade.quantity * current_price  # negative notion for sells

        if abs(total_portfolio_value + proposed_notional) > self.max_notional:
            return False

        # 2) Check position-based constraints
        #    Evaluate the position for that symbol
        current_qty, _ = current_positions.get(proposed_trade.symbol, (0, current_price))
        new_qty = current_qty + proposed_trade.quantity if proposed_trade.side == 'BUY' else current_qty - proposed_trade.quantity
        new_position_notional = new_qty * current_price

        if abs(new_position_notional) > self.max_notional * self.max_position_percentage:
            return False

        return True


class Backtester:
    """
    A higher-grade backtester that:
    1. Takes a DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'signal', ...]
    2. Uses 'signal' to determine position sizes (continuous or discrete).
    3. Fills orders at vwap for each bar.
    4. Tracks PnL, risk metrics, and advanced performance metrics.
    """

    def __init__(self, data: pd.DataFrame, frequency:str, symbol: str, initial_capital: float = 1e6,
                 risk_manager: RiskManager = None):
        """
        :param data: DataFrame for a single symbol containing the necessary columns.
        :param symbol: The symbol being tested (e.g., 'AAPL').
        :param initial_capital: Starting capital in USD.
        :param risk_manager: A RiskManager instance (can be None to skip risk checks).
        """
        self.data = data.copy()
        self.symbol = symbol
        self.frequency = frequency
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.risk_manager = risk_manager if risk_manager else RiskManager()

        # Current positions: dict of {symbol: (quantity, average_cost)}
        # In a multi-symbol scenario, you’d replicate or expand logic for each symbol.
        self.positions = {symbol: (0.0, 0.0)}

        # For tracking trades & performance
        self.file_name = ""
        self.order_history = []
        self.equity_curve = []
        self.risk_metrics = {}

        # Ensure timestamp is a pandas DateTimeIndex
        if not pd.api.types.is_datetime64_any_dtype(self.data.index):
            if 'timestamp' in self.data.columns:
                self.data.set_index('timestamp', inplace=True)
            else:
                raise ValueError("Data must have a 'timestamp' column or a DateTimeIndex.")

        # Sort by index just in case
        self.data.sort_index(inplace=True)

        # Fill missing values if needed
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)

    def run_backtest(self):
        """
        Core loop that runs the backtest bar by bar.
        For each bar, determines the order quantity from the signal and attempts to execute that order.
        """
        for timestamp, row in self.data.iterrows():
            # 1) Determine the desired position from the signal
            #    For example, a signal of 0.3 => 30% of max capital in that symbol
            #    Or interpret signal in your own logic
            qty = row['order']
            vwap_price = row['vwap']  # fill price

            # Desired notional or size
            # E.g. if signal ∈ [-1, 1], interpret it as fraction of total capital
            desired_notional = vwap_price * qty
            current_qty, avg_px = self.positions[self.symbol]
            current_notional = current_qty * vwap_price

            # Position difference
            trade_notional_diff = desired_notional - current_notional
            side = 'BUY' if trade_notional_diff > 0 else 'SELL'

            # Convert to absolute quantity (round for realism)
            # quantity_diff = abs(trade_notional_diff) / vwap_price if vwap_price != 0 else 0
            # quantity_diff = np.floor(quantity_diff)

            # 2) Create an Order if quantity_diff is non-trivial
            # if quantity_diff > 0.0:
            if np.abs(qty) > 0.0:
                order = Order(self.symbol, side, qty, timestamp)

                # 3) Risk check
                if self.risk_manager.check_risk(self.positions, order, vwap_price):
                    # 4) Execute the order
                    order.fill_order(vwap_price)
                    self.order_history.append(order)

                    # 5) Update cash and positions
                    self._update_positions(order)
                else:
                    # Risk manager blocked the trade
                    pass

            # 6) Mark portfolio value at the end of the bar
            total_portfolio_value = self._calculate_portfolio_value(vwap_price)
            self.equity_curve.append((timestamp, total_portfolio_value))

        self.equity_curve = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity']).set_index('timestamp')
        self.risk_metrics = self.compute_performance_metrics()

        # 7) Save results
        self.save_results_to_json()

    def save_results_to_json(self):
        """
        Save backtest results (equity curve, order history, and risk metrics) as JSON.
        """
        # 1) Ensure the folder exists
        if not os.path.exists(STOREPATH):
            os.makedirs(STOREPATH)

        # 2) Prepare the results dictionary
        #    - equity_curve is converted to JSON (string) via .to_json()
        #    - order_history is converted to a list of dicts via each Order's to_dict()
        #    - risk_metrics can be stored as-is if it's already a serializable dict
        results = {
            'equity_curve': self.equity_curve.to_json(orient='records', date_format='iso'),
            'order_history': [o.to_dict() for o in self.order_history],
            'risk_metrics': self.risk_metrics
        }

        # 3) Build the file name with a time stamp, plus JSON suffix
        #    e.g. "AAPL_1Day_20231005_1530.json"
        file_name = (
            f"{self.symbol}_{self.frequency}_"
            f"{dt.datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )
        self.file_name = file_name  # store this if you need reference

        # 4) Write the JSON file
        file_path = os.path.join(STOREPATH, file_name)
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Backtest and risk metrics successfully saved to {file_path}")

    def _update_positions(self, order: Order):
        """
        Updates self.positions and self.cash after a fill.
        """
        current_qty, current_avg_px = self.positions[order.symbol]
        fill_cost = order.fill_price * order.quantity

        if order.side == 'BUY':
            # Weighted average cost for new position
            total_cost = current_qty * current_avg_px + fill_cost
            new_qty = current_qty + order.quantity
            new_avg_px = total_cost / new_qty if new_qty != 0 else 0
            self.positions[order.symbol] = (new_qty, new_avg_px)
            self.cash -= fill_cost
        else:  # SELL
            # Realize PnL portion
            new_qty = current_qty - order.quantity
            realized_pnl = (order.fill_price - current_avg_px) * order.quantity
            self.cash += fill_cost
            if new_qty == 0:
                # Flatten the position
                self.positions[order.symbol] = (0.0, 0.0)
            else:
                self.positions[order.symbol] = (new_qty, current_avg_px)

    def _calculate_portfolio_value(self, price: float) -> float:
        """
        Calculates the total value of the portfolio = sum of position + cash.
        """
        qty, avg_px = self.positions[self.symbol]
        position_value = qty * price
        return self.cash + position_value

    def compute_performance_metrics(self) -> dict:
        """
        Calculates hedge-fund style performance metrics:
        - CAGR
        - Sharpe Ratio
        - Sortino Ratio
        - Maximum Drawdown
        :param freq: 'D' for daily, 'M' for monthly, etc.
        :return: dict containing performance metrics
        """
        # 1) Create returns series
        eq_series = self.equity_curve['equity']
        rets = eq_series.pct_change().fillna(0)

        freq = self.frequency
        # 2) Annualization factor
        if freq == '1D':
            annual_factor = 252
        elif freq == '1W':
            annual_factor = 52
        elif freq == '1M':
            annual_factor = 12
        elif freq == '1Min':
            annual_factor = 98280
        elif freq == '15Min':
            annual_factor = 6552
        elif freq == '30Min':
            annual_factor = 3276
        elif freq == '1H':
            annual_factor = 1638
        else:
            annual_factor = 252  # default daily

        # 3) CAGR
        #    compound annual growth rate from start to end
        start_val = eq_series.iloc[0]
        end_val = eq_series.iloc[-1]
        n_years = (eq_series.index[-1] - eq_series.index[0]).days / 365
        cagr = (end_val / start_val) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 4) Sharpe Ratio
        mean_ret = rets.mean() * annual_factor
        std_ret = rets.std() * np.sqrt(annual_factor)
        sharpe = mean_ret / std_ret if std_ret != 0 else 0

        # 5) Sortino Ratio
        downside_rets = rets[rets < 0]
        std_downside = downside_rets.std() * np.sqrt(annual_factor)
        sortino = mean_ret / std_downside if std_downside != 0 else 0

        # 6) Max Drawdown
        rolling_max = eq_series.cummax()
        drawdown = (eq_series - rolling_max) / rolling_max
        max_dd = drawdown.min()

        return {
            'CAGR': cagr,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'MaxDrawdown': max_dd,
            'AnnualMeanReturn': mean_ret,  # simpler annualized return
        }

    # ========== PLACEHOLDER: future improvements / expansions ==========

    def plot_equity_curve(self):
        """
        Plot the equity curve using Plotly.
        This displays an interactive chart of portfolio equity over time.
        """
        import plotly.graph_objects as go


        if self.equity_curve is None or self.equity_curve.empty:
            print("No equity curve data to plot. Please run the backtest first.")
            return

        # Build the figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.equity_curve.index,
            y=self.equity_curve['equity'],
            mode='lines',
            name='Equity'
        ))

        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Equity (USD)',
            hovermode='x unified'
        )

        # Display the interactive chart in a browser or notebook
        return fig

    def run_sensitivity_analysis(self):
        """
        Placeholder function for advanced scenario analysis,
        e.g. parameter sweeps, stress tests, etc.
        """
        pass

    def export_results(self):
        """
        Placeholder function to export trades, performance metrics, etc. to CSV or DB.
        """
        pass
