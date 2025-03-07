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
        total_portfolio_value = 0
        for sym, (qty, avg_px) in current_positions.items():
            total_portfolio_value += qty * avg_px

        if proposed_trade.side == 'BUY':
            proposed_notional = proposed_trade.quantity * current_price
        else:
            proposed_notional = -proposed_trade.quantity * current_price  # negative notion for sells

        # Check total notional
        if abs(total_portfolio_value + proposed_notional) > self.max_notional:
            return False

        # Check position-based constraints
        current_qty, current_avg_px = current_positions.get(proposed_trade.symbol, (0, current_price))
        if proposed_trade.side == 'BUY':
            new_qty = current_qty + proposed_trade.quantity
        else:
            new_qty = current_qty - proposed_trade.quantity

        new_position_notional = new_qty * current_price
        if abs(new_position_notional) > self.max_notional * self.max_position_percentage:
            return False

        return True


class Backtester:
    """
    A higher-grade backtester that:
    1. Takes a DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'signal', 'order']
    2. Uses 'order' to determine position sizes (continuous or discrete).
    3. Fills orders at vwap for each bar.
    4. Tracks PnL, risk metrics, and advanced performance metrics, including:
       - Bias per trade
         * average edge per trade (expressed as (# buys - # sells) / total trades).
         * indicates whether each trade, on average, is adding positive alpha or not.
       - Turnover
         * ratio of total absolute notional traded to average portfolio value.
         * indicates how frequently the assets are bought/sold (useful for cost & style considerations).
       - Calmar ratio
         * cagr / abs(max_drawdown).
       - Volatility
         * annualized standard deviation of returns.
    5. Also outputs a daily PnL curve in addition to the overall equity curve.
    """

    def __init__(self, data: pd.DataFrame, frequency: str, symbol: str,
                 initial_capital: float = 1e6, risk_manager: RiskManager = None):
        self.data = data.copy()
        self.symbol = symbol
        self.frequency = frequency
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.risk_manager = risk_manager if risk_manager else RiskManager()

        # Current positions: dict of {symbol: (quantity, average_cost)}
        self.positions = {symbol: (0.0, 0.0)}

        # Track trades & performance
        self.file_name = ""
        self.order_history = []
        self.equity_curve = []
        self.pnl_curve = []   # daily PnL
        self.risk_metrics = {}
        self.sharpepa = {}
        self.metricspa = {}

        # Ensure datetime index
        if not pd.api.types.is_datetime64_any_dtype(self.data.index):
            if 'timestamp' in self.data.columns:
                self.data.set_index('timestamp', inplace=True)
            else:
                raise ValueError("Data must have a 'timestamp' column or a DateTimeIndex.")
        self.data.sort_index(inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)

    def run_backtest(self):
        """
        Main loop over bars. For each bar, see if 'order' differs from current position and fill accordingly.
        Also track daily PnL and final metrics.
        """
        prev_equity = self.cash
        for timestamp, row in self.data.iterrows():
            qty = row['order']
            vwap_price = row['vwap']

            current_qty, avg_px = self.positions[self.symbol]
            trade_qty = qty - current_qty

            if abs(trade_qty) > 0.0:
                side = 'BUY' if trade_qty > 0 else 'SELL'
                trade_order = Order(self.symbol, side, abs(trade_qty), timestamp)

                if self.risk_manager.check_risk(self.positions, trade_order, vwap_price):
                    trade_order.fill_order(vwap_price)
                    self.order_history.append(trade_order)
                    self._update_positions(trade_order)
                else:
                    pass  # risk-blocked trade

            total_portfolio_value = self._calculate_portfolio_value(vwap_price)
            self.equity_curve.append((timestamp, total_portfolio_value))

            daily_pnl = total_portfolio_value - prev_equity
            self.pnl_curve.append((timestamp, daily_pnl))
            prev_equity = total_portfolio_value

        self.equity_curve = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity']).set_index('timestamp')
        self.pnl_curve = pd.DataFrame(self.pnl_curve, columns=['timestamp', 'pnl']).set_index('timestamp')

        self.risk_metrics = self.compute_performance_metrics()
        self.save_results_to_json()

    def save_results_to_json(self):
        """
        Save backtest results (equity curve, order history, daily PnL, risk metrics) as JSON.
        """
        if not os.path.exists(STOREPATH):
            os.makedirs(STOREPATH)

        eq_df = self.equity_curve.reset_index(names='Date', drop=False)
        data_df = self.data.reset_index(names='Date', drop=False)
        pnl_df = self.pnl_curve.reset_index(names='Date', drop=False)

        results = {
            'data': data_df.to_json(orient='records', date_format='iso'),
            'equity_curve': eq_df.to_json(orient='records', date_format='iso'),
            'pnl_curve': pnl_df.to_json(orient='records', date_format='iso'),
            'order_history': [o.to_dict() for o in self.order_history],
            'risk_metrics': self.risk_metrics,
            'sharpe_per_year': self.sharpepa,
            'metrics_per_year':self.metricspa
        }

        file_name = f"{self.symbol}_{self.frequency}_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.json"
        self.file_name = file_name
        file_path = os.path.join(STOREPATH, file_name)
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)

        # print(f"Backtest and risk metrics saved to {file_path}")

    def _update_positions(self, order: Order):
        current_qty, current_avg_px = self.positions[order.symbol]
        fill_cost = order.fill_price * order.quantity

        if order.side == 'BUY':
            total_cost = current_qty * current_avg_px + fill_cost
            new_qty = current_qty + order.quantity
            new_avg_px = total_cost / new_qty if new_qty != 0 else 0
            self.positions[order.symbol] = (new_qty, new_avg_px)
            self.cash -= fill_cost
        else:  # SELL
            new_qty = current_qty - order.quantity
            self.cash += fill_cost
            if new_qty == 0:
                self.positions[order.symbol] = (0.0, 0.0)
            else:
                self.positions[order.symbol] = (new_qty, current_avg_px)

    def _calculate_portfolio_value(self, price: float) -> float:
        qty, avg_px = self.positions[self.symbol]
        position_value = qty * price
        return self.cash + position_value


    def _get_annual_factor(self):
        freq = self.frequency
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
            annual_factor = 252
        return annual_factor

    def compute_performance_metrics(self) -> dict:
        """
        Summarizes performance with both standard hedge-fund style metrics and additional:
         - Bias per Trade: (# of buys - # of sells) / total trades
         - Turnover: sum of absolute notional traded / average equity
         - Calmar ratio: cagr / abs(max_drawdown)
         - Volatility: annualized standard deviation of returns

        'Bias per trade' tells us if there's a net directional bias or advantage
        from each trade (are trades systematically profitable/long/short?).
        'Turnover' indicates how frequently we buy/sell relative to our average capital.
        """

        eq_series = self.equity_curve['equity']
        rets = eq_series.pct_change().fillna(0)

        annual_factor = self._get_annual_factor()

        start_val = eq_series.iloc[0]
        end_val = eq_series.iloc[-1]
        n_years = (eq_series.index[-1] - eq_series.index[0]).days / 365
        if n_years <= 0:
            cagr = 0
        else:
            cagr = (end_val / start_val) ** (1 / n_years) - 1

        mean_ret = rets.mean() * annual_factor
        std_ret = rets.std() * np.sqrt(annual_factor)
        sharpe = mean_ret / std_ret if std_ret != 0 else 0

        downside_rets = rets[rets < 0]
        std_downside = downside_rets.std() * np.sqrt(annual_factor)
        sortino = mean_ret / std_downside if std_downside != 0 else 0

        rolling_max = eq_series.cummax()
        dd = (eq_series - rolling_max) / rolling_max
        max_dd = dd.min()

        # Volatility = annualized std of returns
        vol = std_ret

        # Calmar = cagr / abs(max_dd)
        calmar = 0
        if max_dd < 0:
            calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else 0

        # Turnover = sum(|trade notional|) / average equity
        total_abs_notional = 0
        eq_mean = eq_series.mean()
        for o in self.order_history:
            notional = abs(o.fill_price * o.quantity)
            total_abs_notional += notional
        turnover = total_abs_notional / eq_mean if eq_mean != 0 else 0

        # Bias per Trade = (# buys - # sells) / total trades
        # This is a measure of average direction or advantage from each trade.
        n_buys = sum(1 for o in self.order_history if o.side == 'BUY')
        n_sells = sum(1 for o in self.order_history if o.side == 'SELL')
        total_trades = len(self.order_history)
        bias_per_trade = (n_buys - n_sells) / total_trades if total_trades > 0 else 0
        self.sharpepa = self._compute_sharpe_per_year(eq_series)
        self.metricspa = self._compute_yearly_metrics(eq_series)

        metrics = {
            'CAGR': cagr,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'MaxDrawdown': max_dd,
            'AnnualMeanReturn': mean_ret,
            'Volatility': vol,
            'Calmar': calmar,
            'Turnover': turnover,
            'BiasPerTrade': bias_per_trade,
        }

        return metrics

    def _compute_sharpe_per_year(self, equity_series: pd.Series) -> dict:
        """
        For each calendar year in the equity series, compute the Sharpe ratio, assuming daily data.

        Returns a dictionary: {year: sharpe_for_that_year}
        """
        if len(equity_series) < 2:
            return {}

        # We'll assume daily for simplicity, i.e. annual_factor=252
        annual_factor = self._get_annual_factor()

        # Group by equity_series.index.year
        sharpe_dict = {}
        grouped = equity_series.groupby(equity_series.index.year)
        for year, subseries in grouped:
            if len(subseries) < 2:
                sharpe_dict[year] = 0.0
                continue

            rets = subseries.pct_change().dropna()
            # If the year starts or ends in the middle, we'll still compute
            # average daily returns * 252 / stdev * sqrt(252)
            mean_ret = rets.mean() * annual_factor
            std_ret = rets.std() * np.sqrt(annual_factor)
            yearly_sharpe = mean_ret / std_ret if std_ret != 0 else 0

            sharpe_dict[year] = yearly_sharpe

        return sharpe_dict

    def _compute_yearly_metrics(self, equity_series: pd.Series) -> dict:
        """
        For each calendar year in equity_series, compute multiple metrics:
          - Return (approx. partial-year CAGR)
          - Sharpe
          - Sortino
          - Max Drawdown
          - Calmar (Return / abs(MaxDD))
          - Volatility (annualized standard deviation)
          - Turnover (sum of abs notional traded / average equity in that year)
          - Bias Per Trade ((# buys - # sells) / total trades) in that year

        Returns a dict of the form:
          {
            2020: {
              "Return": float,
              "Sharpe": float,
              "Sortino": float,
              "MaxDrawdown": float,
              "Calmar": float,
              "Volatility": float,
              "Turnover": float,
              "BiasPerTrade": float
            },
            2021: { ... },
            ...
          }
        """

        if len(equity_series) < 2:
            return {}

        annual_factor = self._get_annual_factor()
        yearly_metrics = {}

        # Group the equity curve by year, e.g. {2020: Series(...), 2021: Series(...)}
        groups = equity_series.groupby(equity_series.index.year)

        for year, sub_eq in groups:
            if len(sub_eq) < 2:
                # If there's not enough data, store zeros
                yearly_metrics[year] = {
                    "Return": 0.0,
                    "Sharpe": 0.0,
                    "Sortino": 0.0,
                    "MaxDrawdown": 0.0,
                    "Calmar": 0.0,
                    "Volatility": 0.0,
                    "Turnover": 0.0,
                    "BiasPerTrade": 0.0
                }
                continue

            # --- 1) Basic returns, Sharpe, Sortino, Volatility ---
            rets = sub_eq.pct_change().dropna()
            mean_ret = rets.mean() * annual_factor
            std_ret = rets.std() * np.sqrt(annual_factor)

            yearly_sharpe = mean_ret / std_ret if std_ret != 0 else 0

            downside_rets = rets[rets < 0]
            std_downside = downside_rets.std() * np.sqrt(annual_factor)
            yearly_sortino = mean_ret / std_downside if std_downside != 0 else 0

            # Volatility is the annualized std of returns
            yearly_vol = std_ret

            # --- 2) Partial-Year Return (mini-CAGR for that sub-year) ---
            start_val = sub_eq.iloc[0]
            end_val = sub_eq.iloc[-1]
            days_count = (sub_eq.index[-1] - sub_eq.index[0]).days
            if start_val != 0 and days_count > 0:
                # approximate annualization for partial-year
                ret_for_that_year = (end_val / start_val) ** (365 / days_count) - 1
            else:
                ret_for_that_year = 0.0

            # --- 3) Max Drawdown for that year ---
            rolling_max = sub_eq.cummax()
            dd_series = (sub_eq - rolling_max) / rolling_max  # typically ≤ 0
            yearly_max_dd = dd_series.min()  # e.g. -0.15

            # Calmar = (annualized return) / abs(MDD)
            # For partial-year, we use ret_for_that_year. If MDD is near 0 or 0, handle carefully.
            if yearly_max_dd < 0:
                yearly_calmar = ret_for_that_year / abs(yearly_max_dd) if abs(yearly_max_dd) > 1e-9 else 0
            else:
                yearly_calmar = 0

            # --- 4) Turnover & Bias per Trade for that year ---
            # Filter self.order_history to only trades whose timestamp is in 'year'
            # We assume 'o.timestamp' is a Python datetime or can be parsed.
            orders_for_year = []
            for o in self.order_history:
                # Make sure 'timestamp' is a datetime object
                # If it's a string, convert with pd.to_datetime
                ts = o.timestamp if isinstance(o.timestamp, pd.Timestamp) else pd.to_datetime(o.timestamp)
                if ts.year == year:
                    orders_for_year.append(o)

            # Sum absolute notional for that year
            total_abs_notional = 0.0
            for o in orders_for_year:
                notional = abs(o.fill_price * o.quantity)
                total_abs_notional += notional

            mean_equity_for_year = sub_eq.mean()  # average equity for that sub-year
            yearly_turnover = total_abs_notional / mean_equity_for_year if mean_equity_for_year != 0 else 0

            # Bias Per Trade = (# buys - # sells) / total_trades in that year
            n_buys = sum(1 for o in orders_for_year if o.side == 'BUY')
            n_sells = sum(1 for o in orders_for_year if o.side == 'SELL')
            total_trades_for_year = len(orders_for_year)
            if total_trades_for_year > 0:
                yearly_bias_per_trade = (n_buys - n_sells) / total_trades_for_year
            else:
                yearly_bias_per_trade = 0.0

            # --- 5) Store them in the dictionary for this year ---
            yearly_metrics[year] = {
                "Return": ret_for_that_year,
                "Sharpe": yearly_sharpe,
                "Sortino": yearly_sortino,
                "MaxDrawdown": yearly_max_dd,
                "Calmar": yearly_calmar,
                "Volatility": yearly_vol,
                "Turnover": yearly_turnover,
                "BiasPerTrade": yearly_bias_per_trade
            }

        return yearly_metrics

    def plot_equity_curve(self):
        """
        Example function using Plotly for an interactive equity chart.
        """
        import plotly.graph_objects as go

        if self.equity_curve.empty:
            print("No equity curve data. Run the backtest first.")
            return

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
        return fig

    def run_sensitivity_analysis(self):
        # Placeholder
        pass

    def export_results(self):
        # Placeholder
        pass
