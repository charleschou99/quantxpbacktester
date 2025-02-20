"""
GPT enhanced
Forward Looking Strategy -> does not work
THIS IS AN EXAMPLE
"""

import numpy as np
import pandas as pd
from quantxpbacktester.data.Alpaca import AlpacaDataClient

# Example role-play: You, as a lead quant at Millennium, are prototyping a multi-factor alpha.
# This script fetches data from Alpaca (using your attached AlpacaDataClient) and builds
# a signal from multiple factors. Then, you can feed the resulting DataFrame to the
# QuantXPBacktester for simulation.

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes the Average True Range over a given period.
    ATR is a popular measure of volatility in position sizing.
    """
    # True Range (TR) is max of:
    # 1) high - low
    # 2) abs(high - previous close)
    # 3) abs(low - previous close)
    # We'll assume df has columns 'high', 'low', 'close'.
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()  # Simple moving average of TR
    return atr.fillna(method='ffill')

def compute_kelly_fraction(p_win: float = 0.55, ratio_win_loss: float = 1.0, fraction_of_kelly: float = 1.0) -> float:
    """
    Computes a (possibly scaled) Kelly fraction.
    :param p_win: Probability of winning (e.g. 0.55 means 55%).
    :param ratio_win_loss: Average win / average loss.
    :param fraction_of_kelly: Scale Kelly fraction down (0.5 => half Kelly).
    :return: Kelly fraction in [0, 1+] (theoretically can exceed 1).
    """
    kelly_raw = p_win - (1 - p_win) / ratio_win_loss
    kelly_scaled = kelly_raw * fraction_of_kelly
    # If Kelly is negative, we typically set it to 0 or do not trade.
    return max(kelly_scaled, 0)


def calculate_momentum(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Computes a simple momentum factor: (close price today) / (close price N days ago) - 1.
    """
    return df['close'].pct_change(periods=period).fillna(0)


def calculate_volatility(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Computes rolling volatility (standard deviation) of returns over a given period.
    Normalized or inverted for a factor sense (lower vol => higher factor).
    """
    returns = df['close'].pct_change().fillna(0)
    rolling_std = returns.rolling(period).std().fillna(0)
    # Invert so that lower volatility yields a higher 'factor' score
    return 1 / (rolling_std + 1e-9)


def calculate_volume_trend(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Computes a simple ratio of current volume to average volume over a period.
    """
    rolling_vol = df['volume'].rolling(period).mean().fillna(0)
    return df['volume'] / (rolling_vol + 1e-9)


def build_multifactor_signal(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Combines multiple factors into a single signal. You can weight these factors
    or transform them as needed.

    Signal > 0 => bullish, Signal < 0 => bearish, or any continuous measure in between.
    """
    # Calculate each factor
    momentum_factor = calculate_momentum(df, period=period)
    vol_factor = calculate_volatility(df, period=period)
    volume_factor = calculate_volume_trend(df, period=period)

    # Example weighting scheme
    # Typically you'd calibrate these weights with historical data & your alpha research
    weight_momentum = 0.40
    weight_vol = 0.30
    weight_volume = 0.30

    # Combine them
    combined_signal = (weight_momentum * momentum_factor
                       + weight_vol * vol_factor
                       + weight_volume * volume_factor).iloc[period:]

    # Optionally, you might standardize or rescale the signal
    combined_signal = (combined_signal - combined_signal.rolling(100).mean()) / (combined_signal.rolling(100).std() + 1e-9)
    combined_signal = combined_signal.dropna()

    return combined_signal

def create_orders(
    df: pd.DataFrame,
    signal_col: str = 'signal',
    capital: float = 1e5,
    atr_period: int = 14,
    base_risk: float = 0.02,
    use_kelly: bool = False,
    p_win: float = 0.55,
    ratio_win_loss: float = 1.0,
    fraction_of_kelly: float = 1.0,
    upper_threshold: float = 0.5,
    lower_threshold: float = -0.5
) -> pd.DataFrame:
    """
    Translates a *continuous* signal (df[signal_col]) into discrete positions (-1, 0, +1),
    then computes how many shares to buy or sell using ATR-based volatility scaling
    (and optionally Kelly sizing).

    Steps:
    1) Convert continuous signal to discrete direction:
         if signal > upper_threshold => +1 (long),
         if signal < lower_threshold => -1 (short),
         else => 0 (flat).
    2) Compute ATR and define fraction_of_capital = base_risk / ATR (capped at 1).
    3) Multiply fraction_of_capital by Kelly fraction (if use_kelly=True).
    4) Multiply by the discrete direction.
    5) Convert fraction_of_capital to #shares: shares = fraction_final * capital / close_price.
    6) Round to integer => df['order'].

    :param signal_col: Name of the column in df that contains the *continuous* signal.
    :param capital: Total capital to allocate for this symbol in USD.
    :param atr_period: Period for ATR calculation.
    :param base_risk: Base scaling factor for volatility sizing (e.g., 0.02 => risk 2% of capital at typical ATR).
    :param use_kelly: Whether to incorporate Kelly fraction.
    :param p_win: Probability of winning for Kelly.
    :param ratio_win_loss: Average gain/loss ratio for Kelly.
    :param fraction_of_kelly: Scale Kelly fraction (e.g., 0.5 => half Kelly).
    :param upper_threshold: Positive threshold above which we go long.
    :param lower_threshold: Negative threshold below which we go short.
    """
    required_cols = [signal_col, 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in DataFrame.")

    # -------------------------
    # 1) Discretize the Signal
    # -------------------------
    # Example thresholds: if signal > 0.5 => 1, if signal < -0.5 => -1, else 0
    continuous_signal = df[signal_col].fillna(0)
    discrete_signal = pd.Series(
        np.where(continuous_signal > upper_threshold, 1,
                 np.where(continuous_signal < lower_threshold, -1, 0)),
        index=df.index
    )

    # ----------------------------------------
    # 2) Compute ATR for volatility-based sizing
    # ----------------------------------------
    df['ATR'] = compute_atr(df, period=atr_period)

    # -------------------------------------------
    # 3) Compute Kelly fraction if requested
    # -------------------------------------------
    kelly_fraction = 1.0
    if use_kelly:
        kelly_fraction = compute_kelly_fraction(
            p_win=p_win,
            ratio_win_loss=ratio_win_loss,
            fraction_of_kelly=fraction_of_kelly
        )
        # e.g., clamp Kelly fraction if it’s extremely large
        kelly_fraction = min(kelly_fraction, 2.0)  # example clamp

    # -----------------------------------------------------------------
    # 4) fraction_of_capital = min(1, base_risk / ATR) * kelly_fraction
    #    multiplied by discrete direction
    # -----------------------------------------------------------------
    fraction_series = base_risk / (df['ATR'] + 1e-9)
    fraction_series = fraction_series.clip(upper=1.0)
    fraction_series *= kelly_fraction

    # multiply by discrete direction
    fraction_final = fraction_series * discrete_signal

    # ---------------------------------------------------
    # 5) Convert fraction_of_capital -> # shares
    #    order = fraction_final * capital / vwap
    # ---------------------------------------------------
    price = df['vwap'].replace(0, np.nan)
    shares = fraction_final * (capital / price)

    # 6) Round to an integer
    df['order'] = shares.round().fillna(0).astype(int)

    return df


def generate_signals_for_symbol(
    symbol: str,
    period: int = 10,
    timeframe: str = '1Day',
    start_date: str = '2023-01-01',
    end_date: str = '2025-01-01',
    extended_hours: bool = False
) -> pd.DataFrame:
    """
    Fetches the data from Alpaca for a single symbol, builds a multi-factor signal,
    and returns a DataFrame that can be used by a backtester.
    """
    client = AlpacaDataClient()
    df = client.fetch_bars(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        extended_hours=extended_hours
    )

    # Build the multi-factor signal
    combined_signal = build_multifactor_signal(df)
    df = df.iloc[period+(len(df) - len(combined_signal)):]
    df['signal'] = combined_signal

    # Typically for a backtester, you might only need (timestamp, signal, close, etc.)
    # It depends on the backtester's ingestion format. Here we keep a simple structure:
    return df


def main():
    """
    Example main function where you might:
    1. Fetch data for multiple symbols
    2. Generate signals
    3. Save or pass to your backtester
    """
    symbol_list = ['AAPL', 'MSFT', 'AMZN']  # Sample symbols
    signals_dict = {}
    period = 10

    # For demonstration, fetch data + signals for each symbol
    for sym in symbol_list:
        df_signals = generate_signals_for_symbol(sym, period=period)
        df_signals = create_orders(df_signals)
        signals_dict[sym] = df_signals.iloc[period:]

    # At this stage, you'd pass `signals_dict` to the QuantXPBacktester or store locally
    # for further analysis. For example:
    # backtest_results = run_backtest(signals_dict)

    # Print a snippet of signals
    for sym in symbol_list:
        print(f"=== {sym} Signal Sample ===")
        print(signals_dict[sym][['signal', 'close']].tail(5))

    return signals_dict


if __name__ == "__main__":
    signal_dict = main()
