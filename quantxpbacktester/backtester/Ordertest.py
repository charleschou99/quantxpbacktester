import pytest
from datetime import datetime, timedelta
from Order import *


def test_order_execution_flow():
    """End-to-end test of institutional order execution"""
    # Generate synthetic market data with realistic microstructure
    test_symbol = "TEST"
    start_time = datetime(2023, 1, 1, 9, 30)
    ticks = pd.date_range(start=start_time, periods=1800, freq='s')  # 30 minutes

    lob_data = pd.DataFrame({
        'timestamp': ticks,
        'bid': np.random.normal(100, 0.1, 1800),
        'ask': np.random.normal(100.02, 0.1, 1800),
        'bid_size': np.random.randint(100, 1000, 1800),
        'ask_size': np.random.randint(100, 1000, 1800),
        'volume': np.random.poisson(500, 1800),
    }).set_index('timestamp')

    # Create test orders with different execution styles
    twap_order = Order(
        asset=test_symbol,
        quantity=10000,
        order_type=OrderType.TWAP,
        timestamp=start_time,
        duration='30m',
        tick_data=lob_data
    )

    vwap_order = Order(
        asset=test_symbol,
        quantity=5000,
        order_type=OrderType.VWAP,
        timestamp=start_time,
        duration='30m',
        tick_data=lob_data
    )

    # Simulate order execution
    execution_log = []
    current_time = start_time

    while current_time < start_time + timedelta(minutes=30):
        # Get current market state
        try:
            lob_snapshot = lob_data.loc[current_time]
        except KeyError:
            current_time += timedelta(seconds=1)
            continue

        # Process TWAP order
        if not twap_order.is_complete:
            twap_qty, twap_price = twap_order.execute(current_time, lob_snapshot)
            if twap_qty != 0:
                execution_log.append({
                    'timestamp': current_time,
                    'type': 'TWAP',
                    'qty': twap_qty,
                    'price': twap_price
                })


        # Process VWAP order
        if not vwap_order.is_complete:
            vwap_qty, vwap_price = vwap_order.execute(current_time, lob_snapshot)
            if vwap_qty != 0:
                execution_log.append({
                    'timestamp': current_time,
                    'type': 'VWAP',
                    'qty': vwap_qty,
                    'price': vwap_price
                })

        current_time += timedelta(seconds=1)

    # Convert log to DataFrame
    executions = pd.DataFrame(execution_log)

    # --- Assertions ---
    # TWAP Validation
    twap_exec = executions[executions['type'] == 'TWAP']
    assert len(twap_exec) > 0, "TWAP order didn't execute"
    assert abs(twap_exec['qty'].sum()) == 10000, "TWAP didn't fill completely"

    # Check execution duration
    duration = twap_exec['timestamp'].max() - twap_exec['timestamp'].min()
    assert 1750 <= duration.total_seconds() <= 1850, "TWAP duration mismatch"

    # VWAP Validation
    vwap_exec = executions[executions['type'] == 'VWAP']
    assert len(vwap_exec) > 0, "VWAP order didn't execute"
    assert abs(vwap_exec['qty'].sum()) == 5000, "VWAP didn't fill completely"

    # Check VWAP price quality
    market_vwap = (lob_data['volume'] * lob_data['ask']).sum() / lob_data['volume'].sum()
    our_vwap = (vwap_exec['qty'].abs() * vwap_exec['price']).sum() / vwap_exec['qty'].abs().sum()
    assert abs(market_vwap - our_vwap) < 0.01, "VWAP implementation poor"

    # Slippage Analysis
    analyzer = TWAPVWAPAnalyzer([twap_order, vwap_order])
    slippage = analyzer.calculate_slippage()

    assert slippage['mean_bps'] < 2.0, "Excessive TWAP slippage"
    assert slippage['worst_bps'] < 5.0, "Unacceptable worst-case slippage"

    # Partial Fill Validation
    partial_order = Order(
        asset=test_symbol,
        quantity=1000000,  # Larger than available liquidity
        order_type=OrderType.MARKET,
        timestamp=start_time
    )
    qty, price = partial_order.execute(start_time, lob_data.iloc[0])
    assert abs(qty) < 1000000, "Failed partial fill logic"

    # Market Impact Check
    initial_price = lob_data.iloc[0]['ask']
    impacted_price = partial_order.executions[0]['price']
    assert impacted_price > initial_price, "Market impact not modeled"

    # After Hours Protection
    after_hours_time = datetime(2023, 1, 1, 16, 0)
    ah_lob = lob_data.iloc[0].copy()
    ah_lob.name = after_hours_time
    qty, price = twap_order.execute(after_hours_time, ah_lob)
    assert qty == 0, "Traded outside market hours"


def test_order_metadata():
    """Test order state management"""
    test_order = Order(
        asset="META",
        quantity=500,
        order_type=OrderType.LIMIT,
        timestamp=datetime(2023, 1, 1, 9, 30),
        limit_price=150.00
    )

    # Test limit order execution
    lob_snapshot = pd.Series({
        'bid': 149.99,
        'ask': 150.01,
        'bid_size': 1000,
        'ask_size': 1000
    })

    # Shouldn't execute above limit
    qty, price = test_order.execute(
        datetime(2023, 1, 1, 9, 31),
        lob_snapshot
    )
    assert qty == 0, "Limit order executed outside price"

    # Should execute at limit
    lob_snapshot['ask'] = 149.99
    qty, price = test_order.execute(
        datetime(2023, 1, 1, 9, 32),
        lob_snapshot
    )
    assert qty == 500 and price == 150.00, "Limit order failed"


def test_vwap_volume_adaptation():
    """Test VWAP's volume following capability"""
    # Create predictable volume pattern
    test_volume = [1000] * 900 + [5000] * 900  # First 15m low volume, second 15m high
    lob_data = pd.DataFrame({
        'timestamp': pd.date_range(start="2023-01-01 09:30", periods=1800, freq='s'),
        'volume': test_volume,
        'ask': 100.00,
        'bid': 99.98,
        'ask_size': 1000,
        'bid_size': 1000
    }).set_index('timestamp')

    vwap_order = Order(
        asset="TEST",
        quantity=10000,
        order_type=OrderType.VWAP,
        timestamp=datetime(2023, 1, 1, 9, 30),
        duration='30m',
        tick_data=lob_data
    )

    # Simulate execution
    current_time = datetime(2023, 1, 1, 9, 30)
    fills = []

    while current_time <= datetime(2023, 1, 1, 10, 0):
        try:
            snapshot = lob_data.loc[current_time]
            qty, price = vwap_order.execute(current_time, snapshot)
            if qty != 0:
                fills.append({'time': current_time, 'qty': qty})
        except KeyError:
            pass
        current_time += timedelta(seconds=1)

    # Check volume following
    first_half = [f for f in fills if f['time'] < datetime(2023, 1, 1, 9, 45)]
    second_half = [f for f in fills if f['time'] >= datetime(2023, 1, 1, 9, 45)]

    assert len(second_half) > 2 * len(first_half), "VWAP didn't adapt to volume"
    assert abs(sum(f['qty'] for f in fills)) == 10000, "Incomplete VWAP fill"


def test_twap_time_slicing():
    """Verify TWAP's uniform time distribution"""
    order = Order(
        asset="TWAP_TEST",
        quantity=3600,  # 1 share/sec for 1 hour
        order_type=OrderType.TWAP,
        timestamp=datetime(2023, 1, 1, 9, 30),
        duration='1h'
    )

    # Simulate perfect market conditions
    perfect_lob = pd.Series({
        'ask': 100.00,
        'bid': 100.00,
        'ask_size': 10000,
        'bid_size': 10000
    })

    fills = []
    current_time = datetime(2023, 1, 1, 9, 30)

    while current_time <= datetime(2023, 1, 1, 10, 30):
        qty, price = order.execute(current_time, perfect_lob)
        if qty != 0:
            fills.append(current_time)
        current_time += timedelta(seconds=1)

    # Check fill distribution
    time_diffs = np.diff([t.timestamp() for t in fills])
    assert np.allclose(time_diffs, 1.0, atol=0.1), "TWAP not time-uniform"


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
