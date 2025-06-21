"""
Cointegration Testing Module for Trading Strategies

This module provides comprehensive cointegration testing capabilities for pairs trading
and statistical arbitrage strategies. It includes:

1. Unit root tests (ADF, KPSS) to check for integrated time series
2. Engle-Granger cointegration tests with and without constant terms
3. Johansen cointegration tests for multiple time series
4. Utility functions for pairs selection and signal generation
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings('ignore')


def test_unit_root(series: pd.Series, test_type: str = 'adf', **kwargs) -> Dict:
    """
    Test for unit root in a time series using ADF or KPSS test.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to test
    test_type : str
        'adf' for Augmented Dickey-Fuller test
        'kpss' for KPSS test
    **kwargs : 
        Additional arguments for the test functions
        
    Returns:
    --------
    Dict containing test results and p-value
    """
    if test_type.lower() == 'adf':
        # ADF test: H0 = series has unit root (non-stationary)
        # H1 = series is stationary
        result = adfuller(series.dropna(), **kwargs)
        return {
            'test_type': 'ADF',
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05,  # Reject H0 if p < 0.05
            'null_hypothesis': 'Series has unit root (non-stationary)',
            'alternative_hypothesis': 'Series is stationary'
        }
    
    elif test_type.lower() == 'kpss':
        # KPSS test: H0 = series is stationary
        # H1 = series has unit root (non-stationary)
        result = kpss(series.dropna(), **kwargs)
        return {
            'test_type': 'KPSS',
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[3],
            'is_stationary': result[1] > 0.05,  # Fail to reject H0 if p > 0.05
            'null_hypothesis': 'Series is stationary',
            'alternative_hypothesis': 'Series has unit root (non-stationary)'
        }
    
    else:
        raise ValueError("test_type must be 'adf' or 'kpss'")


def test_integration_order(df: pd.DataFrame, max_order: int = 2) -> Dict[str, Dict]:
    """
    Test the integration order of multiple time series.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns representing different time series
    max_order : int
        Maximum integration order to test
        
    Returns:
    --------
    Dict with integration order results for each series
    """
    results = {}
    
    for col in df.columns:
        series = df[col]
        integration_order = 0
        
        # Test original series
        adf_result = test_unit_root(series, 'adf')
        if adf_result['is_stationary']:
            results[col] = {
                'integration_order': 0,
                'tests': [adf_result]
            }
            continue
        
        # Test first differences
        diff1 = series.diff().dropna()
        adf_result_diff1 = test_unit_root(diff1, 'adf')
        
        if adf_result_diff1['is_stationary']:
            results[col] = {
                'integration_order': 1,
                'tests': [adf_result, adf_result_diff1]
            }
            continue
        
        # Test second differences if max_order >= 2
        if max_order >= 2:
            diff2 = diff1.diff().dropna()
            adf_result_diff2 = test_unit_root(diff2, 'adf')
            
            if adf_result_diff2['is_stationary']:
                results[col] = {
                    'integration_order': 2,
                    'tests': [adf_result, adf_result_diff1, adf_result_diff2]
                }
            else:
                results[col] = {
                    'integration_order': '>2',
                    'tests': [adf_result, adf_result_diff1, adf_result_diff2]
                }
        else:
            results[col] = {
                'integration_order': '>1',
                'tests': [adf_result, adf_result_diff1]
            }
    
    return results


def engle_granger_test(series1: pd.Series, series2: pd.Series, 
                      trend: str = 'c', maxlag: Optional[int] = None) -> Dict:
    """
    Perform Engle-Granger cointegration test.
    
    Parameters:
    -----------
    series1 : pd.Series
        First time series
    series2 : pd.Series
        Second time series
    trend : str
        'c' for constant, 'nc' for no constant, 'ct' for constant and trend
    maxlag : int, optional
        Maximum lag for ADF test on residuals
        
    Returns:
    --------
    Dict containing test results
    """
    # Align series
    aligned_data = pd.concat([series1, series2], axis=1).dropna()
    if len(aligned_data) < 30:
        return {
            'error': 'Insufficient data for reliable cointegration test',
            'p_value': np.nan,
            'test_statistic': np.nan,
            'is_cointegrated': False
        }
    
    y = aligned_data.iloc[:, 0]
    x = aligned_data.iloc[:, 1]
    
    # Run OLS regression
    if trend == 'nc':
        # No constant
        model = np.polyfit(x, y, 0)
        residuals = y - model[0] * x
    elif trend == 'ct':
        # Constant and trend
        x_with_trend = np.column_stack([x, np.arange(len(x))])
        model = np.linalg.lstsq(x_with_trend, y, rcond=None)[0]
        residuals = y - (model[0] * x + model[1] * np.arange(len(x)))
    else:
        # Constant only (default)
        x_with_const = np.column_stack([x, np.ones(len(x))])
        model = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
        residuals = y - (model[0] * x + model[1])
    
    # Test residuals for unit root
    adf_result = adfuller(residuals, regression='nc', maxlag=maxlag)
    
    return {
        'test_type': f'Engle-Granger ({trend})',
        'test_statistic': adf_result[0],
        'p_value': adf_result[1],
        'critical_values': adf_result[4],
        'is_cointegrated': adf_result[1] < 0.05,
        'cointegration_vector': model,
        'residuals': residuals,
        'trend_type': trend,
        'null_hypothesis': 'No cointegration',
        'alternative_hypothesis': 'Series are cointegrated'
    }


def johansen_test(df: pd.DataFrame, k_ar_diff: int = 1, 
                  det_order: int = -1, maxlag: Optional[int] = None) -> Dict:
    """
    Perform Johansen cointegration test for multiple time series.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns representing different time series
    k_ar_diff : int
        Number of lagged differences
    det_order : int
        -1: no deterministic terms
         0: constant term
         1: linear trend
    maxlag : int, optional
        Maximum lag for the test
        
    Returns:
    --------
    Dict containing test results
    """
    # Remove any NaN values
    clean_df = df.dropna()
    
    if len(clean_df) < 50:
        return {
            'error': 'Insufficient data for reliable Johansen test',
            'trace_statistics': [],
            'max_eigen_statistics': [],
            'p_values': [],
            'cointegration_rank': 0
        }
    
    try:
        result = coint_johansen(clean_df.values, det_order, k_ar_diff)
        
        # Extract results
        trace_stats = result.lr1
        max_eigen_stats = result.lr2
        critical_values = result.cvt
        max_eigen_critical = result.cvm
        
        # Determine cointegration rank
        coint_rank = 0
        for i, (trace_stat, crit_val) in enumerate(zip(trace_stats, critical_values[:, 1])):
            if trace_stat > crit_val:
                coint_rank = i + 1
        
        return {
            'test_type': 'Johansen',
            'trace_statistics': trace_stats,
            'max_eigen_statistics': max_eigen_stats,
            'critical_values_trace': critical_values,
            'critical_values_max_eigen': max_eigen_critical,
            'cointegration_rank': coint_rank,
            'eigenvectors': result.evec,
            'eigenvalues': result.eig,
            'det_order': det_order,
            'k_ar_diff': k_ar_diff
        }
        
    except Exception as e:
        return {
            'error': f'Johansen test failed: {str(e)}',
            'trace_statistics': [],
            'max_eigen_statistics': [],
            'p_values': [],
            'cointegration_rank': 0
        }


def comprehensive_cointegration_analysis(df: pd.DataFrame, 
                                        symbols: Optional[List[str]] = None) -> Dict:
    """
    Perform comprehensive cointegration analysis on a DataFrame of price series.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price series as columns
    symbols : List[str], optional
        List of symbol names. If None, uses DataFrame column names
        
    Returns:
    --------
    Dict containing comprehensive analysis results
    """
    if symbols is None:
        symbols = df.columns.tolist()
    
    # Ensure symbols is a list
    if not isinstance(symbols, list):
        raise ValueError("symbols must be a list of strings")
    
    results = {
        'integration_analysis': {},
        'pairwise_cointegration': {},
        'johansen_analysis': {},
        'summary': {}
    }
    
    # 1. Test integration order for each series
    print("Testing integration order...")
    results['integration_analysis'] = test_integration_order(df)
    
    # 2. Pairwise Engle-Granger tests
    print("Performing pairwise cointegration tests...")
    n_symbols = len(symbols)
    cointegrated_pairs = []
    
    for i in range(n_symbols):
        for j in range(i+1, n_symbols):
            symbol1, symbol2 = symbols[i], symbols[j]
            
            # Test with constant
            eg_const = engle_granger_test(df[symbol1], df[symbol2], trend='c')
            
            # Test without constant
            eg_nc = engle_granger_test(df[symbol1], df[symbol2], trend='nc')
            
            # Test with constant and trend
            eg_ct = engle_granger_test(df[symbol1], df[symbol2], trend='ct')
            
            pair_key = f"{symbol1}_vs_{symbol2}"
            results['pairwise_cointegration'][pair_key] = {
                'with_constant': eg_const,
                'without_constant': eg_nc,
                'with_trend': eg_ct,
                'any_cointegrated': any([eg_const['is_cointegrated'], 
                                       eg_nc['is_cointegrated'], 
                                       eg_ct['is_cointegrated']])
            }
            
            if results['pairwise_cointegration'][pair_key]['any_cointegrated']:
                cointegrated_pairs.append(pair_key)
    
    # 3. Johansen test for all series together
    print("Performing Johansen test...")
    results['johansen_analysis'] = johansen_test(df)
    
    # 4. Summary statistics
    results['summary'] = {
        'total_series': n_symbols,
        'cointegrated_pairs': cointegrated_pairs,
        'num_cointegrated_pairs': len(cointegrated_pairs),
        'cointegration_rate': len(cointegrated_pairs) / (n_symbols * (n_symbols - 1) / 2) if n_symbols > 1 else 0
    }
    
    return results


def generate_cointegration_report(analysis_results: Dict) -> str:
    """
    Generate a formatted report from cointegration analysis results.
    
    Parameters:
    -----------
    analysis_results : Dict
        Results from comprehensive_cointegration_analysis
        
    Returns:
    --------
    Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("COINTEGRATION ANALYSIS REPORT")
    report.append("=" * 60)
    
    # Integration order summary
    report.append("\n1. INTEGRATION ORDER ANALYSIS")
    report.append("-" * 30)
    for symbol, result in analysis_results['integration_analysis'].items():
        report.append(f"{symbol}: I({result['integration_order']})")
    
    # Pairwise cointegration summary
    report.append("\n2. PAIRWISE COINTEGRATION TESTS")
    report.append("-" * 35)
    for pair, tests in analysis_results['pairwise_cointegration'].items():
        report.append(f"\n{pair}:")
        for test_name, test_result in tests.items():
            if test_name != 'any_cointegrated':
                if 'error' not in test_result:
                    report.append(f"  {test_name}: p-value = {test_result['p_value']:.4f}, "
                                f"Cointegrated = {test_result['is_cointegrated']}")
                else:
                    report.append(f"  {test_name}: {test_result['error']}")
    
    # Johansen test summary
    report.append("\n3. JOHANSEN COINTEGRATION TEST")
    report.append("-" * 32)
    johansen = analysis_results['johansen_analysis']
    if 'error' not in johansen:
        report.append(f"Cointegration Rank: {johansen['cointegration_rank']}")
        report.append("Trace Statistics:")
        for i, stat in enumerate(johansen['trace_statistics']):
            report.append(f"  Rank {i}: {stat:.4f}")
    else:
        report.append(f"Error: {johansen['error']}")
    
    # Summary statistics
    report.append("\n4. SUMMARY")
    report.append("-" * 10)
    summary = analysis_results['summary']
    report.append(f"Total Series: {summary['total_series']}")
    report.append(f"Cointegrated Pairs: {summary['num_cointegrated_pairs']}")
    report.append(f"Cointegration Rate: {summary['cointegration_rate']:.2%}")
    
    if summary['cointegrated_pairs']:
        report.append("Cointegrated Pairs:")
        for pair in summary['cointegrated_pairs']:
            report.append(f"  - {pair}")
    
    return "\n".join(report)


def find_best_cointegrated_pairs(df: pd.DataFrame, 
                                min_p_value: float = 0.05,
                                symbols: Optional[List[str]] = None) -> List[Dict]:
    """
    Find the best cointegrated pairs based on p-values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price series
    min_p_value : float
        Maximum p-value threshold for cointegration
    symbols : List[str], optional
        List of symbol names
        
    Returns:
    --------
    List of dictionaries with pair information and test results
    """
    if symbols is None:
        symbols = df.columns.tolist()
    
    # Ensure symbols is a list
    if not isinstance(symbols, list):
        raise ValueError("symbols must be a list of strings")
    
    best_pairs = []
    n_symbols = len(symbols)
    
    for i in range(n_symbols):
        for j in range(i+1, n_symbols):
            symbol1, symbol2 = symbols[i], symbols[j]
            
            # Test with constant (most common)
            eg_result = engle_granger_test(df[symbol1], df[symbol2], trend='c')
            
            if eg_result['is_cointegrated'] and eg_result['p_value'] <= min_p_value:
                best_pairs.append({
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'p_value': eg_result['p_value'],
                    'test_statistic': eg_result['test_statistic'],
                    'cointegration_vector': eg_result['cointegration_vector'],
                    'residuals': eg_result['residuals']
                })
    
    # Sort by p-value (lowest first)
    best_pairs.sort(key=lambda x: x['p_value'])
    
    return best_pairs


# Example usage function
def example_cointegration_analysis():
    """
    Example function showing how to use the cointegration testing module.
    """
    # Create sample data (replace with your actual data)
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Create cointegrated series
    x = np.cumsum(np.random.randn(500)) + 100
    y = 2 * x + np.random.randn(500) * 0.1  # y = 2x + noise
    z = np.cumsum(np.random.randn(500)) + 50  # Independent series
    
    df = pd.DataFrame({
        'SPY': x,
        'QQQ': y,
        'GLD': z
    }, index=dates)
    
    # Run comprehensive analysis
    results = comprehensive_cointegration_analysis(df)
    
    # Generate report
    report = generate_cointegration_report(results)
    print(report)
    
    # Find best pairs
    best_pairs = find_best_cointegrated_pairs(df)
    print("\nBest Cointegrated Pairs:")
    for pair in best_pairs:
        print(f"{pair['symbol1']} vs {pair['symbol2']}: p-value = {pair['p_value']:.6f}")
    
    return results, best_pairs
