"""
Dedicated Unit & Edge Case Tests for Quantitative Backtesting Engine (QuantLab Pro).
Verifies return calculations, transaction cost deduction, turnover, drawdown, Sharpe/Sortino edge cases, and zero volatility.
"""

import numpy as np
import pandas as pd

from quantlab.evaluation.backtest import run_research_backtest


def test_backtest_return_and_transaction_cost_deduction():
    # 5-day test dataset with 1% daily return
    dates = [f"2024-01-0{i}" for i in range(1, 6)]
    df_aligned = pd.DataFrame({
        "Date": dates,
        "Close": [100.0, 101.0, 102.01, 103.03, 104.06],
        "future_return": [0.01, 0.01, 0.01, 0.01, 0.01]
    })

    # Always Long prediction (1 position change into long on bar 0)
    preds = np.array([1, 1, 1, 1, 1])
    res_10bps = run_research_backtest(df_aligned, preds, dates, transaction_cost_bps=10.0)

    # 1 trade into position (10 bps = 0.0010 cost on day 0)
    assert res_10bps.num_trades == 1
    assert res_10bps.transaction_cost_bps == 10.0
    # Net return on day 0: 0.01 - 0.0010 = 0.0090
    assert np.isclose(res_10bps.strategy_curve[0], 1.0090, atol=1e-4)


def test_backtest_position_turnover_and_costs():
    dates = [f"2024-01-0{i}" for i in range(1, 6)]
    df_aligned = pd.DataFrame({
        "Date": dates,
        "Close": [100.0, 101.0, 100.0, 101.0, 100.0],
        "future_return": [0.01, -0.01, 0.01, -0.01, 0.01]
    })

    # Alternating Long/Cash predictions: 1, 0, 1, 0, 1
    preds = np.array([1, 0, 1, 0, 1])
    res = run_research_backtest(df_aligned, preds, dates, transaction_cost_bps=10.0)

    # Position switches: 0->1, 1->0, 0->1, 1->0, 0->1 => 5 trades
    assert res.num_trades == 5


def test_backtest_drawdown_calculation():
    dates = [f"2024-01-0{i}" for i in range(1, 6)]
    df_aligned = pd.DataFrame({
        "Date": dates,
        "Close": [100.0, 110.0, 99.0, 90.0, 95.0],
        "future_return": [0.10, -0.10, -0.0909, 0.0555, 0.0]
    })

    preds = np.array([1, 1, 1, 1, 1])
    res = run_research_backtest(df_aligned, preds, dates, transaction_cost_bps=0.0)

    # Peak return is 1.10 on bar 0, then drops to ~0.90 => drawdown ~ -0.18
    assert res.max_drawdown < 0.0


def test_backtest_zero_volatility_and_empty_edge_cases():
    dates = ["2024-01-01", "2024-01-02"]
    df_aligned = pd.DataFrame({
        "Date": dates,
        "Close": [100.0, 100.0],
        "future_return": [0.0, 0.0]
    })

    preds = np.array([1, 1])
    res = run_research_backtest(df_aligned, preds, dates, transaction_cost_bps=0.0)

    assert res.sharpe_ratio == 0.0
    assert res.sortino_ratio == 0.0
    assert res.max_drawdown == 0.0



def test_backtest_empty_input_handling():
    res = run_research_backtest(pd.DataFrame(), np.array([]), [], transaction_cost_bps=10.0)
    assert res.total_strategy_return == 0.0
    assert res.num_trades == 0
