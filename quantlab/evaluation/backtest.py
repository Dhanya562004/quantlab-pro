"""
Research-Only Backtesting Engine for QuantLab Pro.
Simulates long/cash strategy with transaction costs, drawdown analysis, and risk-adjusted metrics.
DISCLAIMER: Educational research platform only. Not financial or investment advice.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class BacktestResult(BaseModel):
    """Encapsulates backtest strategy metrics and return series."""
    total_strategy_return: float = Field(description="Total cumulative strategy return (e.g., 0.15 = 15%).")
    total_benchmark_return: float = Field(description="Total cumulative buy & hold benchmark return.")
    sharpe_ratio: float = Field(description="Annualized Sharpe ratio (assuming 252 trading days).")
    sortino_ratio: float = Field(description="Annualized Sortino ratio.")
    max_drawdown: float = Field(description="Maximum Peak-to-Trough Drawdown (e.g., -0.12 = -12%).")
    win_rate: float = Field(description="Fraction of active trading periods with positive net return.")
    num_trades: int = Field(description="Total number of position transitions.")
    transaction_cost_bps: float = Field(description="Transaction cost applied per trade in basis points.")
    strategy_curve: list[float] = Field(description="Cumulative strategy return series for plotting.")
    benchmark_curve: list[float] = Field(description="Cumulative benchmark return series for plotting.")
    dates: list[str] = Field(description="Corresponding dates for return series.")


def run_research_backtest(
    df_aligned: pd.DataFrame,
    predictions: np.ndarray,
    test_dates: list[str],
    transaction_cost_bps: float = 10.0,
    allow_short: bool = False
) -> BacktestResult:
    """
    Simulate a research-only backtest on held-out test data.

    Args:
        df_aligned: Aligned DataFrame containing 'Date', 'Close', 'future_return'.
        predictions: Model binary predictions (1 = Long, 0 = Cash/Short).
        test_dates: List of ISO date strings for test split.
        transaction_cost_bps: Transaction cost per trade in basis points (10 bps = 0.0010).
        allow_short: If True, prediction 0 -> Short (-1); if False, prediction 0 -> Cash (0).

    Returns:
        BacktestResult object.
    """
    # Filter test split records
    test_mask = df_aligned["Date"].isin(test_dates)
    df_test = df_aligned[test_mask].copy().reset_index(drop=True)

    if len(df_test) == 0 or len(predictions) == 0:
        return BacktestResult(
            total_strategy_return=0.0,
            total_benchmark_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            num_trades=0,
            transaction_cost_bps=transaction_cost_bps,
            strategy_curve=[1.0],
            benchmark_curve=[1.0],
            dates=["N/A"],
        )

    # Match predictions length with df_test length
    n_eval = min(len(df_test), len(predictions))
    df_test = df_test.iloc[:n_eval]
    preds = predictions[:n_eval]

    # Asset daily returns
    asset_returns = df_test["future_return"].to_numpy()
    dates = df_test["Date"].tolist()

    # Determine positions: 1 for Long; 0 or -1 for cash/short
    if allow_short:
        positions = np.where(preds == 1, 1.0, -1.0)
    else:
        positions = np.where(preds == 1, 1.0, 0.0)

    # Detect position switches for transaction costs
    # Initial trade into first position
    prev_positions = np.roll(positions, 1)
    prev_positions[0] = 0.0
    position_changes = np.abs(positions - prev_positions)
    num_trades = int(np.sum(position_changes > 0))

    cost_per_trade = transaction_cost_bps / 10000.0
    trade_costs = position_changes * cost_per_trade

    # Strategy net return per bar
    strategy_returns = (positions * asset_returns) - trade_costs

    # Cumulative return curves (starting at 1.0)
    strategy_cum = np.cumprod(1.0 + strategy_returns)
    benchmark_cum = np.cumprod(1.0 + asset_returns)

    total_strat_ret = float(strategy_cum[-1] - 1.0) if len(strategy_cum) > 0 else 0.0
    total_bench_ret = float(benchmark_cum[-1] - 1.0) if len(benchmark_cum) > 0 else 0.0

    # Risk Metrics Calculation
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns, ddof=1) if len(strategy_returns) > 1 else 0.0

    if std_ret > 1e-8:
        sharpe = float((mean_ret / std_ret) * np.sqrt(252))
    else:
        sharpe = 0.0
    if np.isnan(sharpe):
        sharpe = 0.0

    # Downside std for Sortino
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.0
    if downside_std > 1e-8:
        sortino = float((mean_ret / downside_std) * np.sqrt(252))
    else:
        sortino = 0.0
    if np.isnan(sortino):
        sortino = 0.0

    # Max Drawdown Calculation
    running_max = np.maximum.accumulate(strategy_cum)
    drawdowns = (strategy_cum - running_max) / running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    if np.isnan(max_dd):
        max_dd = 0.0

    # Win Rate on Active Days
    active_mask = positions != 0
    if np.sum(active_mask) > 0:
        win_rate = float(np.mean(strategy_returns[active_mask] > 0))
    else:
        win_rate = 0.0

    return BacktestResult(
        total_strategy_return=round(total_strat_ret, 4),
        total_benchmark_return=round(total_bench_ret, 4),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown=round(max_dd, 4),
        win_rate=round(win_rate, 4),
        num_trades=num_trades,
        transaction_cost_bps=transaction_cost_bps,
        strategy_curve=np.round(strategy_cum, 4).tolist(),
        benchmark_curve=np.round(benchmark_cum, 4).tolist(),
        dates=dates,
    )
