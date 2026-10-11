"""Pure functions for the Gibbs Portfolio Lab. No Streamlit here."""
import numpy as np
import pandas as pd

WINDOW = 252
MIN_DAYS = 120


def risk_on(prices: pd.DataFrame, date) -> pd.Series:
    """Annualized volatility (%) of daily returns over the 252 trading days strictly before `date`."""
    past = prices[prices.index < pd.Timestamp(date)]
    out = {}
    for t in prices.columns:
        rets = past[t].dropna().pct_change().dropna().tail(WINDOW)
        if len(rets) < MIN_DAYS:
            raise ValueError(f"{t}: only {len(rets)} days of history before {date}")
        out[t] = rets.std() * np.sqrt(WINDOW) * 100
    return pd.Series(out)


def gibbs_weights(risk: pd.Series, T: float) -> pd.Series:
    """w_i = exp(-risk_i / T) / sum_j exp(-risk_j / T). Shifted by the min risk for numerical safety."""
    z = np.exp(-(risk - risk.min()) / T)
    return z / z.sum()


def _start_day(prices, start):
    return prices.index[prices.index >= pd.Timestamp(start)][0]


def buy_and_hold(prices: pd.DataFrame, dollars: pd.Series, start) -> pd.Series:
    """Buy shares on `start` at that day's close and hold them."""
    d0 = _start_day(prices, start)
    p = prices.loc[d0:, dollars.index]
    shares = dollars / p.iloc[0]
    return (p * shares).sum(axis=1)


def _period_starts(index, freq):
    """First trading day of each new year/quarter, excluding the first day of the index."""
    if freq == "never":
        return []
    key = index.year if freq == "yearly" else index.year * 10 + index.quarter
    key = pd.Series(key, index=index)
    return list(index[(key != key.shift())][1:])


def gibbs_rebalanced(prices: pd.DataFrame, T: float, freq: str, cost_bps: float, start):
    """Gibbs portfolio, re-weighted on the first trading day of each period.

    Returns (values, rebalance_dates, costs, traded) where costs/traded are Series indexed by rebalance date.
    """
    d0 = _start_day(prices, start)
    p = prices.loc[d0:]
    capital = 100_000.0
    shares = gibbs_weights(risk_on(prices, d0), T) * capital / p.iloc[0]
    rebal = set(_period_starts(p.index, freq))
    values, costs, traded = [], {}, {}
    for d, row in p.iterrows():
        if d in rebal:
            current = shares * row
            v = current.sum()
            target = gibbs_weights(risk_on(prices, d), T) * v
            trade = (target - current).abs().sum()
            cost = cost_bps / 10_000 * trade
            shares = target * (v - cost) / v / row
            costs[d], traded[d] = cost, trade
        values.append((shares * row).sum())
    return pd.Series(values, index=p.index), sorted(rebal), pd.Series(costs, dtype=float), pd.Series(traded, dtype=float)


def max_drawdown(series: pd.Series) -> float:
    """Worst drop from a previous peak, in percent (negative or zero)."""
    return float((series / series.cummax() - 1).min() * 100)


def rolling_vol(series: pd.Series) -> pd.Series:
    """12-month (252 day) annualized volatility of a value series, in percent."""
    return series.pct_change().rolling(WINDOW, min_periods=20).std() * np.sqrt(WINDOW) * 100
