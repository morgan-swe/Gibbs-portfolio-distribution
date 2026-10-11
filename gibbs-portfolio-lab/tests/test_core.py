import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.core import buy_and_hold, gibbs_rebalanced, gibbs_weights, risk_on  # noqa: E402

START = "2019-01-02"


@pytest.fixture(scope="module")
def prices():
    return pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "prices.csv", index_col="date", parse_dates=True)


def test_gibbs_weights(prices):
    risk = risk_on(prices, START)
    assert gibbs_weights(risk, 10).sum() == pytest.approx(1)
    assert np.allclose(gibbs_weights(risk, 1e9), 1 / 11)
    assert gibbs_weights(risk, 0.05)[risk.idxmin()] > 0.99
    r = pd.Series({"a": 12.0, "b": 22.0})
    w = gibbs_weights(r, 10)
    assert w["a"] / w["b"] == pytest.approx(np.e)


def test_risk_on_uses_only_past(prices):
    before = risk_on(prices, "2020-03-02")
    changed = prices.copy()
    changed.loc[changed.index >= "2020-03-02"] *= 3.7
    changed.loc["2020-03-02"] = 1.0
    pd.testing.assert_series_equal(before, risk_on(changed, "2020-03-02"))


def test_buy_and_hold_single_etf(prices):
    dollars = pd.Series(0.0, index=prices.columns)
    dollars["XLE"] = 100_000
    v = buy_and_hold(prices, dollars, START)
    p = prices.loc[START:, "XLE"]
    assert np.allclose(v, p / p.iloc[0] * 100_000)


def test_never_equals_gibbs_hold(prices):
    hold = buy_and_hold(prices, gibbs_weights(risk_on(prices, START), 10) * 100_000, START)
    vals, dates, costs, _ = gibbs_rebalanced(prices, 10, "never", 10, START)
    assert np.allclose(vals, hold, rtol=0, atol=1e-8)
    assert dates == [] and costs.sum() == 0


def test_cost_bps(prices):
    eq = pd.Series(100_000 / 11, index=prices.columns)
    hold_w = gibbs_weights(risk_on(prices, START), 10) * 100_000
    before = (buy_and_hold(prices, eq, START), buy_and_hold(prices, hold_w, START))
    _, d0, c0, t0 = gibbs_rebalanced(prices, 10, "yearly", 0, START)
    _, d10, c10, t10 = gibbs_rebalanced(prices, 10, "yearly", 10, START)
    assert d0 == d10 and len(d10) > 0
    assert (c0 == 0).all()
    assert np.allclose(c10 - c0, 0.001 * t10, rtol=0, atol=1e-9)
    # strategies 1 to 3 never trade, so they are unchanged
    pd.testing.assert_series_equal(before[0], buy_and_hold(prices, eq, START))
    pd.testing.assert_series_equal(before[1], buy_and_hold(prices, hold_w, START))
