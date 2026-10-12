"""Gibbs Portfolio Lab: Streamlit UI. All math lives in lab/core.py."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab.core import buy_and_hold, gibbs_rebalanced, gibbs_weights, max_drawdown, optimal_temperature, risk_on, rolling_vol, simulate_prices

HERE = Path(__file__).resolve().parent
LOGO = HERE / "assets" / "ucquants_logo.jpeg"
START = pd.Timestamp("2019-01-02")
CAPITAL = 100_000
COST_BPS = 10
SECTORS = {
    "XLK": "Technology", "XLV": "Healthcare", "XLE": "Energy", "XLF": "Financials",
    "XLU": "Utilities", "XLP": "Consumer Staples", "XLY": "Consumer Discretionary",
    "XLI": "Industrials", "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Communication",
}
# Every line differs by color AND dash style, and is named in the legend.
STYLE = {
    "Your picks": ("#2B50D6", "solid"),
    "Gibbs (hold)": ("#C2571A", "dash"),
    "Gibbs (rebalanced)": ("#C026D3", "dashdot"),
}

st.set_page_config(page_title="Gibbs Portfolio Lab", page_icon=str(LOGO), layout="wide")
st.logo(str(LOGO), size="large")
st.markdown(
    """<style>
    .block-container {padding-top: 2.5rem;}
    .eyebrow {font-family: monospace; letter-spacing: .12em; font-size: .75rem; color: #6B7280;}
    .muted {color: #6B7280; font-size: .85rem;}
    .gibbs {color: #C2571A; font-family: monospace; padding-top: .55rem;}
    .note {border: 1px solid #E5E7EB; border-radius: 6px; padding: .4rem .7rem; font-size: .8rem;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_prices(seed=None):
    """Real prices, or a random simulated market on the same calendar when `seed` is given."""
    real = pd.read_csv(HERE / "data" / "prices.csv", index_col="date", parse_dates=True)[list(SECTORS)]
    return real if seed is None else simulate_prices(real.index, SECTORS, seed)


@st.cache_data
def run_hold(dollars: tuple, seed):
    return buy_and_hold(load_prices(seed), pd.Series(dollars, index=list(SECTORS)), START)


@st.cache_data
def run_rebalanced(T, freq, seed):
    return gibbs_rebalanced(load_prices(seed), T, freq, COST_BPS, START)


@st.cache_data
def best_T(date, seed):
    return optimal_temperature(load_prices(seed), START, date)


@st.cache_data
def risk_at(date, seed):
    return risk_on(load_prices(seed), date)


prices = load_prices()
last = prices.index[-1]
days = prices.index[prices.index >= START]
# Month steps: first trading day of each month, plus the last data date.
months = list(pd.Series(days, index=days).groupby(days.to_period("M")).first())
if months[-1] != last:
    months.append(last)

# Header
logo_col, head_col, note_col = st.columns([1, 6, 3], vertical_alignment="center")
logo_col.image(str(LOGO), width=120)
with head_col:
    st.markdown('<div class="eyebrow">UCQUANTS WORKSHOP · RISK MODELLING</div>', unsafe_allow_html=True)
    st.title("Gibbs Portfolio Lab", anchor=False)
    st.markdown(
        "Invest $100,000 on Jan 2, 2019 and hold it. Drag the date to reveal what happened, "
        "and compare your picks with the portfolios the Gibbs formula would choose."
    )
note_col.markdown('<div class="note">Educational simulation. Not investment advice.</div>', unsafe_allow_html=True)

left, mid, right = st.columns([1.1, 1, 1.1], gap="medium")

# Controls first, since the allocation column needs T.
with mid, st.container(border=True):
    st.subheader("Controls", anchor=False)
    mode = st.radio("Test against", ["Historical data", "Random simulation"], horizontal=True)
    seed = None
    if mode == "Random simulation":
        seed = st.number_input("Random seed (new seed = new random market)", min_value=0, value=42, step=1)
        st.caption(
            "Each sector gets a random, fixed risk level and prices follow a random walk. "
            "Same calendar as the real data, but none of the prices are real."
        )
    date = st.select_slider(
        "Simulation date", options=months, value=months[0], format_func=lambda d: d.strftime("%b %d, %Y"), key="sim_date",
    )
    T = st.slider("Temperature (T)", min_value=2, max_value=50, value=10)
    if date > START:
        opt = best_T(date, seed)
        pct = (opt - 2) / 48 * 100
        # One line of HTML: indented lines would be parsed as a markdown code block.
        st.markdown(
            f'<div style="position:relative;height:6px;margin:1.6rem .5rem .4rem;background:#E5E7EB;border-radius:3px">'
            f'<div style="position:absolute;left:{pct}%;transform:translateX(-50%);top:-1.4rem;color:#C2571A;'
            f'font:600 .8rem monospace;white-space:nowrap">T={opt}</div>'
            f'<div style="position:absolute;left:{pct}%;transform:translateX(-50%);top:-4px;width:14px;height:14px;'
            f'border-radius:50%;background:#C2571A;opacity:.45"></div></div>'
            f'<div class="muted">Shadow: optimized T = <b>{opt}</b> (best return per unit of volatility, '
            f'Jan 2, 2019 to {date:%b %d, %Y}, hindsight)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="muted">Drag the date forward to see the optimized T.</div>', unsafe_allow_html=True)

    freq = st.selectbox("Re-calculate risk", ["Never", "Yearly", "Quarterly"]).lower()
    with st.container(border=True):
        st.markdown("**How the Gibbs portfolio picks**")
        st.latex(r"w_i = \frac{e^{-\mathrm{risk}_i / T}}{\sum_j e^{-\mathrm{risk}_j / T}}")
        st.markdown(
            "Each sector's risk is how much its price swung over the past 12 months, in percent. "
            "Calmer sectors get more money, and T sets how strongly: low T piles into the calmest few, "
            "high T spreads money almost evenly."
        )

risk0 = risk_at(START, seed)
gibbs_dollars = gibbs_weights(risk0, T) * CAPITAL

with left, st.container(border=True):
    st.subheader("Allocation on Jan 2, 2019", anchor=False)
    h1, h2 = st.columns([3, 1])
    h1.markdown('<div class="muted">YOU (dollars)</div>', unsafe_allow_html=True)
    h2.markdown(f'<div class="muted" style="color:#C2571A">GIBBS (T = {T})</div>', unsafe_allow_html=True)
    for t in SECTORS:
        st.session_state.setdefault(t, 0)
    picks = {}
    for t, name in SECTORS.items():
        c1, c2 = st.columns([3, 1])
        picks[t] = c1.number_input(f"{name} ({t})", min_value=0, max_value=CAPITAL, step=1_000, key=t)
        c2.markdown(f'<div class="gibbs"><br>${gibbs_dollars[t]:,.0f}</div>', unsafe_allow_html=True)
    total = sum(picks.values())
    c1, c2 = st.columns(2)
    c1.metric("Allocated", f"${total:,}")
    c2.metric("Remaining", f"${CAPITAL - total:,}")
    st.button(
        "Reset all to $0", on_click=lambda: st.session_state.update({t: 0 for t in SECTORS})
    )
    # 11 inputs of $9,091 add to $100,001, so allow a few dollars of rounding and scale to exactly $100,000.
    picks_ok = abs(total - CAPITAL) <= len(SECTORS)
    if not picks_ok:
        st.warning(f"Your picks add up to \\${total:,}. They must add up to exactly \\$100,000 before your line is drawn.")
    elif total != CAPITAL:
        st.caption(f"Rounding: your ${total:,} is scaled to exactly $100,000.")

# Strategies
series = {}
if picks_ok:
    series["Your picks"] = run_hold(tuple(v * CAPITAL / total for v in picks.values()), seed)
series["Gibbs (hold)"] = run_hold(tuple(gibbs_dollars), seed)
rebal_dates, costs = [], pd.Series(dtype=float)
if freq != "never":
    series["Gibbs (rebalanced)"], rebal_dates, costs, _ = run_rebalanced(T, freq, seed)

with right, st.container(border=True):
    st.subheader(f"On {date:%b %d, %Y}", anchor=False)
    for name, s in series.items():
        upto = s.loc[:date]
        v = upto.iloc[-1]
        vol = rolling_vol(upto).iloc[-1]
        st.markdown(f"**{name}**")
        a, b, c = st.columns(3)
        a.metric("Value", f"${v:,.0f}", f"{(v / CAPITAL - 1) * 100:+.1f}% since start")
        b.metric("12-month volatility", "n/a" if pd.isna(vol) else f"{vol:.0f}%")
        c.metric("Worst drop from a peak", f"{max_drawdown(upto):.0f}%")
        if name == "Gibbs (rebalanced)":
            paid = costs.loc[:date]
            st.markdown(f"Rebalancing costs paid so far: **${paid.sum():,.2f}** ({len(paid)} rebalances)")
    if not picks_ok:
        st.info("Your picks are hidden until they add up to $100,000.")

# Value over time
fig = go.Figure()
for name, s in series.items():
    color, dash = STYLE[name]
    upto = s.loc[:date]
    fig.add_scatter(x=upto.index, y=upto, name=name, mode="lines", line=dict(color=color, dash=dash, width=2))
    if name == "Gibbs (rebalanced)":
        marks = [d for d in rebal_dates if d <= date]
        fig.add_scatter(
            x=marks, y=s.loc[marks], name="Rebalance day", mode="markers",
            marker=dict(color=color, symbol="diamond", size=9, line=dict(color="white", width=1)),
        )
fig.add_vline(x=date, line=dict(color="#111827", width=1))
if date < last:
    fig.add_annotation(
        x=date + (last - date) / 2, y=0.5, yref="paper", text="Drag the date forward to reveal", showarrow=False,
        font=dict(color="#6B7280", size=14),
    )
fig.update_layout(
    title="Portfolio value over time", height=430, margin=dict(l=10, r=10, t=50, b=10),
    xaxis=dict(range=[START, last]), yaxis=dict(tickprefix="$", tickformat=",.0f"),
    legend=dict(orientation="h", y=1.08, x=1, xanchor="right"), hovermode="x unified",
)
st.plotly_chart(fig, width="stretch")

# Risk per sector
risk_now = risk_at(date, seed)
order = risk0.sort_values().index
labels = [SECTORS[t] for t in order]
bars = go.Figure()
bars.add_bar(
    y=labels, x=risk_now[order], orientation="h", name=f"Risk on {date:%b %d, %Y}", marker_color="#C026D3",
    text=[f"now {risk_now[t]:.1f}%  |  Jan 2, 2019: {risk0[t]:.1f}%" for t in order], textposition="outside",
)
bars.add_scatter(
    y=labels, x=risk0[order], mode="markers", name="Risk on Jan 2, 2019",
    marker=dict(symbol="line-ns", size=22, line=dict(width=3, color="#111827")),
)
bars.update_layout(
    title="Risk per sector (12-month volatility, %)", height=460, margin=dict(l=10, r=10, t=50, b=10),
    xaxis=dict(range=[0, max(risk_now.max(), risk0.max()) * 1.6], ticksuffix="%"),
    yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
)
st.plotly_chart(bars, width="stretch")
