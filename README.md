# Gibbs Portfolio Lab

A UCQuants workshop app: invest $100,000 on Jan 2, 2019 across the 11 S&P 500 sector ETFs, then drag the date to see what happened. Compare your picks with portfolios built by the Gibbs formula. Educational simulation. Not investment advice.

**Live app:** https://gibbs-portfolio-distribution-lab.streamlit.app/

## Features

- **Your picks vs Gibbs:** split $100,000 across sectors yourself and compare against Gibbs portfolios (buy-and-hold, or rebalanced yearly/quarterly with 10 bps trading costs).
- **Test against historical data or a random simulation:** historical mode uses real prices. Random mode gives each sector a fixed random risk level (10-40%/yr) and simulates prices as a random walk; change the seed for a new market.
- **Optimized temperature:** a shadow on the T slider marks the T with the best return per unit of volatility from Jan 2, 2019 to the selected date. It uses hindsight, so it shows what would have worked, not a forecast.
- **Risk metrics:** portfolio value, 12-month volatility and worst drop from a peak (max drawdown), plus per-sector risk over time.

## Install (Python 3.11)

    cd gibbs-portfolio-lab
    python3.11 -m venv .venv
    .venv/bin/pip install streamlit pandas numpy plotly yfinance pytest

## Run

    cd gibbs-portfolio-lab
    .venv/bin/streamlit run app.py
    .venv/bin/pytest

The app reads only `data/prices.csv`, so it works offline.

## Refresh data (needs internet)

    cd gibbs-portfolio-lab
    .venv/bin/python scripts/fetch_data.py

## The Gibbs idea (simplified)

1. Each sector gets a risk score: how much its price swung over the past year (annual volatility, in %).
2. In physics, the Gibbs (Boltzmann) distribution says low energy states are more likely: p = exp(-E/T) / sum.
3. Here risk plays the role of energy, so calmer sectors get more money: w = exp(-risk/T) / sum.
4. Temperature T controls how picky it is: low T puts almost everything in the calmest sector, high T spreads money evenly.
5. Example: risk 12% vs 22% at T = 10 gives the calmer sector exp(1), about 2.7 times as much money.
