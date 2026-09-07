# Quantitative Factor-Investing Backtester

A Python-based cross-sectional momentum backtesting project implementing
the 12–1 momentum factor on a universe of U.S. equities.

## Project Overview

The project constructs and evaluates two momentum strategies:

- Long-only momentum: invests in the top 20% of stocks ranked by 12–1 momentum.
- Long-short momentum: goes long the top 20% and short the bottom 20%.

Performance is compared with an equal-weight benchmark.

## Momentum Signal

The 12–1 momentum signal is defined as:

Momentum = P(t-1) / P(t-12) - 1

The most recent month is excluded to reduce the effect of short-term reversal.

## Methodology

1. Download historical equity prices using Yahoo Finance.
2. Clean and convert daily prices to month-end prices.
3. Calculate 12–1 momentum.
4. Standardize signals cross-sectionally using z-scores.
5. Rank stocks by percentile.
6. Construct equal-weight long and long-short portfolios.
7. Shift portfolio weights by one month to prevent look-ahead bias.
8. Compare strategy performance with an equal-weight benchmark.

## Project Structure

- `data_pipeline.py` – downloads and prepares historical price data.
- `factor_engine.py` – calculates momentum signals and cross-sectional rankings.
- `backtester_engine.py` – constructs portfolios and calculates strategy returns.
- `12_1_momentum_notebook_no_classes.ipynb` – step-by-step notebook implementation.

## Key Concepts

- Cross-sectional factor investing
- 12–1 momentum
- Z-score normalization
- Percentile ranking
- Long-short portfolios
- Look-ahead bias prevention
- Portfolio backtesting

## Technologies

Python, pandas, NumPy, yfinance, matplotlib