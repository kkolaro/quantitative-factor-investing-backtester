import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional


class FinancialDataPipeline:
    """
    A production-grade pipeline to fetch, clean, and structure historical financial data
    for cross-sectional factor backtesting, eliminating look-ahead bias.
    """

    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data: Optional[pd.DataFrame] = None
        self.cleaned_prices: Optional[pd.DataFrame] = None

    def fetch_raw_data(self) -> pd.DataFrame:
        print(f"Initiating data download for {len(self.tickers)} tickers...")

        try:
            data = yf.download(
                tickers=self.tickers,
                start=self.start_date,
                end=self.end_date,
                group_by='ticker',
                auto_adjust=True
            )

            self.raw_data = data
            return self.raw_data

        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch data from API: {e}"
            )

    def clean_and_structure_data(self) -> pd.DataFrame:

        if self.raw_data is None:
            self.fetch_raw_data()

        prices_dict = {}

        for ticker in self.tickers:

            if ticker in self.raw_data.columns.levels[0]:

                prices_dict[ticker] = (
                    self.raw_data[ticker]['Close']
                )

        df_prices = pd.DataFrame(prices_dict)

        df_prices.index = pd.to_datetime(
            df_prices.index
        )

        # Forward-fill missing prices
        df_prices = df_prices.ffill()

        # Do not backward-fill using future prices

        df_prices = df_prices.dropna(
            axis=1,
            how='all'
        )

        self.cleaned_prices = df_prices

        print(
            f"Data cleaning complete. "
            f"Structured shape: "
            f"{self.cleaned_prices.shape}"
        )

        return self.cleaned_prices

    def compute_monthly_returns(self) -> pd.DataFrame:

        if self.cleaned_prices is None:
            self.clean_and_structure_data()

        monthly_prices = (
            self.cleaned_prices
            .resample('ME')
            .last()
        )

        monthly_returns = (
            monthly_prices
            .pct_change(fill_method=None)
            .dropna(how='all')
        )

        return monthly_returns