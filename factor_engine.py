import pandas as pd
import numpy as np
from typing import Optional

class FactorEngine:
    """
    Computes mathematical factor signals and generates cross-sectional 
    ranking matrices, completely immunized against look-ahead bias.
    """
    def __init__(self, price_matrix: pd.DataFrame):
        self.daily_prices = price_matrix
        self.monthly_prices = price_matrix.resample('ME').last()
        self.factor_scores: Optional[pd.DataFrame] = None

    def compute_12m_1m_momentum(self) -> pd.DataFrame:
        print("Computing cross-sectional 12M-1M Momentum factor signals...")
        
        # Calculate institutional momentum skipping the most recent month
        price_t_minus_1 = self.monthly_prices.shift(1)
        price_t_minus_12 = self.monthly_prices.shift(12)
        
        momentum_raw = (price_t_minus_1 / price_t_minus_12) - 1.0
        self.factor_scores = momentum_raw.dropna(how='all')
        return self.factor_scores

    def compute_cross_sectional_z_scores(self) -> pd.DataFrame:
        if self.factor_scores is None:
            self.compute_12m_1m_momentum()
            
        # Standardize cross-sectionally along axis=1 (across assets for each month)
        mean_t = self.factor_scores.mean(axis=1)
        std_t = self.factor_scores.std(axis=1)
        
        z_scores = self.factor_scores.sub(mean_t, axis=0).div(std_t, axis=0)
        return z_scores.fillna(0.0)