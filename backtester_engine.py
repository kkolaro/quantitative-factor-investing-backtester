import pandas as pd
import numpy as np
from typing import Tuple

class LongShortBacktester:
    """
    Simulates monthly portfolio rebalancing based on cross-sectional factor rankings,
    tracking performance metrics and compounding returns over time.
    """
    def __init__(self, factor_z_scores: pd.DataFrame, monthly_returns: pd.DataFrame):
        self.signals = factor_z_scores
        # Forward shift returns to avoid peeking into the future
        self.forward_returns = monthly_returns.shift(-1).loc[self.signals.index]
        
    def execute_quintile_strategy(self, quantile_pct: float = 0.20) -> Tuple[pd.Series, pd.Series]:
        print(f"Simulating strategy with a {quantile_pct*100}% tail selection rebalance...")
        long_portfolio_returns, short_portfolio_returns, strategy_dates = [], [], []
        
        for date, row_signal in self.signals.iterrows():
            actual_returns = self.forward_returns.loc[date]
            valid_idx = row_signal.dropna().index.intersection(actual_returns.dropna().index)
            if len(valid_idx) < 2:
                continue
                
            period_signals = row_signal.loc[valid_idx]
            period_returns = actual_returns.loc[valid_idx]
            
            k = max(1, int(len(period_signals) * quantile_pct))
            top_assets = period_signals.nlargest(k).index
            bottom_assets = period_signals.nsmallest(k).index
            
            strategy_dates.append(date)
            long_portfolio_returns.append(period_returns.loc[top_assets].mean())
            short_portfolio_returns.append(period_returns.loc[bottom_assets].mean())
            
        ts_long = pd.Series(long_portfolio_returns, index=strategy_dates).fillna(0.0)
        ts_short = pd.Series(short_portfolio_returns, index=strategy_dates).fillna(0.0)
        return (ts_long - ts_short), ts_long

    @staticmethod
    def calculate_performance_metrics(strategy_returns: pd.Series) -> dict:
        cumulative_growth = (1.0 + strategy_returns).cumprod()
        total_return = cumulative_growth.iloc[-1] - 1.0 if not cumulative_growth.empty else 0.0
        
        ann_return = strategy_returns.mean() * 12
        ann_volatility = strategy_returns.std() * np.sqrt(12)
        sharpe_ratio = ann_return / ann_volatility if ann_volatility > 0 else 0.0
        
        running_max = cumulative_growth.cummax()
        max_drawdown = ((cumulative_growth - running_max) / running_max).min()
        
        return {
            "Total Return": f"{total_return * 100:.2f}%",
            "Annualized Return": f"{ann_return * 100:.2f}%",
            "Annualized Volatility": f"{ann_volatility * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown * 100:.2f}%"
        }

# ==============================================================================
# RUN THE COMPLETE BACKTESTER
# ==============================================================================
if __name__ == "__main__":
    from data_pipeline import FinancialDataPipeline
    from factor_engine import FactorEngine
    
    # 15 Liquid S&P 500 components for the trial
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "GS", "XOM", "CVX", "LLY", "META", "UNH", "HD", "PG", "V"]
    
    pipeline = FinancialDataPipeline(tickers=universe, start_date="2016-01-01", end_date="2026-01-01")
    prices = pipeline.clean_and_structure_data()
    m_returns = pipeline.compute_monthly_returns()
    
    engine = FactorEngine(price_matrix=prices)
    z_scores = engine.compute_cross_sectional_z_scores()
    
    backtester = LongShortBacktester(factor_z_scores=z_scores, monthly_returns=m_returns)
    ls_returns, _ = backtester.execute_quintile_strategy(quantile_pct=0.20)
    
    report = LongShortBacktester.calculate_performance_metrics(ls_returns)
    print("\n=========================================\n QUANT PORTFOLIO PERFORMANCE REPORT      \n=========================================")
    for metric, value in report.items():
        print(f"{metric:<25}: {value}")
    print("=========================================")