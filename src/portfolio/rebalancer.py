import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')

from portfolio.objective import PortfolioObjective
from portfolio.constraints import PortfolioConstraints


class PortfolioRebalancer:

    def __init__(
        self,
        asset_names: list,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints,
        transaction_cost: float = 0.001,
        rebalance_freq: str = "monthly"
    ):
        self.asset_names      = asset_names
        self.n_assets         = len(asset_names)
        self.objective        = objective
        self.constraints      = constraints
        self.transaction_cost = transaction_cost
        self.rebalance_freq   = rebalance_freq

        self.current_weights  = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value  = 1.0

        self.rebalance_log    = []
        self.value_history    = []

    def rebalance(self, new_weights: np.ndarray, date: str = None) -> dict:
        repaired = self.constraints.repair(new_weights)

        turnover = 0.5 * np.sum(np.abs(repaired - self.current_weights))
        cost     = turnover * self.transaction_cost * 2

        metrics  = self.objective.get_all_metrics(repaired)

        active_assets = {
            self.asset_names[i]: round(float(repaired[i]), 4)
            for i in range(self.n_assets) if repaired[i] > 1e-6
        }

        event = {
            "date"            : date or "N/A",
            "n_assets"        : len(active_assets),
            "assets"          : active_assets,
            "turnover"        : round(float(turnover), 4),
            "transaction_cost": round(float(cost), 6),
            "expected_return" : metrics["expected_return_pct"],
            "volatility"      : metrics["volatility_pct"],
            "sharpe_ratio"    : metrics["sharpe_ratio"],
            "sortino_ratio"   : metrics["sortino_ratio"],
            "weights_sum"     : round(float(repaired.sum()), 6)
        }

        self.current_weights = repaired.copy()
        self.constraints.update_prev_weights(repaired)
        self.rebalance_log.append(event)

        return event

    def simulate_period_return(self, period_returns: pd.DataFrame) -> float:
        available = [a for a in self.asset_names if a in period_returns.columns]
        weights_aligned = np.array([
            self.current_weights[self.asset_names.index(a)] for a in available
        ])

        if weights_aligned.sum() > 0:
            weights_aligned /= weights_aligned.sum()

        daily_portfolio = period_returns[available].values @ weights_aligned
        period_return   = float(np.prod(1 + daily_portfolio) - 1)
        self.portfolio_value *= (1 + period_return)
        self.value_history.append(self.portfolio_value)

        return period_return

    def get_rebalance_dates(self, date_index: pd.DatetimeIndex) -> list:
        dates = pd.Series(date_index)

        if self.rebalance_freq == "monthly":
            rebalance = dates.groupby(
                dates.dt.to_period("M")
            ).first().tolist()
        elif self.rebalance_freq == "quarterly":
            rebalance = dates.groupby(
                dates.dt.to_period("Q")
            ).first().tolist()
        else:
            rebalance = [date_index[0]]

        return rebalance

    def get_rebalance_history(self) -> pd.DataFrame:
        if not self.rebalance_log:
            return pd.DataFrame()
        return pd.DataFrame(self.rebalance_log)

    def get_current_portfolio(self) -> dict:
        metrics = self.objective.get_all_metrics(self.current_weights)
        active  = {
            self.asset_names[i]: round(float(self.current_weights[i]), 4)
            for i in range(self.n_assets)
            if self.current_weights[i] > 1e-6
        }
        return {
            "holdings"       : active,
            "n_assets"       : len(active),
            "portfolio_value": round(self.portfolio_value, 6),
            **metrics
        }

    def print_summary(self):
        print("\n" + "=" * 60)
        print("  PORTFOLIO REBALANCING SUMMARY")
        print("=" * 60)

        for i, event in enumerate(self.rebalance_log):
            print(f"\n  Rebalance #{i+1} | Date: {event['date']}")
            print(f"    Assets       : {event['n_assets']}")
            print(f"    Sharpe Ratio : {event['sharpe_ratio']}")
            print(f"    Return       : {event['expected_return']}%")
            print(f"    Volatility   : {event['volatility']}%")
            print(f"    Turnover     : {event['turnover']}")
            print(f"    Trans. Cost  : {event['transaction_cost']}")
            print(f"    Holdings     : {event['assets']}")

        print("\n" + "=" * 60)
        print(f"  Final Portfolio Value : {round(self.portfolio_value, 4)}")
        print("=" * 60)
