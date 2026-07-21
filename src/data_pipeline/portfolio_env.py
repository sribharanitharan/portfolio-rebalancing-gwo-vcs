import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')


class PortfolioEnvironment:

    def __init__(
        self,
        returns,
        covariance,
        expected_returns,
        cardinality_min=3,
        cardinality_max=10,
        weight_min=0.01,
        weight_max=0.40,
        turnover_limit=0.30,
        risk_free_rate=0.04,
        transaction_cost=0.001
    ):
        self.returns          = returns
        self.covariance       = covariance.values
        self.expected_returns = expected_returns.values
        self.n_assets         = len(expected_returns)
        self.asset_names      = list(expected_returns.index)

        self.cardinality_min  = cardinality_min
        self.cardinality_max  = cardinality_max
        self.weight_min       = weight_min
        self.weight_max       = weight_max
        self.turnover_limit   = turnover_limit
        self.risk_free_rate   = risk_free_rate
        self.transaction_cost = transaction_cost

        self.current_weights  = np.ones(self.n_assets) / self.n_assets

    def repair_weights(self, weights):
        weights = np.clip(weights, 0, self.weight_max)

        if np.sum(weights > 0) > self.cardinality_max:
            threshold_idx = np.argsort(weights)[:-self.cardinality_max]
            weights[threshold_idx] = 0.0

        active = np.sum(weights > 0)
        if active < self.cardinality_min:
            zero_idx = np.where(weights == 0)[0]
            activate = np.random.choice(
                zero_idx,
                size=self.cardinality_min - active,
                replace=False
            )
            weights[activate] = self.weight_min

        weights[weights > 0] = np.maximum(weights[weights > 0], self.weight_min)

        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            idx = np.random.choice(self.n_assets, self.cardinality_min, replace=False)
            weights = np.zeros(self.n_assets)
            weights[idx] = 1.0 / self.cardinality_min

        return weights

    def check_turnover(self, new_weights):
        return 0.5 * np.sum(np.abs(new_weights - self.current_weights))

    def apply_turnover_penalty(self, new_weights, penalty_strength=10.0):
        excess = max(0, self.check_turnover(new_weights) - self.turnover_limit)
        return penalty_strength * excess

    def update_current_weights(self, new_weights):
        self.current_weights = new_weights.copy()

    def portfolio_return(self, weights):
        return float(np.dot(weights, self.expected_returns))

    def portfolio_variance(self, weights):
        return float(weights @ self.covariance @ weights)

    def portfolio_std(self, weights):
        return float(np.sqrt(self.portfolio_variance(weights)))

    def sharpe_ratio(self, weights):
        ret = self.portfolio_return(weights)
        std = self.portfolio_std(weights)
        if std < 1e-9:
            return -999.0
        return (ret - self.risk_free_rate) / std

    def fitness(self, weights, apply_repair=True, apply_turnover_penalty=True):
        if apply_repair:
            weights = self.repair_weights(weights)
        neg_sharpe = -self.sharpe_ratio(weights)
        if apply_turnover_penalty:
            neg_sharpe += self.apply_turnover_penalty(weights)
        return neg_sharpe

    def get_portfolio_summary(self, weights):
        weights = self.repair_weights(weights)
        active_assets = {
            self.asset_names[i]: round(float(weights[i]), 4)
            for i in range(self.n_assets) if weights[i] > 0
        }
        return {
            "assets"          : active_assets,
            "n_assets"        : sum(1 for w in weights if w > 0),
            "expected_return" : round(self.portfolio_return(weights) * 100, 3),
            "volatility"      : round(self.portfolio_std(weights) * 100, 3),
            "sharpe_ratio"    : round(self.sharpe_ratio(weights), 4),
            "turnover"        : round(self.check_turnover(weights), 4),
            "weights_sum"     : round(float(weights.sum()), 6)
        }


if __name__ == "__main__":
    from preprocessor import compute_returns, compute_covariance_matrix, compute_expected_returns
    from data_loader import load_stock_data, extract_close_prices

    raw_df   = load_stock_data("data/raw/stock_prices.csv")
    close_df = extract_close_prices(raw_df)
    returns  = compute_returns(close_df)
    cov      = compute_covariance_matrix(returns)
    exp_ret  = compute_expected_returns(returns)

    env        = PortfolioEnvironment(returns, cov, exp_ret)
    eq_weights = np.ones(env.n_assets) / env.n_assets
    summary    = env.get_portfolio_summary(eq_weights)

    print("\n=== Equal-Weight Portfolio Test ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
