import numpy as np
import pandas as pd


class PortfolioObjective:
    """
    Portfolio Objective Functions
    ──────────────────────────────
    Defines WHAT we are optimizing.

    Available objectives:
        1. sharpe_ratio     → Maximize risk-adjusted return (default)
        2. min_variance     → Minimize portfolio risk only
        3. mean_variance    → Balance return vs risk (Markowitz)
        4. sortino_ratio    → Like Sharpe but penalizes downside risk only
        5. calmar_ratio     → Return / Max Drawdown

    All return NEGATIVE value (minimization problem for optimizer).
    Lower fitness = better portfolio.
    """

    def __init__(
        self,
        expected_returns: np.ndarray,   # Annualized expected return per asset
        covariance: np.ndarray,         # Annualized covariance matrix
        returns_df: pd.DataFrame,       # Daily returns DataFrame
        risk_free_rate: float = 0.04,   # Annual risk-free rate
        objective_type: str = "sharpe"  # Which objective to use
    ):
        self.expected_returns = expected_returns
        self.covariance       = covariance
        self.returns_df       = returns_df
        self.risk_free_rate   = risk_free_rate
        self.objective_type   = objective_type
        self.n_assets         = len(expected_returns)

    # ── Core Metrics ──────────────────────────────────────────────────────────

    def portfolio_return(self, weights: np.ndarray) -> float:
        """Annualized expected portfolio return."""
        return float(np.dot(weights, self.expected_returns))

    def portfolio_variance(self, weights: np.ndarray) -> float:
        """Annualized portfolio variance."""
        return float(weights @ self.covariance @ weights)

    def portfolio_std(self, weights: np.ndarray) -> float:
        """Annualized portfolio standard deviation (volatility)."""
        return float(np.sqrt(max(self.portfolio_variance(weights), 1e-12)))

    # ── Objective Functions ───────────────────────────────────────────────────

    def sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Sharpe Ratio = (Return - Risk Free Rate) / Std Dev
        Higher = better → return negative for minimizer.
        """
        ret = self.portfolio_return(weights)
        std = self.portfolio_std(weights)
        return (ret - self.risk_free_rate) / std

    def min_variance(self, weights: np.ndarray) -> float:
        """
        Minimum Variance Objective.
        Purely minimize risk regardless of return.
        """
        return self.portfolio_variance(weights)

    def mean_variance(
        self,
        weights: np.ndarray,
        risk_aversion: float = 2.0
    ) -> float:
        """
        Markowitz Mean-Variance:
        Maximize: Return - (risk_aversion/2) * Variance

        risk_aversion controls return vs risk tradeoff:
            Low  (0.5) → aggressive (return focused)
            High (5.0) → conservative (risk focused)
        """
        ret = self.portfolio_return(weights)
        var = self.portfolio_variance(weights)
        return ret - (risk_aversion / 2.0) * var

    def sortino_ratio(self, weights: np.ndarray) -> float:
        """
        Sortino Ratio = (Return - RFR) / Downside Deviation
        Only penalizes negative returns (downside risk).
        Better than Sharpe for asymmetric return distributions.
        """
        ret = self.portfolio_return(weights)

        # Compute portfolio daily returns
        portfolio_daily = self.returns_df.values @ weights

        # Downside deviation — only negative returns
        downside    = portfolio_daily[portfolio_daily < 0]
        if len(downside) == 0:
            return (ret - self.risk_free_rate) / 1e-9

        downside_std = np.std(downside) * np.sqrt(252)  # Annualize

        if downside_std < 1e-9:
            return (ret - self.risk_free_rate) / 1e-9

        return (ret - self.risk_free_rate) / downside_std

    def calmar_ratio(self, weights: np.ndarray) -> float:
        """
        Calmar Ratio = Annualized Return / Max Drawdown
        Useful for evaluating long-term risk-adjusted performance.
        """
        ret = self.portfolio_return(weights)

        # Compute portfolio cumulative returns
        portfolio_daily  = self.returns_df.values @ weights
        cumulative       = np.cumprod(1 + portfolio_daily)
        rolling_max      = np.maximum.accumulate(cumulative)
        drawdowns        = (cumulative - rolling_max) / (rolling_max + 1e-9)
        max_drawdown     = abs(drawdowns.min())

        if max_drawdown < 1e-9:
            return ret / 1e-9

        return ret / max_drawdown

    # ── Main Fitness Function ─────────────────────────────────────────────────

    def compute(self, weights: np.ndarray) -> float:
        """
        Returns NEGATIVE objective value (for minimization).
        Called by optimizer as the fitness function.

        Args:
            weights : np.ndarray — portfolio weights

        Returns:
            float — negative objective (lower = better)
        """
        if self.objective_type == "sharpe":
            value = self.sharpe_ratio(weights)

        elif self.objective_type == "min_variance":
            value = -self.min_variance(weights)   # already minimizing

        elif self.objective_type == "mean_variance":
            value = self.mean_variance(weights)

        elif self.objective_type == "sortino":
            value = self.sortino_ratio(weights)

        elif self.objective_type == "calmar":
            value = self.calmar_ratio(weights)

        else:
            raise ValueError(f"Unknown objective: {self.objective_type}")

        return -value   # Negate → minimization

    def get_all_metrics(self, weights: np.ndarray) -> dict:
        """
        Returns all metrics for a given weight vector.
        Used in evaluation and reporting.
        """
        return {
            "expected_return_pct" : round(self.portfolio_return(weights) * 100, 3),
            "volatility_pct"      : round(self.portfolio_std(weights) * 100, 3),
            "variance"            : round(self.portfolio_variance(weights), 6),
            "sharpe_ratio"        : round(self.sharpe_ratio(weights), 4),
            "sortino_ratio"       : round(self.sortino_ratio(weights), 4),
            "calmar_ratio"        : round(self.calmar_ratio(weights), 4),
        }
