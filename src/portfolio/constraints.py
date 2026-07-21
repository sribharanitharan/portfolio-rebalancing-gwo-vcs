import numpy as np


class PortfolioConstraints:
    """
    Portfolio Constraints Handler
    ──────────────────────────────
    Defines and enforces all real-world portfolio constraints.

    Constraints handled:
        1. Budget      → weights must sum to 1.0
        2. Box         → each weight in [weight_min, weight_max]
        3. Cardinality → exactly K assets selected (not more, not less)
        4. Turnover    → total weight change limited per rebalance
        5. Long-only   → no short selling (weights >= 0)

    Two modes:
        repair()  → fix an infeasible solution (used during optimization)
        penalty() → compute constraint violation penalty (added to fitness)
    """

    def __init__(
        self,
        n_assets: int,
        cardinality_min: int   = 3,
        cardinality_max: int   = 10,
        weight_min: float      = 0.01,
        weight_max: float      = 0.40,
        turnover_limit: float  = 0.30,
        penalty_lambda: float  = 10.0   # Penalty strength for violations
    ):
        self.n_assets        = n_assets
        self.cardinality_min = cardinality_min
        self.cardinality_max = cardinality_max
        self.weight_min      = weight_min
        self.weight_max      = weight_max
        self.turnover_limit  = turnover_limit
        self.penalty_lambda  = penalty_lambda

        # Reference weights for turnover calculation
        self.prev_weights    = np.ones(n_assets) / n_assets

    # ── Individual Constraint Checks ─────────────────────────────────────────

    def check_budget(self, weights: np.ndarray) -> float:
        """Returns violation: |sum(weights) - 1.0|"""
        return abs(weights.sum() - 1.0)

    def check_box(self, weights: np.ndarray) -> float:
        """Returns total box constraint violation."""
        lower_viol = np.sum(np.maximum(0, self.weight_min - weights[weights > 0]))
        upper_viol = np.sum(np.maximum(0, weights - self.weight_max))
        return float(lower_viol + upper_viol)

    def check_cardinality(self, weights: np.ndarray) -> float:
        """Returns cardinality violation count."""
        n_active = np.sum(weights > 1e-6)
        lower_v  = max(0, self.cardinality_min - n_active)
        upper_v  = max(0, n_active - self.cardinality_max)
        return float(lower_v + upper_v)

    def check_turnover(self, weights: np.ndarray) -> float:
        """Returns turnover constraint violation."""
        turnover = 0.5 * np.sum(np.abs(weights - self.prev_weights))
        return max(0.0, turnover - self.turnover_limit)

    def is_feasible(self, weights: np.ndarray) -> bool:
        """Returns True if all constraints are satisfied."""
        budget_ok     = self.check_budget(weights) < 1e-4
        box_ok        = self.check_box(weights) < 1e-4
        cardinality_ok= self.check_cardinality(weights) == 0
        turnover_ok   = self.check_turnover(weights) < 1e-4
        return budget_ok and box_ok and cardinality_ok and turnover_ok

    # ── Repair Operator ───────────────────────────────────────────────────────

    def repair(self, weights: np.ndarray) -> np.ndarray:
        """
        Projects infeasible weights onto feasible space.

        Steps:
            1. Clip weights to [0, weight_max]      → box upper bound
            2. Enforce cardinality (keep top K)      → cardinality max
            3. Activate min assets if too few        → cardinality min
            4. Apply min weight to active assets     → box lower bound
            5. Normalize to sum = 1.0               → budget constraint
        """
        w = weights.copy()

        # Step 1 — Long-only + upper box
        w = np.clip(w, 0.0, self.weight_max)

        # Step 2 — Enforce cardinality max (keep top-K by weight)
        n_active = np.sum(w > 0)
        if n_active > self.cardinality_max:
            sorted_idx = np.argsort(w)
            zero_idx   = sorted_idx[:n_active - self.cardinality_max]
            w[zero_idx] = 0.0

        # Step 3 — Enforce cardinality min
        n_active = np.sum(w > 0)
        if n_active < self.cardinality_min:
            zero_idx = np.where(w == 0.0)[0]
            needed   = self.cardinality_min - n_active
            if len(zero_idx) >= needed:
                chosen   = np.random.choice(zero_idx, size=needed, replace=False)
                w[chosen] = self.weight_min

        # Step 4 — Apply minimum weight to active positions
        active_mask       = w > 0
        w[active_mask]    = np.maximum(w[active_mask], self.weight_min)

        # Step 5 — Normalize (budget constraint)
        total = w.sum()
        if total > 1e-9:
            w = w / total
        else:
            # Fallback: equal weight
            idx = np.random.choice(self.n_assets, self.cardinality_min, replace=False)
            w   = np.zeros(self.n_assets)
            w[idx] = 1.0 / self.cardinality_min

        return w

    # ── Penalty Function ─────────────────────────────────────────────────────

    def penalty(self, weights: np.ndarray) -> float:
        """
        Computes total constraint violation penalty.
        Added to fitness for soft constraint enforcement.

        penalty = lambda * (budget_viol + box_viol + card_viol + turnover_viol)
        """
        budget_v   = self.check_budget(weights)
        box_v      = self.check_box(weights)
        card_v     = self.check_cardinality(weights)
        turnover_v = self.check_turnover(weights)

        total_viol = budget_v + box_v + card_v + turnover_v
        return self.penalty_lambda * total_viol

    def get_violation_report(self, weights: np.ndarray) -> dict:
        """Returns a breakdown of all constraint violations."""
        n_active = int(np.sum(weights > 1e-6))
        turnover = 0.5 * np.sum(np.abs(weights - self.prev_weights))

        return {
            "budget_violation"     : round(self.check_budget(weights), 6),
            "box_violation"        : round(self.check_box(weights), 6),
            "cardinality_violation": round(self.check_cardinality(weights), 2),
            "turnover_violation"   : round(self.check_turnover(weights), 6),
            "n_active_assets"      : n_active,
            "actual_turnover"      : round(turnover, 4),
            "weights_sum"          : round(float(weights.sum()), 6),
            "is_feasible"          : self.is_feasible(weights)
        }

    def update_prev_weights(self, weights: np.ndarray):
        """Call after each rebalance to update turnover reference."""
        self.prev_weights = weights.copy()
