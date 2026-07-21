import numpy as np


class GreyWolfOptimizer:
    """
    Grey Wolf Optimizer (GWO)
    ─────────────────────────
    Mimics the leadership hierarchy and hunting strategy of grey wolves.

    Hierarchy:
        Alpha (α) → Best solution found
        Beta  (β) → Second best
        Delta (δ) → Third best
        Omega (ω) → Rest of the pack (search agents)

    Role in Hybrid:
        GWO handles GLOBAL EXPLORATION —
        it scans the full portfolio weight space broadly.
    """

    def __init__(
        self,
        n_wolves: int = 30,
        max_iter: int = 100,
        dim: int = 20,          # Number of assets
        lb: float = 0.0,        # Lower bound of weights
        ub: float = 1.0         # Upper bound of weights
    ):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.dim      = dim
        self.lb       = lb
        self.ub       = ub

        # ── Wolf positions (portfolio weights) ────────────────────────────────
        self.positions = np.random.uniform(lb, ub, (n_wolves, dim))

        # ── Alpha, Beta, Delta placeholders ───────────────────────────────────
        self.alpha_pos   = np.zeros(dim)
        self.alpha_score = float("inf")

        self.beta_pos    = np.zeros(dim)
        self.beta_score  = float("inf")

        self.delta_pos   = np.zeros(dim)
        self.delta_score = float("inf")

        # ── Convergence history (for plotting later) ──────────────────────────
        self.convergence_curve = []

    def _update_leaders(self, fitness_fn):
        """
        Evaluates all wolves and updates Alpha, Beta, Delta.
        Alpha  = best fitness
        Beta   = second best
        Delta  = third best
        """
        for i in range(self.n_wolves):
            score = fitness_fn(self.positions[i].copy())

            if score < self.alpha_score:
                self.alpha_score = score
                self.alpha_pos   = self.positions[i].copy()

            elif score < self.beta_score:
                self.beta_score = score
                self.beta_pos   = self.positions[i].copy()

            elif score < self.delta_score:
                self.delta_score = score
                self.delta_pos   = self.positions[i].copy()

    def _update_positions(self, iteration):
        """
        Updates each wolf's position based on Alpha, Beta, Delta guidance.

        Key formula:
            a  linearly decreases from 2 → 0 over iterations
            A  = 2 * a * r1 - a   (controls step size)
            C  = 2 * r2           (controls influence weight)

            D_alpha = |C1 * alpha_pos - position|
            X1      = alpha_pos - A1 * D_alpha

            Same for beta → X2, delta → X3

            new_position = (X1 + X2 + X3) / 3
        """
        # a decreases linearly 2 → 0
        a = 2 - iteration * (2 / self.max_iter)

        for i in range(self.n_wolves):
            pos = self.positions[i]

            # ── Alpha influence ────────────────────────────────────────────
            r1, r2  = np.random.rand(self.dim), np.random.rand(self.dim)
            A1      = 2 * a * r1 - a
            C1      = 2 * r2
            D_alpha = np.abs(C1 * self.alpha_pos - pos)
            X1      = self.alpha_pos - A1 * D_alpha

            # ── Beta influence ─────────────────────────────────────────────
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            A2     = 2 * a * r1 - a
            C2     = 2 * r2
            D_beta = np.abs(C2 * self.beta_pos - pos)
            X2     = self.beta_pos - A2 * D_beta

            # ── Delta influence ────────────────────────────────────────────
            r1, r2  = np.random.rand(self.dim), np.random.rand(self.dim)
            A3      = 2 * a * r1 - a
            C3      = 2 * r2
            D_delta = np.abs(C3 * self.delta_pos - pos)
            X3      = self.delta_pos - A3 * D_delta

            # ── New position = average of all three influences ─────────────
            new_pos = (X1 + X2 + X3) / 3.0

            # ── Clip to valid bounds ───────────────────────────────────────
            self.positions[i] = np.clip(new_pos, self.lb, self.ub)

    def optimize(self, fitness_fn):
        """
        Main GWO loop.

        Args:
            fitness_fn: callable — takes weight array, returns float (lower = better)

        Returns:
            best_position : np.ndarray — best portfolio weights found
            best_score    : float      — best fitness value
            convergence   : list       — fitness per iteration
        """
        print("[GWO] Starting optimization...")

        for iteration in range(self.max_iter):
            # Step 1 — Evaluate all wolves, update leaders
            self._update_leaders(fitness_fn)

            # Step 2 — Move all wolves toward leaders
            self._update_positions(iteration)

            # Step 3 — Record best score this iteration
            self.convergence_curve.append(self.alpha_score)

            if (iteration + 1) % 10 == 0:
                print(f"  [GWO] Iter {iteration+1:4d}/{self.max_iter} "
                      f"| Best Fitness: {self.alpha_score:.6f}")

        print(f"[GWO] Done. Best Fitness: {self.alpha_score:.6f}")
        return self.alpha_pos, self.alpha_score, self.convergence_curve

    def get_best(self):
        return self.alpha_pos.copy(), self.alpha_score
