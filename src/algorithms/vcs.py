import numpy as np


class VirusColonySearch:
    """
    Virus Colony Search (VCS)
    ──────────────────────────
    Mimics how a virus colony infects host cells, replicates, and evolves
    through immune response pressure.

    Three Operators:
        1. Infection    → Virus spreads to nearby host cells (exploration)
        2. Replication  → Successful viruses replicate with mutation (exploitation)
        3. Immunization → Weak viruses are eliminated, new ones introduced

    Role in Hybrid:
        VCS handles DEEP LOCAL EXPLOITATION —
        it refines solutions found by GWO in promising regions.
    """

    def __init__(
        self,
        n_viruses: int = 20,
        max_iter: int = 100,
        dim: int = 20,
        lb: float = 0.0,
        ub: float = 1.0,
        infection_rate: float = 0.5,    # Controls spread radius
        replication_rate: float = 0.3,  # Fraction of top viruses that replicate
        mutation_rate: float = 0.1      # Mutation strength during replication
    ):
        self.n_viruses        = n_viruses
        self.max_iter         = max_iter
        self.dim              = dim
        self.lb               = lb
        self.ub               = ub
        self.infection_rate   = infection_rate
        self.replication_rate = replication_rate
        self.mutation_rate    = mutation_rate

        # ── Virus population (portfolio weights) ──────────────────────────────
        self.population = np.random.uniform(lb, ub, (n_viruses, dim))
        self.fitness_scores = np.full(n_viruses, float("inf"))

        # ── Best solution ──────────────────────────────────────────────────────
        self.best_pos   = np.zeros(dim)
        self.best_score = float("inf")

        # ── Convergence history ────────────────────────────────────────────────
        self.convergence_curve = []

    def _evaluate_all(self, fitness_fn):
        """Evaluates fitness for all viruses in population."""
        for i in range(self.n_viruses):
            score = fitness_fn(self.population[i].copy())
            self.fitness_scores[i] = score

            if score < self.best_score:
                self.best_score = score
                self.best_pos   = self.population[i].copy()

    def _infect(self):
        """
        Infection Operator:
        Each virus spreads to a new position near a randomly chosen host.
        Simulates virus exploring nearby regions of search space.

        new_virus = current + infection_rate * (random_host - current) + noise
        """
        new_population = self.population.copy()

        for i in range(self.n_viruses):
            # Pick a random host (different from current virus)
            host_idx = np.random.randint(0, self.n_viruses)
            while host_idx == i:
                host_idx = np.random.randint(0, self.n_viruses)

            host  = self.population[host_idx]
            virus = self.population[i]

            # Infection spread toward host + Gaussian noise
            noise      = np.random.normal(0, 0.01, self.dim)
            new_virus  = virus + self.infection_rate * (host - virus) + noise
            new_population[i] = np.clip(new_virus, self.lb, self.ub)

        self.population = new_population

    def _replicate(self, fitness_fn):
        """
        Replication Operator:
        Top-performing viruses replicate with small mutations.
        Replicas replace the worst-performing viruses.

        replica = best_virus + mutation_rate * random_direction
        """
        n_replicate = max(1, int(self.n_viruses * self.replication_rate))

        # Sort viruses by fitness (ascending = better)
        sorted_idx  = np.argsort(self.fitness_scores)
        top_idx     = sorted_idx[:n_replicate]    # Best viruses
        worst_idx   = sorted_idx[-n_replicate:]   # Worst viruses (to replace)

        for rank, (t_idx, w_idx) in enumerate(zip(top_idx, worst_idx)):
            # Replicate top virus with mutation
            mutation  = np.random.normal(0, self.mutation_rate, self.dim)
            replica   = self.population[t_idx] + mutation
            replica   = np.clip(replica, self.lb, self.ub)

            # Replace worst virus with replica
            new_score = fitness_fn(replica.copy())
            self.population[w_idx]     = replica
            self.fitness_scores[w_idx] = new_score

            if new_score < self.best_score:
                self.best_score = new_score
                self.best_pos   = replica.copy()

    def _immunize(self, iteration):
        """
        Immunization Operator:
        Simulates immune response — eliminates very weak viruses
        and introduces fresh random ones to maintain diversity.

        Triggered every 20 iterations to prevent stagnation.
        """
        if iteration % 20 != 0 or iteration == 0:
            return

        n_immune   = max(1, int(self.n_viruses * 0.2))  # Remove bottom 20%
        sorted_idx = np.argsort(self.fitness_scores)
        immune_idx = sorted_idx[-n_immune:]              # Worst viruses

        for idx in immune_idx:
            # Replace with fresh random virus
            self.population[idx]     = np.random.uniform(self.lb, self.ub, self.dim)
            self.fitness_scores[idx] = float("inf")

        print(f"  [VCS]  Immunization at iter {iteration} — "
              f"replaced {n_immune} weak viruses")

    def optimize(self, fitness_fn):
        """
        Main VCS loop.

        Args:
            fitness_fn: callable — takes weight array, returns float (lower = better)

        Returns:
            best_position : np.ndarray — best portfolio weights found
            best_score    : float      — best fitness value
            convergence   : list       — fitness per iteration
        """
        print("[VCS] Starting optimization...")

        # Initial evaluation
        self._evaluate_all(fitness_fn)

        for iteration in range(self.max_iter):
            # Step 1 — Infect (explore nearby regions)
            self._infect()

            # Step 2 — Evaluate after infection
            self._evaluate_all(fitness_fn)

            # Step 3 — Replicate top performers
            self._replicate(fitness_fn)

            # Step 4 — Immunize (remove weak, add fresh diversity)
            self._immunize(iteration)

            # Step 5 — Record best score
            self.convergence_curve.append(self.best_score)

            if (iteration + 1) % 10 == 0:
                print(f"  [VCS] Iter {iteration+1:4d}/{self.max_iter} "
                      f"| Best Fitness: {self.best_score:.6f}")

        print(f"[VCS] Done. Best Fitness: {self.best_score:.6f}")
        return self.best_pos, self.best_score, self.convergence_curve

    def set_initial_population(self, positions: np.ndarray):
        """
        Allows Hybrid to seed VCS with GWO's best solutions
        instead of starting from random positions.
        """
        n = min(len(positions), self.n_viruses)
        self.population[:n] = positions[:n].copy()

    def get_best(self):
        return self.best_pos.copy(), self.best_score
