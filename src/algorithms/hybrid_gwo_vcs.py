import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')

from algorithms.gwo import GreyWolfOptimizer
from algorithms.vcs import VirusColonySearch


class HybridGWO_VCS:

    def __init__(
        self,
        dim: int = 20,
        gwo_wolves: int = 30,
        vcs_viruses: int = 20,
        gwo_iter: int = 60,
        vcs_iter: int = 40,
        lb: float = 0.0,
        ub: float = 1.0,
        space_reducer=None
    ):
        self.dim           = dim
        self.lb            = lb
        self.ub            = ub
        self.space_reducer = space_reducer

        self.gwo = GreyWolfOptimizer(
            n_wolves = gwo_wolves,
            max_iter = gwo_iter,
            dim      = dim,
            lb       = lb,
            ub       = ub
        )

        self.vcs = VirusColonySearch(
            n_viruses = vcs_viruses,
            max_iter  = vcs_iter,
            dim       = dim,
            lb        = lb,
            ub        = ub
        )

        self.best_pos         = None
        self.best_score       = float("inf")
        self.gwo_convergence  = []
        self.vcs_convergence  = []
        self.full_convergence = []

    def _apply_space_reduction(self, fitness_fn):
        if self.space_reducer is None:
            return fitness_fn, list(range(self.dim))

        promising_indices = self.space_reducer.get_promising_indices()
        print(f"[Hybrid] ML reduced search space: "
              f"{self.dim} to {len(promising_indices)} assets")

        def reduced_fitness(weights):
            full_weights = np.zeros(self.dim)
            full_weights[promising_indices] = weights[:len(promising_indices)]
            return fitness_fn(full_weights)

        return reduced_fitness, promising_indices

    def optimize(self, fitness_fn, verbose=True):
        print("=" * 55)
        print("   HYBRID GWO-VCS PORTFOLIO OPTIMIZER")
        print("=" * 55)

        # Phase 1 — GWO
        print("\n[Phase 1] GWO - Global Exploration")
        print("-" * 40)

        gwo_best_pos, gwo_best_score, self.gwo_convergence = \
            self.gwo.optimize(fitness_fn)

        print(f"\n[Phase 1 Done] GWO Best Fitness : {gwo_best_score:.6f}")

        # Phase 2 — ML Space Reduction
        print("\n[Phase 2] ML Search Space Reduction")
        print("-" * 40)

        reduced_fitness_fn, promising_idx = self._apply_space_reduction(fitness_fn)

        if self.space_reducer is None:
            print("[Phase 2] No ML reducer attached - using full space.")

        # Phase 3 — VCS
        print("\n[Phase 3] VCS - Local Exploitation")
        print("-" * 40)

        top_k = min(5, self.gwo.n_wolves)
        gwo_scores = np.array([
            fitness_fn(self.gwo.positions[i]) for i in range(self.gwo.n_wolves)
        ])
        top_indices    = np.argsort(gwo_scores)[:top_k]
        seed_positions = self.gwo.positions[top_indices]

        self.vcs.set_initial_population(seed_positions)
        print(f"[Hybrid] Seeded VCS with top {top_k} GWO solutions.")

        vcs_best_pos, vcs_best_score, self.vcs_convergence = \
            self.vcs.optimize(reduced_fitness_fn)

        print(f"\n[Phase 3 Done] VCS Best Fitness : {vcs_best_score:.6f}")

        # Combine results
        self.full_convergence = self.gwo_convergence + self.vcs_convergence

        if vcs_best_score < gwo_best_score:
            self.best_pos   = vcs_best_pos
            self.best_score = vcs_best_score
            winner          = "VCS"
        else:
            self.best_pos   = gwo_best_pos
            self.best_score = gwo_best_score
            winner          = "GWO"

        print("\n" + "=" * 55)
        print(f"   OPTIMIZATION COMPLETE")
        print(f"   Winner       : {winner}")
        print(f"   Best Fitness : {self.best_score:.6f}")
        print("=" * 55)

        return self.best_pos, self.best_score, self.full_convergence

    def get_best(self):
        return self.best_pos.copy(), self.best_score

    def get_convergence_summary(self):
        return {
            "gwo_best"        : min(self.gwo_convergence) if self.gwo_convergence else None,
            "vcs_best"        : min(self.vcs_convergence) if self.vcs_convergence else None,
            "overall_best"    : self.best_score,
            "gwo_iterations"  : len(self.gwo_convergence),
            "vcs_iterations"  : len(self.vcs_convergence),
            "total_iterations": len(self.full_convergence)
        }
