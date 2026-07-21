import numpy as np
import pandas as pd
import sys
import os

# ── Path setup s.....................................................................................................................................................................................................................lo all src modules are importable ──────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Import all modules ────────────────────────────────────────────────────────
from data_pipeline.data_loader      import download_stock_data, load_stock_data, \
                                           extract_close_prices, DEFAULT_TICKERS
from data_pipeline.preprocessor     import compute_returns, compute_covariance_matrix, \
                                           compute_expected_returns, build_all_features
from data_pipeline.portfolio_env    import PortfolioEnvironment

from algorithms.gwo                 import GreyWolfOptimizer
from algorithms.vcs                 import VirusColonySearch
from algorithms.hybrid_gwo_vcs      import HybridGWO_VCS

from ml_predictor.feature_extractor import FeatureExtractor
from ml_predictor.region_predictor  import RegionPredictor
from ml_predictor.search_space_reducer import SearchSpaceReducer, build_reducer

from portfolio.objective            import PortfolioObjective
from portfolio.constraints          import PortfolioConstraints
from portfolio.rebalancer           import PortfolioRebalancer


# ══════════════════════════════════════════════════════════════════════════════
#   CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Data settings
    "tickers"          : DEFAULT_TICKERS,
    "start_date"       : "2018-01-01",
    "end_date"         : "2024-12-31",
    "download_fresh"   : False,          # True = re-download from Yahoo Finance

    # Portfolio constraints
    "cardinality_min"  : 3,
    "cardinality_max"  : 10,
    "weight_min"       : 0.01,
    "weight_max"       : 0.40,
    "turnover_limit"   : 0.30,
    "risk_free_rate"   : 0.04,
    "transaction_cost" : 0.001,
    "objective_type"   : "sharpe",       # "sharpe", "sortino", "min_variance"

    # ML predictor settings
    "use_ml_reducer"   : True,           # False = skip ML, use full asset space
    "ml_model_type"    : "rf",           # "rf", "mlp", "gb"
    "ml_top_k"         : 10,             # Keep top-K assets after ML scoring
    "ml_train_new"     : True,           # False = load saved model

    # Optimizer settings
    "gwo_wolves"       : 30,
    "vcs_viruses"      : 20,
    "gwo_iter"         : 60,
    "vcs_iter"         : 40,

    # Rebalancing
    "rebalance_freq"   : "monthly",      # "monthly" or "quarterly"

    # Paths
    "raw_data_path"    : "data/raw/stock_prices.csv",
    "returns_path"     : "data/processed/returns.csv",
    "cov_path"         : "data/processed/covariance_matrix.csv",
    "features_path"    : "data/processed/features.csv",
}


# ══════════════════════════════════════════════════════════════════════════════
#   STEP 1 — DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_data_pipeline(cfg):
    print("\n" + "═" * 60)
    print("  STEP 1 — DATA PIPELINE")
    print("═" * 60)

    # Download or load raw data
    if cfg["download_fresh"] or not os.path.exists(cfg["raw_data_path"]):
        raw_df = download_stock_data(
            tickers   = cfg["tickers"],
            start     = cfg["start_date"],
            end       = cfg["end_date"],
            save_path = cfg["raw_data_path"]
        )
    else:
        raw_df = load_stock_data(cfg["raw_data_path"])

    # Extract close prices
    close_df = extract_close_prices(raw_df, cfg["tickers"])

    # Compute returns
    if not os.path.exists(cfg["returns_path"]):
        returns = compute_returns(close_df, method="log",
                                  save_path=cfg["returns_path"])
    else:
        returns = pd.read_csv(cfg["returns_path"],
                              index_col="Date", parse_dates=True)
        print(f"[Main] Loaded returns from {cfg['returns_path']}")

    # Compute covariance
    if not os.path.exists(cfg["cov_path"]):
        cov = compute_covariance_matrix(returns, save_path=cfg["cov_path"])
    else:
        cov = pd.read_csv(cfg["cov_path"], index_col=0)
        print(f"[Main] Loaded covariance from {cfg['cov_path']}")

    # Compute expected returns
    exp_ret = compute_expected_returns(returns)

    # Build ML features
    if not os.path.exists(cfg["features_path"]):
        build_all_features(raw_df, cfg["tickers"],
                           save_path=cfg["features_path"])

    print(f"\n[Main] Data pipeline complete.")
    print(f"  Assets   : {len(exp_ret)}")
    print(f"  Date range: {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"  Trading days: {len(returns)}")

    return raw_df, returns, cov, exp_ret


# ══════════════════════════════════════════════════════════════════════════════
#   STEP 2 — ML SEARCH SPACE REDUCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_ml_reducer(cfg):
    print("\n" + "═" * 60)
    print("  STEP 2 — ML SEARCH SPACE REDUCTION")
    print("═" * 60)

    if not cfg["use_ml_reducer"]:
        print("[Main] ML reducer disabled — using full asset space.")
        return None

    reducer = build_reducer(
        features_path = cfg["features_path"],
        returns_path  = cfg["returns_path"],
        model_type    = cfg["ml_model_type"],
        top_k         = cfg["ml_top_k"],
        train_new     = cfg["ml_train_new"]
    )

    # Run scoring and get promising asset indices
    reducer.run()

    # Print scores table
    scores_df = reducer.get_scores_dataframe()
    print("\n[Main] Asset Scores:")
    print(scores_df.to_string(index=False))

    return reducer


# ══════════════════════════════════════════════════════════════════════════════
#   STEP 3 — PORTFOLIO ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════════════════

def build_environment(returns, cov, exp_ret, cfg):
    print("\n" + "═" * 60)
    print("  STEP 3 — PORTFOLIO ENVIRONMENT SETUP")
    print("═" * 60)

    env = PortfolioEnvironment(
        returns          = returns,
        covariance       = cov,
        expected_returns = exp_ret,
        cardinality_min  = cfg["cardinality_min"],
        cardinality_max  = cfg["cardinality_max"],
        weight_min       = cfg["weight_min"],
        weight_max       = cfg["weight_max"],
        turnover_limit   = cfg["turnover_limit"],
        risk_free_rate   = cfg["risk_free_rate"],
        transaction_cost = cfg["transaction_cost"]
    )

    print(f"[Main] Environment ready.")
    print(f"  Objective     : {cfg['objective_type']}")
    print(f"  Cardinality   : {cfg['cardinality_min']} - {cfg['cardinality_max']} assets")
    print(f"  Weight bounds : {cfg['weight_min']} - {cfg['weight_max']}")
    print(f"  Turnover limit: {cfg['turnover_limit']}")

    return env


# ══════════════════════════════════════════════════════════════════════════════
#   STEP 4 — RUN HYBRID GWO-VCS OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

def run_optimizer(env, reducer, cfg):
    print("\n" + "═" * 60)
    print("  STEP 4 — HYBRID GWO-VCS OPTIMIZER")
    print("═" * 60)

    n_assets = env.n_assets

    # ── Build hybrid optimizer ─────────────────────────────────────────────
    hybrid = HybridGWO_VCS(
        dim          = n_assets,
        gwo_wolves   = cfg["gwo_wolves"],
        vcs_viruses  = cfg["vcs_viruses"],
        gwo_iter     = cfg["gwo_iter"],
        vcs_iter     = cfg["vcs_iter"],
        lb           = 0.0,
        ub           = 1.0,
        space_reducer= reducer         # Attach ML reducer (can be None)
    )

    # ── Define fitness function ────────────────────────────────────────────
    def fitness_fn(weights):
        return env.fitness(
            weights,
            apply_repair           = True,
            apply_turnover_penalty = True
        )

    # ── Run optimization ───────────────────────────────────────────────────
    best_weights, best_score, convergence = hybrid.optimize(fitness_fn)

    # ── Repair final weights ───────────────────────────────────────────────
    best_weights = env.repair_weights(best_weights)

    # ── Print convergence summary ──────────────────────────────────────────
    summary = hybrid.get_convergence_summary()
    print("\n[Main] Convergence Summary:")
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")

    return hybrid, best_weights, best_score, convergence


# ══════════════════════════════════════════════════════════════════════════════
#   STEP 5 — PORTFOLIO REBALANCING
# ══════════════════════════════════════════════════════════════════════════════

def run_rebalancing(returns, cov, exp_ret, best_weights, cfg):
    print("\n" + "═" * 60)
    print("  STEP 5 — PORTFOLIO REBALANCING")
    print("═" * 60)

    asset_names = list(exp_ret.index)

    # Build objective and constraints
    objective = PortfolioObjective(
        expected_returns = exp_ret.values,
        covariance       = cov.values,
        returns_df       = returns,
        risk_free_rate   = cfg["risk_free_rate"],
        objective_type   = cfg["objective_type"]
    )

    constraints = PortfolioConstraints(
        n_assets        = len(asset_names),
        cardinality_min = cfg["cardinality_min"],
        cardinality_max = cfg["cardinality_max"],
        weight_min      = cfg["weight_min"],
        weight_max      = cfg["weight_max"],
        turnover_limit  = cfg["turnover_limit"]
    )

    rebalancer = PortfolioRebalancer(
        asset_names      = asset_names,
        objective        = objective,
        constraints      = constraints,
        transaction_cost = cfg["transaction_cost"],
        rebalance_freq   = cfg["rebalance_freq"]
    )

    # ── Simulate rebalancing over date index ───────────────────────────────
    rebalance_dates = rebalancer.get_rebalance_dates(returns.index)

    print(f"[Main] Rebalancing {len(rebalance_dates)} periods "
          f"({cfg['rebalance_freq']})...")

    for date in rebalance_dates:
        # Add small noise to weights to simulate re-optimization per period
        noisy_weights = best_weights + np.random.normal(0, 0.02, len(best_weights))
        noisy_weights = np.clip(noisy_weights, 0, 1)

        event = rebalancer.rebalance(
            new_weights = noisy_weights,
            date        = str(date.date())
        )

        # Simulate returns for the period
        period_mask    = returns.index >= date
        period_returns = returns[period_mask].iloc[:21]   # ~1 month
        if not period_returns.empty:
            rebalancer.simulate_period_return(period_returns)

    # ── Print rebalancing summary ──────────────────────────────────────────
    rebalancer.print_summary()

    # ── Save history ───────────────────────────────────────────────────────
    os.makedirs("results/portfolios", exist_ok=True)
    history_df = rebalancer.get_rebalance_history()
    history_df.to_csv("results/portfolios/rebalance_history.csv", index=False)
    print("\n[Main] Rebalance history saved → results/portfolios/rebalance_history.csv")

    return rebalancer


# ══════════════════════════════════════════════════════════════════════════════
#   STEP 6 — FINAL RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def print_final_results(env, best_weights):
    print("\n" + "═" * 60)
    print("  STEP 6 — FINAL OPTIMAL PORTFOLIO")
    print("═" * 60)

    summary = env.get_portfolio_summary(best_weights)

    print(f"\n  {'Metric':<25} {'Value'}")
    print(f"  {'-'*40}")
    print(f"  {'Number of Assets':<25} {summary['n_assets']}")
    print(f"  {'Expected Return':<25} {summary['expected_return']}%")
    print(f"  {'Volatility':<25} {summary['volatility']}%")
    print(f"  {'Sharpe Ratio':<25} {summary['sharpe_ratio']}")
    print(f"  {'Turnover':<25} {summary['turnover']}")
    print(f"  {'Weights Sum':<25} {summary['weights_sum']}")

    print(f"\n  {'Asset':<10} {'Weight':>10}")
    print(f"  {'-'*22}")
    for asset, weight in sorted(
        summary["assets"].items(), key=lambda x: -x[1]
    ):
        bar = "█" * int(weight * 30)
        print(f"  {asset:<10} {weight:>8.2%}  {bar}")

    # Save optimal weights
    os.makedirs("results/portfolios", exist_ok=True)
    weights_df = pd.DataFrame({
        "asset" : list(summary["assets"].keys()),
        "weight": list(summary["assets"].values())
    })
    weights_df.to_csv("results/portfolios/optimal_weights.csv", index=False)
    print("\n[Main] Optimal weights saved → results/portfolios/optimal_weights.csv")


# ══════════════════════════════════════════════════════════════════════════════
#   MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "═" * 60)
    print("   LEARNING-ASSISTED SWARM OPTIMIZATION")
    print("   Portfolio Rebalancing — GWO + VCS + ML")
    print("═" * 60)

    # Step 1 — Data
    raw_df, returns, cov, exp_ret = run_data_pipeline(CONFIG)

    # Step 2 — ML reducer
    reducer = run_ml_reducer(CONFIG)

    # Step 3 — Environment
    env = build_environment(returns, cov, exp_ret, CONFIG)

    # Step 4 — Optimize
    hybrid, best_weights, best_score, convergence = run_optimizer(
        env, reducer, CONFIG
    )

    # Step 5 — Rebalance
    rebalancer = run_rebalancing(returns, cov, exp_ret, best_weights, CONFIG)

    # Step 6 — Final results
    print_final_results(env, best_weights)

    print("\n✅ All done! Check results/ folder for outputs.")
