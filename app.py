import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_pipeline.data_loader         import load_stock_data, extract_close_prices, DEFAULT_TICKERS
from data_pipeline.preprocessor        import compute_returns, compute_covariance_matrix, compute_expected_returns
from data_pipeline.portfolio_env       import PortfolioEnvironment
from algorithms.hybrid_gwo_vcs         import HybridGWO_VCS
from ml_predictor.feature_extractor    import FeatureExtractor
from ml_predictor.region_predictor     import RegionPredictor
from ml_predictor.search_space_reducer import SearchSpaceReducer, build_reducer
from portfolio.objective               import PortfolioObjective
from portfolio.constraints             import PortfolioConstraints
from portfolio.rebalancer              import PortfolioRebalancer


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title = "Portfolio Optimization — GWO + VCS + ML",
    page_icon  = "📈",
    layout     = "wide"
)

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #313244;
    }
    .winner-badge {
        background: linear-gradient(135deg, #a6e3a1, #94e2d5);
        color: #1e1e2e;
        border-radius: 20px;
        padding: 8px 24px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#   CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_all_data():
    raw_df   = load_stock_data("data/raw/stock_prices.csv")
    close_df = extract_close_prices(raw_df)
    returns  = pd.read_csv(
        "data/processed/returns.csv", index_col="Date", parse_dates=True
    )
    cov      = pd.read_csv(
        "data/processed/covariance_matrix.csv", index_col=0
    )
    exp_ret  = compute_expected_returns(returns)
    return raw_df, close_df, returns, cov, exp_ret


@st.cache_resource(show_spinner=False)
def run_ml_reducer():
    reducer = build_reducer(
        features_path = "data/processed/features.csv",
        returns_path  = "data/processed/returns.csv",
        model_type    = "rf",
        top_k         = 10,
        train_new     = True
    )
    reducer.run()
    return reducer


@st.cache_resource(show_spinner=False)
def run_optimizer(_env, _reducer):
    hybrid = HybridGWO_VCS(
        dim           = _env.n_assets,
        gwo_wolves    = 30,
        vcs_viruses   = 20,
        gwo_iter      = 60,
        vcs_iter      = 40,
        lb            = 0.0,
        ub            = 1.0,
        space_reducer = _reducer
    )

    def fitness_fn(w):
        return _env.fitness(w, apply_repair=True, apply_turnover_penalty=True)

    best_w, best_s, conv = hybrid.optimize(fitness_fn, verbose=False)
    best_w = _env.repair_weights(best_w)
    return hybrid, best_w, best_s, conv


@st.cache_resource(show_spinner=False)
def run_rebalancing(_returns, _cov, _exp_ret, _best_weights):
    asset_names = list(_exp_ret.index)

    objective = PortfolioObjective(
        expected_returns = _exp_ret.values,
        covariance       = _cov.values,
        returns_df       = _returns,
        risk_free_rate   = 0.04,
        objective_type   = "sharpe"
    )
    constraints = PortfolioConstraints(
        n_assets        = len(asset_names),
        cardinality_min = 3,
        cardinality_max = 10,
        weight_min      = 0.01,
        weight_max      = 0.40,
        turnover_limit  = 0.30
    )
    rebalancer = PortfolioRebalancer(
        asset_names      = asset_names,
        objective        = objective,
        constraints      = constraints,
        transaction_cost = 0.001,
        rebalance_freq   = "monthly"
    )

    rebalance_dates = rebalancer.get_rebalance_dates(_returns.index)

    for date in rebalance_dates:
        noisy = _best_weights + np.random.normal(0, 0.02, len(_best_weights))
        noisy = np.clip(noisy, 0, 1)
        rebalancer.rebalance(new_weights=noisy, date=str(date.date()))
        period_mask    = _returns.index >= date
        period_returns = _returns[period_mask].iloc[:21]
        if not period_returns.empty:
            rebalancer.simulate_period_return(period_returns)

    return rebalancer


# ══════════════════════════════════════════════════════════════════════════════
#   SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 Portfolio Optimizer")
    st.markdown("**GWO + VCS + ML Hybrid**")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "🤖 ML Predictor",
            "🐺 Optimizer",
            "📊 Optimal Portfolio",
            "🔁 Rebalancing History"
        ]
    )

    st.divider()
    st.caption("")
    st.caption("")
    st.caption("")


# ══════════════════════════════════════════════════════════════════════════════
#   LOAD DATA — runs once, cached after
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("⏳ Loading market data..."):
    raw_df, close_df, returns, cov, exp_ret = load_all_data()

env = PortfolioEnvironment(
    returns          = returns,
    covariance       = cov,
    expected_returns = exp_ret,
    cardinality_min  = 3,
    cardinality_max  = 10,
    weight_min       = 0.01,
    weight_max       = 0.40,
    turnover_limit   = 0.30,
    risk_free_rate   = 0.04
)

with st.spinner("🤖 Training ML model..."):
    reducer = run_ml_reducer()

with st.spinner("🐺 Running GWO + VCS optimizer..."):
    hybrid, best_weights, best_score, convergence = run_optimizer(env, reducer)

with st.spinner("🔁 Simulating rebalancing..."):
    rebalancer = run_rebalancing(returns, cov, exp_ret, best_weights)

# ── Derived data ──────────────────────────────────────────────────────────────
summary      = env.get_portfolio_summary(best_weights)
conv_summary = hybrid.get_convergence_summary()
history_df   = rebalancer.get_rebalance_history()
scores_df    = reducer.get_scores_dataframe()


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.title("📈 Learning-Assisted Swarm Optimization")
    st.markdown("#### Portfolio Rebalancing using GWO + VCS + ML | MSC AIML Project")
    st.divider()

    # ── Top metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Assets",      f"{env.n_assets}")
    c2.metric("Trading Days",      f"{len(returns):,}")
    c3.metric("Best Sharpe Ratio", f"{-best_score:.4f}")
    c4.metric("Expected Return",   f"{summary['expected_return']}%")
    c5.metric("Volatility",        f"{summary['volatility']}%")

    st.divider()

    # ── Pipeline steps ────────────────────────────────────────────────────────
    st.markdown("### System Pipeline")
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.info("**📥 Step 1 — Data**\n\n"
                "20 S&P 500 Stocks\n\n"
                "2018 – 2024\n\n"
                "1760 Trading Days")
    with p2:
        st.info("**🤖 Step 2 — ML Reducer**\n\n"
                "Random Forest\n\n"
                "340 Features\n\n"
                "Top-10 Assets Selected")
    with p3:
        st.info("**🐺 Step 3 — GWO + VCS**\n\n"
                "60 GWO Iterations\n\n"
                "40 VCS Iterations\n\n"
                "Hybrid Optimization")
    with p4:
        st.success("**📊 Step 4 — Rebalance**\n\n"
                   "Monthly Frequency\n\n"
                   f"{len(history_df)} Periods\n\n"
                   "Sharpe Optimized")

    st.divider()

    # ── Convergence + Winner ──────────────────────────────────────────────────
    st.markdown("### Optimization Overview")
    left, right = st.columns([2, 1])

    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y    = hybrid.gwo_convergence,
            x    = list(range(1, len(hybrid.gwo_convergence) + 1)),
            name = "GWO Phase",
            line = dict(color="#cba6f7", width=2.5)
        ))
        fig.add_trace(go.Scatter(
            y    = hybrid.vcs_convergence,
            x    = list(range(
                       len(hybrid.gwo_convergence) + 1,
                       len(hybrid.gwo_convergence) + len(hybrid.vcs_convergence) + 1
                   )),
            name = "VCS Phase",
            line = dict(color="#94e2d5", width=2.5)
        ))
        fig.add_vline(
            x                = len(hybrid.gwo_convergence) + 0.5,
            line_dash        = "dash",
            line_color       = "#fab387",
            annotation_text  = "GWO → VCS"
        )
        fig.update_layout(
            title       = "Hybrid Convergence Curve",
            xaxis_title = "Iteration",
            yaxis_title = "Fitness (Lower = Better)",
            template    = "plotly_dark",
            height      = 320
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        winner = "GWO" if conv_summary["gwo_best"] < conv_summary["vcs_best"] else "VCS"
        st.markdown("### Result")
        st.markdown(
            f'<div class="winner-badge">🏆 Winner: {winner}</div>',
            unsafe_allow_html=True
        )
        st.write("")
        st.metric("GWO Best Fitness",  f"{conv_summary['gwo_best']:.6f}")
        st.metric("VCS Best Fitness",  f"{conv_summary['vcs_best']:.6f}")
        st.metric("Total Iterations",  conv_summary["total_iterations"])
        st.metric("Rebalance Periods", len(history_df))


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE 2 — ML PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 ML Predictor":
    st.title("🤖 ML Search Space Reducer")
    st.markdown("Random Forest classifier predicts which assets are in **promising regions** of the search space.")
    st.divider()

    # ── Stats ─────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    promising_count = int(scores_df["promising"].sum())
    c1.metric("Total Assets",      env.n_assets)
    c2.metric("Promising Assets",  promising_count)
    c3.metric("Space Reduced By",  f"{(1 - promising_count/env.n_assets)*100:.0f}%")

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Asset Scores Table")

        def color_promising(val):
            return "background-color: #a6e3a122" if val else ""

        st.dataframe(
            scores_df.style
                .format({"score": "{:.4f}"})
                .applymap(color_promising, subset=["promising"]),
            use_container_width=True,
            height=520
        )

    with right:
        st.markdown("### Score Bar Chart")

        bar_colors = ["#a6e3a1" if p else "#f38ba8"
                      for p in scores_df["promising"]]

        fig = go.Figure(go.Bar(
            x            = scores_df["asset"],
            y            = scores_df["score"],
            marker_color = bar_colors,
            text         = scores_df["score"].round(3),
            textposition = "outside"
        ))
        threshold = scores_df[scores_df["promising"]]["score"].min()
        fig.add_hline(
            y                = threshold,
            line_dash        = "dash",
            line_color       = "#fab387",
            annotation_text  = f"Threshold: {threshold:.3f}"
        )
        fig.update_layout(
            title       = "Asset Promising Scores (Green = Selected)",
            template    = "plotly_dark",
            height      = 420,
            xaxis_title = "Asset",
            yaxis_title = "Score",
            showlegend  = False
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Expected return comparison ─────────────────────────────────────────────
    st.markdown("### Expected Annual Returns — Selected vs Not Selected")

    promising_assets    = list(scores_df[scores_df["promising"]]["asset"])
    not_promising       = list(scores_df[~scores_df["promising"]]["asset"])

    exp_df = pd.DataFrame({
        "Asset"  : list(exp_ret.index),
        "Return" : (exp_ret.values * 100).round(2),
        "Status" : [
            "Promising" if a in promising_assets else "Not Selected"
            for a in exp_ret.index
        ]
    }).sort_values("Return", ascending=False)

    fig2 = px.bar(
        exp_df,
        x          = "Asset",
        y          = "Return",
        color      = "Status",
        color_discrete_map = {
            "Promising"    : "#a6e3a1",
            "Not Selected" : "#f38ba8"
        },
        template   = "plotly_dark",
        title      = "Annual Expected Return by Asset",
        labels     = {"Return": "Annual Return (%)"}
    )
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE 3 — OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🐺 Optimizer":
    st.title("🐺 Hybrid GWO + VCS Optimizer")
    st.divider()

    # ── Algorithm cards ───────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("### Grey Wolf Optimizer (GWO)")
        st.markdown("""
- **Role:** Global Exploration (Phase 1)
- **Wolves:** 30 agents
- **Iterations:** 60
- **Mechanism:** Alpha → Beta → Delta hierarchy
- Explores full weight space broadly
- Passes top-5 solutions to VCS
        """)
        st.success(f"**GWO Best Fitness: {conv_summary['gwo_best']:.6f}**")

    with right:
        st.markdown("### Virus Colony Search (VCS)")
        st.markdown("""
- **Role:** Local Exploitation (Phase 2)
- **Viruses:** 20 agents
- **Iterations:** 40
- **Mechanism:** Infect → Replicate → Immunize
- Seeded with GWO's best solutions
- Deep search in promising regions
        """)
        st.info(f"**VCS Best Fitness: {conv_summary['vcs_best']:.6f}**")

    st.divider()

    # ── Full convergence curve ────────────────────────────────────────────────
    st.markdown("### Full Convergence Curve")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y    = hybrid.gwo_convergence,
        x    = list(range(1, len(hybrid.gwo_convergence) + 1)),
        name = "GWO (Phase 1)",
        line = dict(color="#cba6f7", width=3),
        mode = "lines"
    ))
    fig.add_trace(go.Scatter(
        y    = hybrid.vcs_convergence,
        x    = list(range(
                   len(hybrid.gwo_convergence) + 1,
                   len(hybrid.gwo_convergence) + len(hybrid.vcs_convergence) + 1
               )),
        name = "VCS (Phase 2)",
        line = dict(color="#94e2d5", width=3),
        mode = "lines"
    ))
    fig.add_vline(
        x               = len(hybrid.gwo_convergence) + 0.5,
        line_dash       = "dash",
        line_color      = "#fab387",
        annotation_text = "GWO → VCS Handoff"
    )
    fig.update_layout(
        xaxis_title = "Iteration",
        yaxis_title = "Fitness Value (Lower = Better)",
        template    = "plotly_dark",
        height      = 420,
        legend      = dict(x=0.75, y=0.95)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("### Optimization Summary")

    winner = "GWO" if conv_summary["gwo_best"] < conv_summary["vcs_best"] else "VCS"
    st.markdown(
        f'<div class="winner-badge">🏆 Winner: {winner}</div>',
        unsafe_allow_html=True
    )
    st.write("")

    summary_df = pd.DataFrame({
        "Metric" : [
            "GWO Best Fitness", "VCS Best Fitness", "Overall Best",
            "Winner", "GWO Iterations", "VCS Iterations", "Total Iterations"
        ],
        "Value"  : [
            f"{conv_summary['gwo_best']:.6f}",
            f"{conv_summary['vcs_best']:.6f}",
            f"{conv_summary['overall_best']:.6f}",
            winner,
            conv_summary["gwo_iterations"],
            conv_summary["vcs_iterations"],
            conv_summary["total_iterations"]
        ]
    })
    st.dataframe(summary_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE 4 — OPTIMAL PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Optimal Portfolio":
    st.title("📊 Optimal Portfolio")
    st.divider()

    # ── Metrics ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe Ratio",    f"{summary['sharpe_ratio']}")
    c2.metric("Expected Return", f"{summary['expected_return']}%")
    c3.metric("Volatility",      f"{summary['volatility']}%")
    c4.metric("Assets Selected", f"{summary['n_assets']}")

    st.divider()

    assets  = list(summary["assets"].keys())
    weights = list(summary["assets"].values())

    left, right = st.columns(2)

    with left:
        st.markdown("### Portfolio Weights — Pie Chart")
        fig = go.Figure(go.Pie(
            labels      = assets,
            values      = weights,
            hole        = 0.4,
            textinfo    = "label+percent",
            hoverinfo   = "label+value+percent"
        ))
        fig.update_layout(
            template   = "plotly_dark",
            height     = 420,
            showlegend = True
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Portfolio Weights — Bar Chart")
        w_df = pd.DataFrame({
            "Asset"  : assets,
            "Weight" : [w * 100 for w in weights]
        }).sort_values("Weight", ascending=True)

        fig2 = go.Figure(go.Bar(
            x            = w_df["Weight"],
            y            = w_df["Asset"],
            orientation  = "h",
            marker_color = "#cba6f7",
            text         = w_df["Weight"].round(2).astype(str) + "%",
            textposition = "outside"
        ))
        fig2.update_layout(
            template    = "plotly_dark",
            height      = 420,
            xaxis_title = "Weight (%)",
            yaxis_title = "Asset"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Weights table ─────────────────────────────────────────────────────────
    st.markdown("### Detailed Weights Table")

    weights_table = pd.DataFrame({
        "Asset"            : assets,
        "Weight"           : weights,
        "Weight %"         : [f"{w*100:.2f}%" for w in weights],
        "Annual Return %"  : [
            f"{exp_ret[a]*100:.2f}%" if a in exp_ret.index else "N/A"
            for a in assets
        ]
    }).sort_values("Weight", ascending=False).reset_index(drop=True)

    st.dataframe(weights_table, use_container_width=True)

    st.divider()

    # ── Risk-Return position ──────────────────────────────────────────────────
    st.markdown("### Individual Asset Risk-Return")

    asset_returns = exp_ret * 100
    asset_vols    = pd.Series({
        col: returns[col].std() * np.sqrt(252) * 100
        for col in returns.columns
    })

    rr_df = pd.DataFrame({
        "Asset"     : asset_returns.index,
        "Return %"  : asset_returns.values,
        "Vol %"     : asset_vols.values,
        "Selected"  : [a in assets for a in asset_returns.index]
    })

    fig3 = px.scatter(
        rr_df,
        x          = "Vol %",
        y          = "Return %",
        text       = "Asset",
        color      = "Selected",
        color_discrete_map = {True: "#a6e3a1", False: "#f38ba8"},
        template   = "plotly_dark",
        title      = "Risk vs Return (Green = Selected in Portfolio)",
        labels     = {"Vol %": "Volatility (%)", "Return %": "Annual Return (%)"}
    )
    fig3.update_traces(textposition="top center", marker_size=10)
    fig3.update_layout(height=420)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#   PAGE 5 — REBALANCING HISTORY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔁 Rebalancing History":
    st.title("🔁 Monthly Rebalancing History")
    st.divider()

    # ── Summary metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rebalances",  len(history_df))
    c2.metric("Avg Sharpe Ratio",  f"{history_df['sharpe_ratio'].mean():.4f}")
    c3.metric("Avg Return",        f"{history_df['expected_return'].mean():.2f}%")
    c4.metric("Avg Turnover",      f"{history_df['turnover'].mean():.4f}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("### Sharpe Ratio Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x      = history_df["date"],
            y      = history_df["sharpe_ratio"],
            mode   = "lines+markers",
            line   = dict(color="#cba6f7", width=2),
            marker = dict(size=4)
        ))
        fig.add_hline(
            y               = history_df["sharpe_ratio"].mean(),
            line_dash       = "dash",
            line_color      = "#fab387",
            annotation_text = f"Avg: {history_df['sharpe_ratio'].mean():.3f}"
        )
        fig.update_layout(
            template    = "plotly_dark",
            height      = 320,
            xaxis_title = "Date",
            yaxis_title = "Sharpe Ratio"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Portfolio Value Over Time")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x         = list(range(len(rebalancer.value_history))),
            y         = rebalancer.value_history,
            mode      = "lines",
            fill      = "tozeroy",
            line      = dict(color="#a6e3a1", width=2),
            fillcolor = "rgba(166,227,161,0.1)"
        ))
        fig2.update_layout(
            template    = "plotly_dark",
            height      = 320,
            xaxis_title = "Period",
            yaxis_title = "Portfolio Value (Normalized)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Return and volatility over time ───────────────────────────────────────
    st.markdown("### Return & Volatility Over Time")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x    = history_df["date"],
        y    = history_df["expected_return"],
        name = "Expected Return %",
        line = dict(color="#a6e3a1", width=2)
    ))
    fig3.add_trace(go.Scatter(
        x    = history_df["date"],
        y    = history_df["volatility"],
        name = "Volatility %",
        line = dict(color="#f38ba8", width=2)
    ))
    fig3.update_layout(
        template    = "plotly_dark",
        height      = 320,
        xaxis_title = "Date",
        yaxis_title = "Percentage (%)"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── Risk-Return scatter ───────────────────────────────────────────────────
    st.markdown("### Risk-Return Scatter per Period")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x    = history_df["volatility"],
        y    = history_df["expected_return"],
        mode = "markers+text",
        text = history_df["date"].str[:7],
        textposition = "top center",
        marker = dict(
            size       = 8,
            color      = history_df["sharpe_ratio"],
            colorscale = "Viridis",
            showscale  = True,
            colorbar   = dict(title="Sharpe")
        )
    ))
    fig4.update_layout(
        title       = "Risk-Return per Period (color = Sharpe Ratio)",
        template    = "plotly_dark",
        height      = 420,
        xaxis_title = "Volatility (%)",
        yaxis_title = "Expected Return (%)"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── Full table ────────────────────────────────────────────────────────────
    st.markdown("### Full Rebalancing Table")
    display_df = history_df[[
        "date", "n_assets", "sharpe_ratio",
        "expected_return", "volatility",
        "turnover", "transaction_cost"
    ]].copy()
    display_df.columns = [
        "Date", "Assets", "Sharpe",
        "Return %", "Volatility %",
        "Turnover", "Trans. Cost"
    ]
    st.dataframe(display_df, use_container_width=True, height=450)
