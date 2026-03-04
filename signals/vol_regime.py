"""
JARVIS V2 - Signal 4: Volatility Regime Detection
====================================================
Inspired by: D.E. Shaw Oculus Strategy / Renaissance Technologies HMMs

HOW THIS WORKS (for non-coders):
- Markets aren't always the same. Sometimes they're calm, sometimes volatile,
  sometimes in full-blown crisis mode.
- This signal uses a "Hidden Markov Model" (HMM) to figure out WHICH MODE
  the market is currently in.
- Think of it like a weather forecaster: "We are currently in a STORM regime"
  → take shelter (buy bonds, gold) or "We are in SUNSHINE regime" → go outside
  (buy stocks, growth assets)

THE THREE REGIMES:
- CALM (State 0):     Low VIX, tight credit spreads. Buy stocks, growth ETFs.
- TRANSITION (State 1): Increasing uncertainty. Reduce risk, diversify.
- CRISIS (State 2):    High VIX, wide credit spreads. Buy bonds, gold, cash.

WHY IT WORKS:
- Different strategies work in different regimes
- Momentum works great in calm markets but fails in crises
- Mean reversion works great in transitions but fails in trends
- By knowing which regime we're in, we can adjust which signals to trust
"""

import pandas as pd
import numpy as np
from loguru import logger

from config.settings import HMM_N_STATES


# Regime labels for human readability
REGIME_NAMES = {0: "CALM", 1: "TRANSITION", 2: "CRISIS"}


def compute_regime_signal(
    macro_features: pd.DataFrame,
    prices: pd.DataFrame
) -> dict:
    """
    Detect the current market regime and generate ETF-level tilt signals.

    Parameters:
        macro_features: DataFrame with VIX, credit spreads, yield curve
        prices: DataFrame with date index, ticker columns, adj_close values

    Returns:
        dict with:
            "regime_probabilities": DataFrame (date × [calm, transition, crisis])
            "current_regime": str ("CALM", "TRANSITION", or "CRISIS")
            "regime_history": Series of regime labels
            "etf_signals": DataFrame (date × ticker) with regime-adjusted scores
    """
    logger.info("Computing volatility regime detection")

    # ── Build the observation matrix for the HMM ──
    observations = build_regime_observations(macro_features, prices)

    if observations.empty or len(observations) < 252:
        logger.warning("Insufficient data for regime detection, defaulting to CALM")
        return _default_regime_result(prices)

    # ── Fit the Hidden Markov Model ──
    try:
        regime_probs, regime_labels = fit_hmm(observations)
    except Exception as e:
        logger.error(f"HMM fitting failed: {e}. Using fallback regime detection.")
        regime_probs, regime_labels = fallback_regime_detection(observations)

    # ── Identify which state is which ──
    # The HMM doesn't label states — we need to figure out which state
    # corresponds to calm, transition, and crisis based on average VIX in each
    regime_probs, regime_labels = label_regimes(regime_probs, regime_labels, observations)

    # ── Generate ETF-level signals based on regime ──
    etf_signals = generate_regime_etf_signals(regime_probs, prices)

    # ── Current regime ──
    current_probs = regime_probs.iloc[-1]
    current_regime = REGIME_NAMES[current_probs.values.argmax()]

    logger.info(f"Current regime: {current_regime} | "
                f"Probabilities: Calm={current_probs.iloc[0]:.2f}, "
                f"Transition={current_probs.iloc[1]:.2f}, "
                f"Crisis={current_probs.iloc[2]:.2f}")

    return {
        "regime_probabilities": regime_probs,
        "current_regime": current_regime,
        "regime_history": regime_labels,
        "etf_signals": etf_signals,
    }


def build_regime_observations(macro_features: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix that the HMM uses to detect regimes.

    Features:
    1. VIX level (fear gauge)
    2. VIX 5-day change (is fear rising or falling?)
    3. Credit spread (are investors demanding more compensation for risk?)
    4. Yield curve slope (inverted = recession warning)
    5. SPY realized volatility (actual market choppiness)
    6. Market breadth proxy (are stocks moving together or independently?)
    """
    obs = pd.DataFrame()

    # VIX features
    if not macro_features.empty:
        if "vix" in macro_features.columns:
            obs["vix"] = macro_features["vix"]
        if "vix_change_5d" in macro_features.columns:
            obs["vix_change"] = macro_features["vix_change_5d"]
        if "credit_spread" in macro_features.columns:
            obs["credit_spread"] = macro_features["credit_spread"]
        if "yield_curve" in macro_features.columns:
            obs["yield_curve"] = macro_features["yield_curve"]

    # SPY realized volatility
    if "SPY" in prices.columns:
        spy_returns = prices["SPY"].pct_change()
        obs["spy_vol"] = spy_returns.rolling(21).std() * np.sqrt(252)

    # Market breadth: average pairwise correlation of equity ETFs
    equity_tickers = [t for t in ["SPY", "QQQ", "IWM", "EFA", "EEM"] if t in prices.columns]
    if len(equity_tickers) >= 3:
        equity_returns = prices[equity_tickers].pct_change()
        rolling_corr = equity_returns.rolling(63).corr()
        # Average correlation across all pairs for each date
        avg_corr = []
        for date in equity_returns.index[63:]:
            try:
                corr_matrix = rolling_corr.loc[date]
                if isinstance(corr_matrix, pd.DataFrame):
                    # Get upper triangle (exclude diagonal)
                    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                    avg = corr_matrix.values[mask].mean()
                    avg_corr.append({"date": date, "avg_corr": avg})
            except Exception:
                continue

        if avg_corr:
            corr_df = pd.DataFrame(avg_corr).set_index("date")
            obs["market_breadth"] = corr_df["avg_corr"]

    # Clean up
    obs = obs.ffill().dropna()

    # Standardize each column (HMM works better with standardized data)
    for col in obs.columns:
        mean = obs[col].mean()
        std = obs[col].std()
        if std > 0:
            obs[col] = (obs[col] - mean) / std

    return obs


def fit_hmm(observations: pd.DataFrame) -> tuple:
    """
    Fit a Gaussian Hidden Markov Model to the observation data.

    Returns:
        regime_probs: DataFrame with columns [state_0, state_1, state_2]
        regime_labels: Series with the most likely state for each day
    """
    from hmmlearn.hmm import GaussianHMM

    # Prepare data
    X = observations.values

    # Fit the HMM
    model = GaussianHMM(
        n_components=HMM_N_STATES,
        covariance_type="full",
        n_iter=200,
        random_state=42,
        tol=0.01,
    )
    model.fit(X)

    # Get state probabilities for each day
    probs = model.predict_proba(X)
    labels = model.predict(X)

    # Convert to DataFrames
    regime_probs = pd.DataFrame(
        probs,
        index=observations.index,
        columns=[f"state_{i}" for i in range(HMM_N_STATES)]
    )
    regime_labels = pd.Series(labels, index=observations.index)

    return regime_probs, regime_labels


def fallback_regime_detection(observations: pd.DataFrame) -> tuple:
    """
    Simple rule-based regime detection as fallback if HMM fails.
    Uses VIX levels directly:
    - VIX < 0 (below average after standardization) → Calm
    - VIX 0-1 → Transition
    - VIX > 1 (more than 1 std above average) → Crisis
    """
    logger.info("Using fallback rule-based regime detection")

    if "vix" in observations.columns:
        vix = observations["vix"]
    elif "spy_vol" in observations.columns:
        vix = observations["spy_vol"]
    else:
        # Default to calm if no data
        vix = pd.Series(0.0, index=observations.index)

    labels = pd.Series(0, index=observations.index)  # Default: calm
    labels[vix > 0] = 1     # Transition
    labels[vix > 1.0] = 2   # Crisis

    # Create probability-like scores
    probs = pd.DataFrame(0.0, index=observations.index, columns=["state_0", "state_1", "state_2"])
    probs.loc[labels == 0, "state_0"] = 0.8
    probs.loc[labels == 0, "state_1"] = 0.15
    probs.loc[labels == 0, "state_2"] = 0.05
    probs.loc[labels == 1, "state_0"] = 0.2
    probs.loc[labels == 1, "state_1"] = 0.6
    probs.loc[labels == 1, "state_2"] = 0.2
    probs.loc[labels == 2, "state_0"] = 0.05
    probs.loc[labels == 2, "state_1"] = 0.2
    probs.loc[labels == 2, "state_2"] = 0.75

    return probs, labels


def label_regimes(regime_probs: pd.DataFrame, regime_labels: pd.Series,
                  observations: pd.DataFrame) -> tuple:
    """
    The HMM doesn't know which state is "calm" vs "crisis" — we need to
    figure that out by looking at which state has the highest average VIX.

    State with LOWEST avg VIX → Calm (0)
    State with HIGHEST avg VIX → Crisis (2)
    Middle → Transition (1)
    """
    if "vix" not in observations.columns and "spy_vol" not in observations.columns:
        return regime_probs, regime_labels

    indicator = observations["vix"] if "vix" in observations.columns else observations["spy_vol"]

    # Average VIX in each state
    state_means = {}
    for state in range(HMM_N_STATES):
        mask = regime_labels == state
        if mask.sum() > 0:
            state_means[state] = indicator[mask].mean()
        else:
            state_means[state] = 0

    # Sort states by VIX: lowest VIX = Calm, highest = Crisis
    sorted_states = sorted(state_means, key=state_means.get)
    state_mapping = {sorted_states[i]: i for i in range(len(sorted_states))}

    # Remap
    new_labels = regime_labels.map(state_mapping)
    new_probs = regime_probs.copy()
    new_probs.columns = [f"state_{state_mapping.get(i, i)}" for i in range(HMM_N_STATES)]
    new_probs = new_probs[["state_0", "state_1", "state_2"]]

    return new_probs, new_labels


def generate_regime_etf_signals(regime_probs: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Translate regime probabilities into ETF-level trading signals.

    CALM regime → tilt toward: equities, growth, high-yield (risk-on)
    CRISIS regime → tilt toward: bonds, gold, low-volatility (risk-off)
    TRANSITION → balanced, slight defensive lean
    """
    from config.universe import get_asset_class_map

    asset_map = get_asset_class_map()

    # Define how much each asset class benefits from each regime
    # Positive = benefits, Negative = hurts
    regime_tilts = {
        "equity":       {"calm":  0.8, "transition":  0.0, "crisis": -0.8},
        "fixed_income": {"calm": -0.3, "transition":  0.3, "crisis":  0.7},
        "commodity":    {"calm":  0.2, "transition":  0.2, "crisis":  0.4},
        "real_estate":  {"calm":  0.5, "transition": -0.2, "crisis": -0.5},
        "volatility":   {"calm": -0.5, "transition":  0.3, "crisis":  0.8},
        "inverse":      {"calm": -0.8, "transition":  0.0, "crisis":  0.6},
        "currency":     {"calm":  0.0, "transition":  0.2, "crisis":  0.3},
    }

    signals = pd.DataFrame(0.0, index=regime_probs.index, columns=prices.columns)

    for ticker in prices.columns:
        asset_class = asset_map.get(ticker, "equity")
        tilts = regime_tilts.get(asset_class, {"calm": 0, "transition": 0, "crisis": 0})

        # Weighted sum: probability of each regime × tilt for that regime
        signal = (
            regime_probs["state_0"] * tilts["calm"] +
            regime_probs["state_1"] * tilts["transition"] +
            regime_probs["state_2"] * tilts["crisis"]
        )
        signals[ticker] = signal

    return signals


def get_latest_regime(macro_features: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """
    Get the current regime and ETF signals for today's trading decision.
    """
    result = compute_regime_signal(macro_features, prices)
    return {
        "regime": result["current_regime"],
        "etf_signals": result["etf_signals"].iloc[-1] if not result["etf_signals"].empty else pd.Series(dtype=float),
    }


def _default_regime_result(prices: pd.DataFrame) -> dict:
    """Return a default result when regime detection can't run."""
    return {
        "regime_probabilities": pd.DataFrame(),
        "current_regime": "CALM",
        "regime_history": pd.Series(dtype=int),
        "etf_signals": pd.DataFrame(0.0, index=prices.index[-10:], columns=prices.columns),
    }
