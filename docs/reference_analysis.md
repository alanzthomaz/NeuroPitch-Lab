# FIFA World Cup 2026 Reference Model Analysis

This document provides a detailed breakdown of the statistical prediction engine from the [world-cup-2026-prediction-model](https://github.com/Hicruben/world-cup-2026-prediction-model) repository.

---

## 1. Core Architecture

The prediction engine consists of three key layers:
1.  **Elo Rating System**: Tracks and dynamically updates team strengths based on game outcomes, match importance (K-factor), and recency.
2.  **Dixon-Coles Poisson Model**: Converts Elo ratings to expected goals ($\lambda, \mu$) and applies a low-scoring draw correction to compute match outcome probabilities (win/draw/loss).
3.  **Monte Carlo Simulator**: Simulates matches sequentially through the tournament bracket, resolving group standings and knockouts over thousands of trials to compute overall tournament odds.

---

## 2. Mathematical Methodology

### Elo Rating System
The expected outcome of a match between team $A$ (rating $R_A$) and team $B$ (rating $R_B$) is computed using a logistic function:
$$E_A = \frac{1}{1 + 10^{(R_B - (R_A + H_A)) / 400}}$$
where $H_A$ is the home advantage bonus.

After a match is played, the Elo ratings are updated:
$$R'_A = R_A + K \cdot (S_A - E_A)$$
where:
*   $S_A$ is the actual score ($1.0$ for a win, $0.5$ for a draw, $0.0$ for a loss).
*   $K$ is the importance-weighted update factor. It depends on:
    *   Competition type (e.g., World Cup Final = 55, Friendlies = 18).
    *   Goal difference multiplier:
        $$G(d) = \begin{cases} 1 & \text{if } d \le 1 \\ 1.5 & \text{if } d = 2 \\ \frac{11 + d}{8} & \text{if } d \ge 3 \end{cases}$$
    *   Time-decay recency weight:
        $$\text{recency} = 0.5^{(\text{months elapsed} / 18)}$$

### Expected Goals ($\lambda, \mu$)
The model translates ratings to expected goals scored by each team:
$$\lambda_A = 1.35 + \frac{(R_A + H_A) - R_B}{400}$$
$$\lambda_B = 1.35 + \frac{R_B - (R_A + H_A/2)}{400}$$
Both expected goal rates are clipped to the range $[0.3, 3.5]$.

### Dixon-Coles Adjustment
A standard Poisson distribution under-predicts low-scoring draws ($0-0$, $1-1$, etc.). To fix this, Dixon and Coles (1997) introduced an adjustment factor $\tau(x, y)$ applied to the joint probability of scores $x$ and $y$:
$$P(X=x, Y=y) = \text{Poisson}(x; \lambda) \cdot \text{Poisson}(y; \mu) \cdot \tau(x, y)$$
where the correction factor $\tau(x, y)$ is defined based on $\rho = -0.13$:
$$\tau(x, y) = \begin{cases}
1 - \lambda \mu \rho & \text{if } x = 0, y = 0 \\
1 + \lambda \rho & \text{if } x = 0, y = 1 \\
1 + \mu \rho & \text{if } x = 1, y = 0 \\
1 - \rho & \text{if } x = 1, y = 1 \\
1 & \text{otherwise}
\end{cases}$$

---

## 3. Module Breakdown

### `elo.mjs`
Exposes the core probability calculations:
*   `dcTau(a, b, lambda, mu, rho)`: Evaluates the Dixon-Coles correction coefficient.
*   `poissonPmf(k, lambda)`: Evaluates the Poisson probability of scoring $k$ goals.
*   `poissonSample(lambda, rng)`: Uses Knuth's algorithm to sample goals from a Poisson distribution.
*   `matchProb(ratingA, ratingB, homeBonusA)`: Computes the 3-way probabilities ($1X2$) by summing the joint probability grid of goal combinations from $0 \dots 8$.
*   `sampleMatch(ratingA, ratingB, homeBonus, allowDraw)`: Simulates a single scoreline for Monte Carlo runs. If a draw is not allowed (knockout rounds), it resolves ties by a shootout probability based on Elo ratings.

### `calibrate.mjs`
Calibrates team Elo ratings using historical match logs:
1.  Seeds teams with historical priors (anchored values ranging from Argentina at 2085 to Guatemala at 1345).
2.  Iterates chronologically through international matches in `data/results.json`.
3.  Applies recency decay and goal-difference scaling to compute updates.
4.  Applies a final shrinkage estimator (70% calibrated rating + 30% prior seed) to prevent friendly match noise from causing drift.

### `backtest.mjs`
Evaluates prediction performance in a strict out-of-sample walk-forward manner:
*   Matches are processed chronologically.
*   Predictions are issued using ratings compiled *only* from matches played before the kickoff date.
*   Scores predictions using:
    *   **Accuracy**: Percentage of correct win/draw/loss picks.
    *   **Ranked Probability Score (RPS)**: Evaluates probability distribution accuracy.
    *   **Log-loss & Brier Score**: Measures probability calibration.
    *   **Expected Calibration Error (ECE)**: Groups predictions into 10 bins to measure confidence calibration.
