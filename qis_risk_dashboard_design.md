# QIS Portfolio Correlation and Diversification Review

> Implementation status and approved overrides are recorded in Appendix I. Where it differs from the original draft, Appendix I governs this build.

## 1. Purpose and scope

Build a PM-friendly Jupyter notebook for reviewing diversification and correlation risk across approximately ten QIS strategies. Explain current modeled risk, changes since the previous monthly snapshot, the relationships responsible, and vulnerability to lost diversification.

The portfolio holds nonnegative allocations to excess-return indices. Target allocations total 100%. Convert targets to index units at each rebalance close and hold units constant between rebalances. Measure weights and percentage risk against the resulting **synthetic excess-return portfolio value**. Collateral yield is not added.

This document specifies the dashboard; it does not implement the notebook. The visible report is a portfolio review. Full strategy tables and model details belong in the technical appendix, complementing the separate strategy-level dashboard.

Allocation optimization, proposed-trade analysis, realized loss reviews, historical-regime stress estimation, and factor modelling are outside this version. Annualized volatility measures second-moment risk; it does not summarize all QIS risks, including nonlinear losses, liquidity, or tail dependence.

## 2. PM review flow

### 2.1 How diversified is the portfolio today?

Use three headline cards:

| Card | Supporting context |
|---|---|
| Portfolio volatility | Annualized estimate and change since the previous monthly snapshot |
| Diversification Ratio (DR) | Current value and monthly change |
| Correlation uplift | Volatility difference from the zero-correlation reference; show that reference volatility alongside it |

Explain the counterfactual in plain language: **“At today's holdings and standalone volatilities, estimated portfolio volatility is X%; it would be Y% with zero pairwise correlation, a difference of Z volatility points.”** Preserve the sign when the difference is negative. Zero correlation is a reference scenario, not a diversification target or the lowest achievable risk.

Below the cards, show aligned historical panels for portfolio volatility and its zero-correlation reference; DR; and the concentration/correlation decomposition of DR. Use separate aligned axes for the effective number of standalone risk exposures and the correlation multiplier. These have different units and should not share a dual-axis chart.

A compact correlation-conditions panel shows the fixed equal-risk reference indicator and fast-versus-slow model disagreement. These are supporting context, not additional headline cards or calibrated alerts.

### 2.2 Why did risk change?

Show a waterfall from the previous monthly portfolio volatility to the current estimate, labelled **Weights / Standalone volatility / Correlation**. Each effect is in annualized volatility points and reconciles to the total change.

The weights effect includes both deliberate reallocation and performance-driven drift. Do not describe the entire bar as a PM trading decision. The correlation bar measures the modeled effect of changing the correlation matrix, with interactions allocated by the convention in Appendix D.

Use end-of-close holdings at each endpoint. State both snapshot dates. For an intra-month report, label the comparison “Change since [previous month-end date]”; do not imply that it covers a completed month.

### 2.3 Which relationships matter?

Provide separate compact pair tables for current cross-covariance contributions and monthly correlation-driven effects:

| View | Ranking measure | Supporting columns |
|---|---|---|
| Current relationships | Absolute current pair contribution, retaining its sign | Pair, current correlation, current contribution in volatility points |
| Monthly correlation drivers | Absolute pair Shapley correlation effect, retaining its sign | Pair, previous/current correlation, correlation change, monthly risk effect |

Default to five pairs per view. Include a signed **Other pairs** row equal to the sum of omitted contributions and a total that reconciles to the relevant aggregate. Break ranking ties by the stable strategy identifier pair. Show fewer rows when fewer pairs exist. For a single-strategy portfolio, show “No strategy pairs.”

Keep current contribution, change in that contribution, and correlation-driven monthly effect distinct. Rank monthly drivers by portfolio-risk impact, not by raw correlation changes. Signed contributions can be negative even though strategy allocations are nonnegative.

Place the optional correlation heatmap after the ranked tables. Keep its color scale fixed at [-1, 1] and strategy ordering stable. Full pair and strategy tables are available in the appendix.

### 2.4 Where could diversification disappear?

Show correlation-convergence scenarios that close 25% and 50% of the distance from each current pair correlation to +1, holding holdings and standalone volatilities fixed. Report stressed volatility, uplift in volatility points, percentage increase, and stressed DR.

Rank five pair contributors to the **25% convergence scenario** and reconcile the remainder as Other pairs. This highlights lost diversification: a pair contributing little risk today can still be important if its correlation rises.

Add a small grid with correlation-convergence fractions of 0%, 25%, and 50% as columns and uniform standalone-volatility multipliers of 1.00, 1.25, and 1.50 as rows. Each cell shows portfolio volatility and uplift from the current baseline. Show DR once per correlation column because uniform volatility scaling leaves it unchanged.

Label all stresses **illustrative sensitivities without assigned probabilities**. They describe vulnerability under the specified shocks, not forecasts or comprehensive worst cases.

## 3. Presentation and controls

| Report element | Visibility |
|---|---|
| Portfolio, requested/resolved as-of date, comparison date, holdings/rebalance dates | Visible |
| Compact data status and available model-history dates | Visible |
| Three headline cards and a short calculated narrative | Visible |
| Four PM review sections above | Visible |
| Estimator controls, full strategy/pair tables, diagnostics and methodology | Collapsed/expandable appendix |

```python
PM_CONTROLS = {
    "as_of_date": None,       # Latest observed input date when None
    "history_months": 36,     # Display request, limited to available model history
    "top_n_pairs": 5,
    "show_heatmap": True,
}
```

Use one run-all workflow without requiring manual execution of intermediate cells. Presentation controls must not change the estimation sample or model parameters. Keep tables and charts readable as static notebook output; widgets may enhance navigation but must not be necessary to understand the report.

Presentation rules:

- Express volatility levels as annualized percentages and effects as annualized volatility percentage points; 0.01 in decimal volatility equals 1 volatility point.
- Use the official slow correlation model consistently for baseline risk, DR, contributions, attribution and stress.
- Preserve contribution signs. Avoid percentage attribution shares, especially when effects offset or the net change is near zero.
- Generate narrative statements mechanically from displayed calculations. Summarize the largest signed changes without claiming statistical significance, causality, or prescribing trades.
- Show unavailable values with a short reason. Do not replace missing history with zero or carry estimates forward through invalid snapshots.
- Display rounded values, calculate with full precision, and explain small displayed reconciliation differences due to rounding.

## 4. Technical methodology appendix

The following sections define the calculations and input behavior. They should be accessible from the notebook but need not occupy the default PM view.

### Appendix A. Inputs, holdings and snapshot timing

#### A.1 Input contract

| Input | Required convention |
|---|---|
| Strategy data | Wide daily index levels, or daily arithmetic returns, indexed by date and stable strategy identifier |
| Return basis | Comparable excess returns relative to collateral yield, in the declared portfolio currency; any required conversion occurs upstream |
| Target allocations | Complete vectors by rebalance-close date; explicitly zero for unallocated strategies; nonnegative and summing to 1 within tolerance |
| Calendar | Declared trading-date calendar and common valuation-close convention |
| Universe | Explicit ordered list, fixed for a report's estimates, attribution and reference history |
| Run metadata | Portfolio identifier, currency, return-basis declaration, input provenance and model version |

Use one authoritative strategy input mode per run. Validate sorted, unique dates and unique identifiers, finite observed values, and positive index levels. Arithmetic returns must exceed -1 so that compounded index levels remain positive. Missing values are permitted only with the explicit availability behavior below.

Reject unidentified strategies, missing allocation entries, negative allocations, and target sums outside tolerance. Do not silently renormalize weights or redistribute unavailable strategy allocations. All index levels must be valid at a rebalance close. Reject rebalance dates outside the declared trading calendar instead of silently moving the trade.

For arithmetic-return input, construct normalized levels through:

$$
I_{i,t}=I_{i,t-1}(1+r_{i,t}).
$$

Use an arbitrary positive common base, such as 100, at the close preceding the first return. A missing return prevents reconstruction of subsequent levels for that strategy until an explicit level anchor is supplied; do not assume a zero return or compound across the gap. The default return-only adapter therefore requires an uninterrupted return chain through the valuation period.

For index-level input, compute daily returns before selecting common observations:

$$
r_{i,t}=\frac{I_{i,t}}{I_{i,t-1}}-1.
$$

Here $t-1$ is the preceding date on the declared calendar. Require valid levels at both dates. Do not forward-fill prices or remove missing dates before calculating returns: that would misclassify a multi-day return as a daily return. Valid zero returns remain observations; repeated zero returns receive a stale-data diagnostic rather than automatic imputation or deletion.

#### A.2 Synthetic excess-return book

Let $I_{i,t}$ be an excess-return index level, $n_{i,t}^{+}$ closing units after any rebalance, and $E_t$ closing synthetic excess-return portfolio value. Starting from the first supplied rebalance, choose $E_{b_0}=100$ and set units using that date's targets. Do not infer holdings before the first allocation record.

At a subsequent rebalance close $b$, first mark the previously held units:

$$
E_b=\sum_i n_{i,b^-}I_{i,b}.
$$

Then set new units:

$$
n_{i,b}^{+}=\frac{x^{\mathrm{target}}_{i,b}E_b}{I_{i,b}}.
$$

This preserves portfolio value because the target allocations sum to 1. Between rebalances, hold units constant:

$$
E_t=\sum_i n_i I_{i,t},\qquad
x_{i,t}=\frac{n_i I_{i,t}}{E_t}.
$$

Closing weights on a rebalance date equal the new targets; closing weights on other dates are drifted weights. For consecutive valued trading dates, the synthetic portfolio return is:

$$
r_{p,t}=\frac{E_t}{E_{t-1}}-1
=\sum_i x_{i,t-1}^{+}r_{i,t}.
$$

The old units earn the rebalance day's return. The new units exist immediately after that close and earn the next trading day's return. Use closing post-rebalance weights with estimates through the same close for a risk snapshot; this is the model-implied risk of the holdings going forward, not an estimate of the exposure held earlier that day.

Require positive, finite portfolio value. Initial value and arbitrary index rebasing change unit counts but not returns, weights or percentage risk. No external flows, transaction costs, collateral yield, or additional financing adjustments enter this synthetic book. Label its value and risk basis explicitly; it is not a reconstruction of actual funded NAV.

In the holdings detail, retain latest target weights and date, current drifted weights, current units, and the most recent rebalance's before/after weights. Do not split the waterfall's weights bar into trading and drift effects without a separately defined attribution method.

#### A.3 Reporting dates, missing data and historical comparisons

Resolve a non-trading requested as-of date to the preceding declared trading close. With `as_of_date=None`, start from the latest observed input date. Show both requested and resolved dates. If required valuations at that close are missing, mark the requested snapshot unavailable rather than silently substituting an older close.

The comparison endpoint is the last declared trading close of the preceding calendar month. Monthly history uses each calendar month's last trading close; an intra-month current snapshot is shown separately from completed months. Use the holdings and risk estimates applicable at each historical close, not today's weights applied retrospectively.

Maintain separate availability checks for holdings valuation and estimator observations. A missing index level makes that day's valuation unavailable. With index-level input, later complete levels can still value the known units, provided every intervening rebalance was valid. A gap must not produce a fabricated one-day portfolio or strategy return.

For estimation, use the joint set of complete daily return vectors across the fixed universe. Disclose excluded observation dates, sample counts and coverage. Do not use pairwise deletion or imputation. A missing comparison snapshot suppresses the monthly waterfall and monthly pair effects; it does not invalidate an otherwise valid current snapshot.

Permit zero allocations within the fixed universe. A change in the configured universe creates a new report version: recompute comparable history on the new universe where eligible, and leave earlier unsupported periods unavailable. Do not splice old-universe and new-universe results into an apparently continuous comparison or reference series.

### Appendix B. Risk model and estimator specification

#### B.1 Notation

| Symbol | Meaning |
|---|---|
| $x_i$ | Closing strategy weight in the synthetic excess-return book |
| $s_i$ | Annualized standalone volatility |
| $a_i=x_i s_i$ | Standalone risk exposure |
| $S=\sum_i a_i$ | Sum of standalone risk exposures |
| $R$ | Official slow correlation matrix |
| $\Sigma$ | Annualized covariance matrix |
| $\sigma_p$ | Annualized portfolio volatility |

Define:

$$
a=x\odot s,\qquad
\Sigma=\operatorname{diag}(s)R\operatorname{diag}(s),\qquad
\sigma_p=\sqrt{x^\top\Sigma x}=\sqrt{a^\top Ra}.
$$

#### B.2 Finite-history EWMA and lagged standardization

Index dates by their ordinal position on the declared trading calendar. For half-life $h$, set $\lambda_h=2^{-1/h}$. Let $\mathcal T_t$ contain complete daily strategy-return vectors available through close $t$. Normalize weights over the available finite history:

$$
\omega_{h,t,u}=\frac{\lambda_h^{t-u}}
{\sum_{v\in\mathcal T_t}\lambda_h^{t-v}},\qquad u\in\mathcal T_t.
$$

Use every eligible observation through the snapshot; the PM display window does not truncate estimation history. Decay uses elapsed trading-calendar positions, including dates excluded because of missing observations. Do not compress gaps into adjacent observations.

Under the zero-mean model, daily standalone variance and annualized volatility are:

$$
d_{i,t}^2=\sum_{u\in\mathcal T_t}\omega_{60,t,u}r_{i,u}^2,
\qquad s_{i,t}=\sqrt{252}\,d_{i,t}.
$$

The weights above define initialization explicitly; there is no infinite-history seed, demeaning, or degrees-of-freedom adjustment. Require at least 60 preceding common observations before forming a standardized return:

$$
z_{i,u}=\frac{r_{i,u}}{d_{i,u-1}}.
$$

The denominator uses only returns available through the preceding calendar close. The return being standardized must not enter its own denominator. Do not clip standardized returns or introduce an undocumented volatility floor. A zero or numerically negligible required standalone variance makes the standardized model unavailable; do not silently remove that strategy from the reference universe.

Let $\mathcal Z_t$ contain eligible standardized-return vectors. For each correlation half-life $h\in\{126,42\}$, calculate:

$$
Q_{h,t}=\frac{\sum_{u\in\mathcal Z_t}\lambda_h^{t-u}z_u z_u^\top}
{\sum_{u\in\mathcal Z_t}\lambda_h^{t-u}},\qquad
R_{h,t}^{\mathrm{raw}}=\operatorname{diag}(Q_{h,t})^{-1/2}
Q_{h,t}\operatorname{diag}(Q_{h,t})^{-1/2}.
$$

Here $\operatorname{diag}(Q)$ denotes the diagonal matrix containing $Q$'s diagonal entries. The common weighted outer products preserve positive semidefiniteness; normalizing their diagonal produces the correlation matrix. Validate positive finite diagonal entries before normalization. These are lag-volatility-standardized EWMA correlation estimates. The separation of conditional volatility and standardized-return dependence follows the approach described in [NYU V-Lab's methodology](https://vlab.stern.nyu.edu/docs/correlation/GJR-DCC-NL); this specification supplies its own EWMA estimator.

Slow and fast models share the same input sample, standardization and shrinkage settings. Only correlation half-life differs. Use the slow matrix for official risk and the fast matrix only for the stated comparison.

#### B.3 Shrinkage and eligibility

Default shrinkage is zero. For sensitivity $\alpha\in\{0,0.10,0.25\}$, use:

$$
R_{h,t}^{(\alpha)}=(1-\alpha)R_{h,t}^{\mathrm{raw}}+\alpha T_{h,t},
$$

where $T_{h,t}$ has diagonal 1 and every off-diagonal entry equal to the unweighted mean of the unique off-diagonal entries of $R_{h,t}^{\mathrm{raw}}$. This constant-correlation target and the convex mixture are positive semidefinite for a valid input correlation matrix. For a one-strategy universe, set $T=[1]$. Apply a chosen sensitivity consistently to both endpoints of any attribution calculation.

Require at least 504 common daily return observations for a published risk snapshot, plus valid holdings and model inputs. With uninterrupted nondegenerate data and 60 initialization observations, the first eligible snapshot contains 444 standardized vectors. Display both counts, calendar span, and normalized-weight effective sample size $1/\sum_u\omega_u^2$ for each estimator; use the standardized sample's weights for correlation effective sample size. This count is a weighting diagnostic, not a claim about independent observations or statistical confidence.

Historical output starts only when both the estimator and holdings are eligible. A 36-month display request does not guarantee 36 months of modeled history: more than two years of input history are consumed before the first default risk snapshot.

### Appendix C. Current diversification and risk contributions

#### C.1 Zero-correlation reference and concentration

Define:

$$
V_{\mathrm{self}}=\sum_i a_i^2,\qquad
V_{\mathrm{cross}}=2\sum_{i<j}a_i a_j\rho_{ij},\qquad
\sigma_p^2=V_{\mathrm{self}}+V_{\mathrm{cross}}.
$$

Then:

$$
\sigma_0=\sqrt{V_{\mathrm{self}}},\qquad
U_R=\sigma_p-\sigma_0,\qquad
M_R=\frac{\sigma_p}{\sigma_0}.
$$

$U_R$ is the primary current correlation-effect metric: positive values mean uplift relative to zero correlation; negative values mean lower risk. $M_R$ expresses the same comparison as a ratio. Both hold the current standalone risk exposures fixed for the counterfactual, but their histories also change when the exposure mix changes.

The Diversification Ratio and concentration measure are:

$$
DR=\frac{S}{\sigma_p},\qquad
N_{\mathrm{eff}}=\frac{S^2}{\sum_i a_i^2},\qquad
\boxed{DR=\frac{\sqrt{N_{\mathrm{eff}}}}{M_R}}.
$$

The DR numerator is the sum of weighted standalone volatilities. Euler contributions must not be used in that numerator because they already sum to portfolio volatility.

For nonnegative exposures and meaningful denominators:

- $DR=1$ means no volatility diversification benefit relative to the sum of standalone risk exposures; larger values mean more benefit.
- $N_{\mathrm{eff}}$ is the **effective number of standalone risk exposures**. It measures concentration of the $a_i$, ranges from 1 to the number of positive exposures, and equals that count only when those exposures are equal.
- $N_{\mathrm{eff}}$ is not an estimate of independent economic bets. Neither $N_{\mathrm{eff}}$ nor $DR^2$ should be labelled a discovered number of independent factors.
- Under zero correlation, $M_R=1$ and $DR=\sqrt{N_{\mathrm{eff}}}$. Negative correlations can produce $M_R<1$ and DR above this reference.
- Uniform positive scaling of exposures leaves DR, $N_{\mathrm{eff}}$ and $M_R$ unchanged.

Use the exact identity to explain a DR change alongside concentration and the live correlation multiplier. It is a level decomposition, not an additive or causal attribution of the change in DR. A decline in DR alone is not evidence of deteriorating correlations.

#### C.2 Fixed reference and fast-versus-slow disagreement

For the fixed ordered universe of $N$ strategies, use an equal-standalone-risk reference vector $a^*=\mathbf1$:

$$
M_{R,t}^*=\sqrt{\frac{\mathbf1^\top R_t\mathbf1}{N}}.
$$

This reference isolates changes in the estimated correlation matrix from changes in live holdings and standalone risk exposures. It does not isolate economic causality or remove estimator uncertainty. Use the same reference universe throughout a plotted history.

Hold current weights and standalone volatilities fixed when comparing the correlation models:

$$
W_t=\sigma_p(x_t,s_t,R_t^{\mathrm{fast}})
-\sigma_p(x_t,s_t,R_t^{\mathrm{slow}}).
$$

Positive $W_t$ means recent standardized-return dependence implies more portfolio risk than the official model; negative values imply less. Call this **fast-versus-slow model disagreement**. Do not assign probability, statistical-significance, or traffic-light thresholds. Show it beside $M_{R,t}^*$ because the gap can return toward zero after the slow model catches up even when correlation conditions remain elevated.

#### C.3 Euler and additive cross-covariance contributions

For strategy $i$:

$$
\rho_{i,p}=\frac{(\Sigma x)_i}{s_i\sigma_p},\qquad
RC_i=x_i s_i\rho_{i,p}=\frac{x_i(\Sigma x)_i}{\sigma_p},\qquad
\sum_i RC_i=\sigma_p.
$$

Split this into:

$$
RC_i^{\mathrm{self}}=\frac{a_i^2}{\sigma_p},\qquad
RC_i^{\mathrm{cross}}=\frac{a_i\sum_{j\ne i}a_j\rho_{ij}}{\sigma_p}.
$$

At portfolio level:

$$
C_{\mathrm{self}}=\frac{V_{\mathrm{self}}}{\sigma_p},\qquad
C_R=\frac{V_{\mathrm{cross}}}{\sigma_p},\qquad
C_{\mathrm{self}}+C_R=\sigma_p.
$$

Label $C_R$ **additive cross-covariance contribution to portfolio volatility**. It depends on allocations, standalone volatilities and correlations. It is distinct from the counterfactual uplift $U_R$; the two are not interchangeable or generally equal.

For each unique pair $i<j$:

$$
PC_{ij}=\frac{2a_i a_j\rho_{ij}}{\sigma_p},\qquad
\sum_{i<j}PC_{ij}=C_R.
$$

Assign half of each pair contribution to each strategy when aggregating to $RC_i^{\mathrm{cross}}$. The full strategy table contains current and latest target weights, standalone volatility, standalone risk exposure, portfolio correlation, and Euler/self/cross contributions. Negative Euler or pair contributions are valid and remain signed.

### Appendix D. Monthly risk-change attribution

Compare closing snapshots $(x^0,s^0,R^0)$ and $(x^1,s^1,R^1)$ using:

$$
f(x,s,R)=(x\odot s)^\top R(x\odot s).
$$

Evaluate all six orders of updating the three blocks: weights $X$, standalone volatilities $s$, and correlation $R$. In each order, start with the previous snapshot, update one block at a time, and record its incremental variance effect. Average each block's increment over the six orders:

$$
\Delta V_p=\phi_X^V+\phi_s^V+\phi_R^V.
$$

Convert to annualized volatility units:

$$
\phi_k^\sigma=\frac{\phi_k^V}{\sigma_{p,1}+\sigma_{p,0}},\qquad
\Delta\sigma_p=\phi_X^\sigma+\phi_s^\sigma+\phi_R^\sigma.
$$

Call this **variance-Shapley attribution expressed in volatility units**. It symmetrically allocates interactions and reconciles exactly. It is an accounting convention, not proof of economic causality or a Shapley calculation performed directly on volatility. Use “Standalone volatility” in the PM waterfall; reserve $\Sigma$ for the covariance matrix.

For the exact pair attribution of the correlation bar, define:

$$
\Delta\rho_{ij}=\rho_{ij}^1-\rho_{ij}^0,\qquad
a^{00}=x^0\odot s^0,\quad a^{10}=x^1\odot s^0,\quad
a^{01}=x^0\odot s^1,\quad a^{11}=x^1\odot s^1.
$$

Then:

$$
\phi_{R,ij}^V=2\Delta\rho_{ij}
\left[
\frac13a_i^{00}a_j^{00}
+\frac16a_i^{10}a_j^{10}
+\frac16a_i^{01}a_j^{01}
+\frac13a_i^{11}a_j^{11}
\right],
$$

$$
\phi_{R,ij}^\sigma=\frac{\phi_{R,ij}^V}{\sigma_{p,1}+\sigma_{p,0}},\qquad
\sum_{i<j}\phi_{R,ij}^\sigma=\phi_R^\sigma.
$$

Assign half of each pair effect to each strategy for the appendix's strategy-level monthly correlation-driver table. Do not substitute $PC_{ij}^1-PC_{ij}^0$ for this effect: that change also reflects changing exposures and the contribution denominator.

Use identical model configuration, ordered universe, estimation conventions and input vintage for the two snapshots. A change in model settings requires recomputing both endpoints; it must not be attributed to a market correlation change.

### Appendix E. Correlation convergence and combined sensitivity

For $0\le\kappa\le1$:

$$
R_\kappa=(1-\kappa)R+\kappa\mathbf1\mathbf1^\top,\qquad
\sigma_\kappa=\sqrt{a^\top R_\kappa a}.
$$

The matrix remains positive semidefinite with unit diagonal. The scenario closes fraction $\kappa$ of each pair's distance to +1; it does not add $\kappa$ correlation points. For nonnegative exposures:

$$
\sigma_\kappa^2=(1-\kappa)\sigma_p^2+\kappa S^2.
$$

Portfolio volatility cannot decrease as $\kappa$ increases, and at $\kappa=1$ it equals $S$. This is the maximum volatility attainable by varying a valid correlation matrix with these nonnegative exposures held fixed.

The exact contribution of each pair to correlation-only stressed volatility uplift is:

$$
SC_{ij}(\kappa)=\frac{2\kappa a_i a_j(1-\rho_{ij})}
{\sigma_\kappa+\sigma_p},\qquad
\sum_{i<j}SC_{ij}(\kappa)=\sigma_\kappa-\sigma_p.
$$

These pair effects are nonnegative and reconcile to the same simultaneous, valid matrix stress. A low or negative current correlation can create a large convergence contribution. The PM stress pair table uses $\kappa=0.25$ and unchanged standalone volatilities. Do not present its total as attribution of the combined volatility-and-correlation grid.

For the grid, let $m>0$ uniformly multiply every standalone volatility, keeping weights fixed:

$$
\sigma_{m,\kappa}=m\sigma_\kappa,\qquad
U_{m,\kappa}=m\sigma_\kappa-\sigma_p,\qquad
P_{m,\kappa}=\frac{m\sigma_\kappa}{\sigma_p}-1,
$$

$$
DR_{m,\kappa}=\frac{mS}{m\sigma_\kappa}=\frac{S}{\sigma_\kappa}.
$$

Use $\kappa\in\{0,0.25,0.50\}$ and $m\in\{1,1.25,1.50\}$. In the correlation-only table, $m=1$; its outputs are stressed volatility, absolute uplift, percentage increase and DR. The grid uses the same original current baseline for every uplift. Uniform volatility scaling is a transparent sensitivity and does not model strategy-specific volatility responses.

### Appendix F. Configuration, diagnostics and numerical behavior

#### F.1 Authoritative model configuration

```python
MODEL_CONFIG = {
    "portfolio_value_basis": "synthetic_excess_return",
    "holdings_method": "constant_units_between_rebalances",
    "rebalance_timing": "close",
    "target_weight_sum": 1.0,
    "allow_negative_weights": False,
    "vol_half_life": 60,
    "annualization": 252,
    "mean_model": "zero",
    "standardization": "lagged_ewma_daily_volatility",
    "standardization_warmup_observations": 60,
    "corr_half_life_slow": 126,
    "corr_half_life_fast": 42,
    "ewma_weighting": "normalized_finite_history",
    "ewma_decay_clock": "declared_trading_calendar",
    "shrinkage_alpha": 0.00,
    "shrinkage_target": "constant_correlation",
    "min_history_observations": 504,
    "missing_data_policy": "common_complete_daily_return_vectors",
    "change_attribution": "variance_shapley",
    "rho_reference": "fixed_equal_standalone_risk",
    "correlation_stress_kappa": [0.00, 0.25, 0.50],
    "volatility_stress_multipliers": [1.00, 1.25, 1.50],
    "pair_stress_kappa": 0.25,
    "percentile_history_months": 36,
    "percentile_min_prior_snapshots": 24,
    "weight_sum_atol": 1e-8,
    "reconciliation_rtol": 1e-8,
    "reconciliation_atol": 1e-12,
    "correlation_matrix_atol": 1e-10,
    "annualized_volatility_epsilon": 1e-10,
}

SHRINKAGE_SENSITIVITY = [0.00, 0.10, 0.25]
```

Require the portfolio identifier, ordered universe, calendar, currency and return-basis metadata separately from these numerical defaults and record them with the run. The configuration object is authoritative; example formulas use the defaults, but calculation code reads its resolved parameters. Model choices without a specified alternative, such as the zero-mean convention and holdings basis, are fixed supported modes for this version; reject unsupported overrides.

Validate positive half-lives and annualization, at least one initialization observation, minimum history greater than the initialization count, $0\le\alpha,\kappa\le1$, positive volatility multipliers, and inclusion of the designated pair stress in the configured stress scenarios. Changing a quant setting reruns all compared snapshots consistently.

#### F.2 Diagnostics

Keep the following outside the headline PM view:

- Input coverage, missing observations, repeated zero-return runs, holdings dates, sample counts and effective sample sizes.
- Correlation symmetry, diagonal, minimum eigenvalue and all contribution reconciliations.
- Shrinkage sensitivity of portfolio risk, DR and pair rankings; show changed signs or rankings without presenting them as statistical confidence intervals.
- Half-life sensitivity using paired correlation settings (slow, fast) of (84, 28), (126, 42) and (189, 63), with the volatility estimator held fixed.
- Autocorrelation and a comparison using non-overlapping weekly returns, with 52-period annualization. Require consecutive weekly valuation endpoints; flag weeks that do not span the intended calendar interval. This is an annualization diagnostic, not a replacement official estimator.
- Historical DR percentiles using valid completed month-end snapshots in the preceding 36 calendar months, excluding the current snapshot. Require at least 24 preceding valid snapshots; otherwise show unavailable. Report reference dates and count. Use midrank percentile $100[\#(DR_u<DR_t)+0.5\#(DR_u=DR_t)]/K$, calculated before display rounding.

The percentile window is a model/diagnostic setting independent of the PM chart window. Historical percentiles are descriptive and must not imply a calibrated risk limit. Evaluate serial dependence before treating daily square-root-of-time annualization as a useful multi-period approximation. Broader forecast calibration can be added later; no forecast-versus-realized review is required for this version.

#### F.3 Validation and unavailable results

Validate target sums with absolute tolerance $10^{-8}$ and no normalization. Validate matrix symmetry and unit diagonal within $10^{-10}$ and minimum eigenvalue at least $-10^{-10}$. Require positive finite diagonals before correlation normalization; singular positive-semidefinite matrices are allowed. Do not silently repair materially invalid matrices. A negative quadratic form within documented floating-point tolerance may be clamped to zero only after matrix validation, with the adjustment recorded.

Check reconciliations in decimal annualized-volatility or variance units as appropriate:

$$
|\mathrm{lhs}-\mathrm{rhs}|\le10^{-12}+10^{-8}\max(|\mathrm{lhs}|,|\mathrm{rhs}|).
$$

Use an annualized-volatility epsilon of $10^{-10}$ for volatility denominators, and its squared value for corresponding variance denominators. Apply the equivalent daily threshold when testing standalone variances for standardization. Do not replace small denominators by epsilon and display an artificial finite ratio.

When portfolio volatility is zero or negligible, display the volatility estimate and any well-defined counterfactuals, but mark DR, Euler/current pair contributions and percentage stress uplifts unavailable where their denominators fail. $M_R$ remains defined, including zero, when $\sigma_0$ is meaningful; $N_{\mathrm{eff}}$ remains defined when $V_{\mathrm{self}}$ is meaningful. Evaluate the Shapley and pair stress conversion denominators independently: one zero-volatility endpoint is acceptable if the sum of endpoint volatilities is meaningful. If both are negligible, keep available variance effects in diagnostics and suppress the volatility-unit conversion. With no meaningful pair contributions, show unavailable rather than zero contributions that imply a valid reconciliation.

Invalid input contracts, an invalid required rebalance, an invalid current model, or a material reconciliation failure block publication of a valid PM risk report and instead show a diagnostic failure summary. Missing historical endpoints or mathematically undefined ratios suppress only the affected outputs. Insufficient current estimation history produces an unavailable-risk report with coverage details, not an invented estimate.

### Appendix G. Acceptance checks

Implement the following checks with the future notebook calculation modules. This table is part of the design specification, not a request to add a test suite with this document revision.

| Scenario | Required result |
|---|---|
| Units held between rebalances | Units remain constant; weights change with relative index performance and sum to 1 within tolerance |
| Rebalance close | Old units earn that day's return; the trade preserves value; new weights equal targets and earn the following day's return |
| Initial-value or index-base scaling | Percentage returns, drifted weights and percentage risk are unchanged |
| Missing/invalid allocations | Negative weights, incomplete vectors and sums outside tolerance are rejected without normalization |
| Euler/self/cross reconciliation | Euler contributions and the self-plus-cross decomposition each sum to portfolio volatility |
| Unique pairs and strategy aggregation | Pair contributions sum to $C_R$; half-pair aggregation reproduces strategy cross contributions |
| Monthly Shapley | Six-order variance effects sum to variance change; volatility-unit effects sum to volatility change; pair correlation effects sum to the correlation bar |
| Concentration identity | $DR=\sqrt{N_{\mathrm{eff}}}/M_R$ whenever denominators are meaningful |
| Independent strategies | $R=I$ gives $U_R=0$, $M_R=1$, $C_R=0$ and $DR=\sqrt{N_{\mathrm{eff}}}$, while convergence stress can still increase risk |
| Perfect positive correlation | Portfolio volatility equals $S$, DR equals 1, and correlation-convergence uplift is zero |
| Negative correlations | Valid signed contributions and $M_R<1$ are preserved where applicable; no clipping of hedging effects |
| Single strategy or one allocated strategy | $N_{\mathrm{eff}}=DR=M_R=1$ for meaningful risk, portfolio cross contribution and convergence uplift are zero; an actual one-strategy universe has no pairs |
| Zero allocations | Zero-exposure strategies have zero portfolio and pair contributions; they remain in the explicitly configured estimator/reference universe |
| Negligible portfolio volatility | Undefined ratios/contributions are unavailable; valid variance and counterfactual calculations remain accessible |
| Correlation stress | Matrices remain valid; risk is nondecreasing in $\kappa$; pair stress contributions reconcile exactly to volatility uplift |
| Uniform volatility stress | Volatility scales by $m$, DR is unchanged at fixed $\kappa$, and grid uplifts use the original current baseline |
| Missing price or return | No forward-fill, no multi-day observation presented as daily, and unavailable periods remain visible |
| EWMA initialization and gaps | Finite weights sum to 1; missing dates preserve calendar decay; 504 uninterrupted raw observations yield 444 standardized vectors |
| No future-data leakage | Changing returns or weights after an as-of close does not alter that snapshot; a return is excluded from its own standardization denominator |
| Limited history and percentiles | Charts begin at model/holdings eligibility; percentile is unavailable with fewer than 24 valid preceding monthly snapshots |
| Universe or model change | Comparisons use one universe and configuration; unsupported endpoints are unavailable rather than spliced across definitions |
| Top-pair presentation | Rankings use the specified impact, signs are preserved, and shown rows plus Other pairs equal the complete aggregate |

A deterministic holdings example fixes the timing convention. Start with value 100 and two indices at 100, allocated 50% each, giving 0.5 units of each. At the next close the indices are 120 and 90: value is 105 and pretrade weights are $4/7$ and $3/7$. Rebalance to 25%/75% at that close, producing 0.21875 and 0.875 units. At the following close, levels 132 and 81 give value 99.75, a -5% return on the new holdings. The preceding +5% return belongs to the old holdings.

### Appendix H. Implementation boundaries and reproducibility

Keep the future notebook thin. Supporting Python modules own calculations and return structured results; presentation code formats those results without recomputing risk under different conventions.

| Component | Responsibility |
|---|---|
| `qis_risk_dashboard.ipynb` | Load inputs/configuration, run the review and render output |
| `qis_risk/data.py` | Input validation, index/return preparation, calendar alignment, unit ledger and snapshot holdings |
| `qis_risk/estimators.py` | Finite-history EWMA, lagged standardization, correlation and shrinkage |
| `qis_risk/risk.py` | Risk, DR, concentration/reference metrics and current contributions |
| `qis_risk/attribution.py` | Monthly Shapley and pair decomposition |
| `qis_risk/stress.py` | Correlation convergence, pair stress attribution and combined grid |
| `qis_risk/plots.py` | PM cards, charts, ranked/reconciled tables and calculated narrative |
| `qis_risk/validation.py` | Availability, numerical validation and reconciliation results |
| `config/default_model.yaml` | Versioned numerical defaults |

Apply notebook overrides once, validate the resolved configuration, and pass that same object to every module. A snapshot result should carry closing holdings, estimator outputs, risk metrics and contributions, sample coverage and availability reasons. Pair views and narratives consume those outputs. Monetary values and risk quantities stay unrounded until presentation.

Record portfolio and model versions, full resolved configuration, PM display settings, ordered universe, calendar/valuation convention, currency and excess-return basis, requested/resolved/comparison dates, rebalance/holdings dates, input coverage and content fingerprints, source provenance, numerical adjustments, and run timestamp. Record live/backtested status and data vintage if supplied; otherwise label them unknown.

Reproducing a historical estimate with today's revised input file is a calculation on the current data vintage. Claim a point-in-time historical record only when archived inputs or availability metadata support it. Within a run, use no observations after each snapshot date. The same inputs and model configuration must reproduce the same analytical results regardless of display window or run timestamp.

### Appendix I. Approved implementation plan and milestone ledger

#### I.1 Approved scope and architecture

Implement the full PM report and diagnostics as one thin notebook plus three substantive modules: `qis_risk.data` (CSV contracts, calendar, holdings, simulation), `qis_risk.model` (validated dataclasses, estimation, analytics, diagnostics, structured results), and `qis_risk.report` (shared static notebook/HTML presentation). A thin package initializer exposes the public functions. Numerical defaults are Python dataclass fields, not a separate YAML file. The notebook contains only imports, paths, parameters, analysis orchestration, display, and export calls.

Use Python 3.13 with uv and a committed lockfile; NumPy, pandas, and Matplotlib supply calculations and static charts. Keep notebook and development dependency groups separate. HTML embeds its charts and styling, needs no network or JavaScript, and uses the same renderer as the notebook. Deliver to private `TylerWang1996/qis-dashboards` main after verification. Keep real data, generated reports, environments, and executed notebook copies out of Git. Commit only the two explicitly simulated CSV inputs.

#### I.2 Approved conventions and concrete defaults

- V1 accepts index levels only; the arithmetic-return adapter in Appendix A is deferred. CSVs start with `date`, with matching strategy identifiers and an explicit ordered universe. Target weights are decimal fractions (`0.25 = 25%`) summing to one; retain all original holdings and rebalance-close conventions.
- CSV dates are the authoritative trading calendar. Blank strategy values remain missing. Entirely omitted trading dates cannot be detected. `calendar_complete_through` defaults to the latest index date and may explicitly certify later non-trading calendar coverage. Only periods whose calendar boundary is within certified coverage and the report cutoff count as completed. Otherwise display the latest snapshot separately. Weeks end on Friday. Missing whole periods remain gaps.
- Structural CSV validation applies globally. Numeric domain and required-rebalance validation apply through the resolved as-of date, so future numeric errors cannot invalidate an earlier report. Explicit as-of requests outside certified coverage are unavailable; do not silently move them back into supported coverage.
- Published model risk requires both the configured minimum raw common returns and at least `min_history_observations - standardization_warmup_observations` eligible standardized vectors: defaults 504 and 444. Invalid lag denominators exclude the standardized vector with a recorded reason; never drop a strategy from the fixed universe. Current required nonpositive/negligible variances invalidate the required model.
- Use normalized running EWMA accumulators that decay on every declared date. Reuse standardized vectors across correlation settings, calculate scalar historical metrics only at needed endpoints, and retain complete pair details only where needed. No persistent cache or repeated expanding-history fits.
- Weekly diagnostics use the trailing 36 calendar months and require 104 valid completed Friday-ended weekly intervals. Use consecutive weekly endpoints and complete intervening common daily observations, retaining identical intervals for daily and weekly estimates. Report standalone-strategy zero-mean second-moment volatilities annualized by 252 and 52, their ratio, and daily/weekly lag-one Pearson autocorrelations. Do not bridge gaps. These diagnostics do not replace the official estimator.
- After validating a correlation matrix, define quadratic-form tolerance as `correlation_matrix_atol * dot(a, a) + 64 * machine_epsilon * sum(abs(a))**2`. Clamp a negative variance only within this tolerance, recording the adjustment; reject more negative values. Do not substitute epsilon into ratio denominators.
- Simulation uses seed 42, ten stable strategy identifiers, eight years (2018–2025) of weekday observations, changing correlated volatility conditions, and complete monthly rebalance targets including zero allocations. It is an illustrative synthetic calendar without an exchange holiday claim. A 5,000-row simulator option supports local performance measurement.

#### I.3 Public interfaces and agent ownership

`load_inputs(...) -> InputData`; `run_review(inputs, portfolio, model, as_of_date=None) -> ReviewResult`; `render_report(result, display) -> str`; `export_html(...)` writes that same standalone report. Structured results distinguish valid, unavailable (with reasons), and failed states. Display settings cannot change estimation history or model settings.

Parallel builders own data/tests and model/tests respectively. The primary integrator owns presentation, notebook, packaging, documentation, and Git checkpoints. An independent reviewer examines mathematical behavior against this specification and reference calculations. Only one writer edits a file at a time. Continue through tested checkpoints without routine approval pauses; preserve unresolved external blockers explicitly and resume from the ledger.

#### I.4 Milestones and evidence

| Milestone | Acceptance gate | Status / evidence |
|---|---|---|
| 1. Foundation and contracts | Package/configuration imports; private owner/remote verified | Complete: locked environment and imports verified; GitHub confirms owner TylerWang1996 and isPrivate=true. Foundation checkpoint 8158ad4. |
| 2. Data and holdings | Canonical holdings example, scaling, missingness, allocation and timing checks | Complete: 29 data tests pass; independent reviewer found no blocking defect. Includes exact 100→105→99.75 example, gap recovery, historical numeric cutoffs, and deterministic CSV roundtrip. |
| 3. Estimation and current risk | Direct-reference EWMA, eligibility, no leakage, current-risk identities | Complete: direct finite-history reference and gap/lag/count tests pass. Independent 200-case covariance/Euler check agrees within 1.11e-16. |
| 4. Attribution and stress | Shapley/pair reconciliations and stress identities | Complete: independent Shapley coalition reference, pair decompositions, zero-volatility endpoints, stress limits and scaling checks pass. |
| 5. Complete diagnostics | Correct sensitivity/weekly/percentile samples and reproducibility | Complete: 57 model tests pass. Reviewer verified fixes for complete calendar-month percentiles, duplicate scenario rejection, and unavailable zero-volatility sensitivity ranks. No outstanding model-review blocker. |
| 6. Notebook and HTML | Fresh-kernel run-all; offline report parity; visual inspection | Complete: fresh-kernel execution and exact HTML/display parity pass. Browser inspection verified cards, aligned histories, waterfall, tables, correlation matrix, stress grid, and expandable appendix. JupyterLab inspection verified the executed notebook's output, styles and readable cards. |
| 7. Review and delivery | Full checks; documented benchmark; clean committed notebook; verified private main push | Local verification complete: 98 tests and Ruff pass; all independent review findings resolved; performance targets met; source notebook has no outputs. Final private-main push and remote CI verification pending. |

Use three focused test files for data, model, and report, including Appendix G cases. CI runs Ruff, pytest, and a fresh-kernel notebook test. Benchmark ten strategies and approximately 5,000 dates locally: aim for analysis under five seconds after imports and notebook execution under thirty seconds. Record hardware and timings; do not use machine-sensitive CI timing assertions. Record evidence and commit each accepted checkpoint.

Acceptance mapping: data tests cover Appendix G holdings/timing/scaling, invalid allocations, missing observations, and calendar contracts. Model tests cover EWMA initialization/gaps/leakage, contribution and concentration identities, Shapley, correlation/uniform-volatility stresses, singular/negative/single/zero-exposure cases, denominator availability, model changes, and diagnostic history. Report tests cover stable signed top-pair reconciliation, unavailable reports, display invariance, embedded assets, a thin clean notebook, and exact fresh-kernel notebook/HTML parity. The complete integration suite currently passes 98 tests.

Percentile reference membership uses calendar-month buckets from current month minus 36 through current month minus one, then retains valid completed snapshots. A weekend/holiday month-end does not shorten the reference window. Stress fractions and multipliers must be unique. Sensitivity rank, baseline rank, rank change, and sign change are unavailable when their required pair contributions are undefined.

Final local measurement (2026-09-07): AMD Ryzen 9 PRO 8945HS, Linux 7.1.13, Python 3.13.12. For 5,000 dates × 10 strategies, warmed analysis runs were 1.212, 1.205 and 1.202 seconds (median 1.205 seconds). The complete demo notebook, including a fresh kernel and HTML rendering/export, took 2.649 seconds. HTML size: 288,149 bytes. Both performance targets pass. Full suite: 98 tests passed; Ruff passed. Generated verification metadata and the executed notebook are retained only under ignored `reports/`.

#### I.5 Durable goal

Complete all milestones, resolve independent review findings, verify notebook and HTML visually, record performance, and push validated commits to private main. A working subset or a merely passing test suite is not completion. On interruption, read this ledger and inspect the current Git state before resuming; avoid redoing verified checkpoints.
