# GTA VI Trophy Rarity Forecast — E/N Decomposition Model

[![Status](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Entries](https://make.fun/badge/gta6-trophy-rarity-forecast-01-entries.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Prize](https://make.fun/badge/gta6-trophy-rarity-forecast-01-prize.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

An entry for the [make.fun](https://make.fun) competition **gta6-trophy-rarity-forecast-01**: forecasting every PlayStation 5 trophy earn rate for *Grand Theft Auto VI* on Exophase at the target checkpoint as five quantiles per trophy.

## The E/N Decomposition Model

Traditional forecasters extrapolate trophy earn rates directly. However, Exophase earn rates are ratios:
$$\text{rate}(t) = \frac{E(t)}{N(t)}$$
where $N(t)$ is the number of players tracked and $E(t)$ is how many of them have earned a given trophy. As new players join (launch growth, $N \uparrow$), every rate gets diluted unless progress ($E \uparrow$) matches it. Crucially, new players joining early on hold almost exclusively the instant 100% trophy ("Just Like Paradise") with minimal initial progress on other trophies.

Our model explicitly separates numerator and denominator:
1. **Deduplication:** History is parsed into distinct states (since Exophase counts only update at refreshes).
2. **Players Tracked $N(t)$ Extrapolation:** Least-squares fit of $\log N$ against $\ln(1 + \text{hours})$ over the last 24 hours of distinct states, with saturation damping ($0.7 \times$ extra log-growth) and guardrails. If history is flat (no refresh observed), a documented-launch prior assumes $N$ multiplies by ~6 and new players hold only the 100% trophy.
3. **Earned Counts $E_i(t)$ Extrapolation:** Per trophy, least-squares fit of $\log(E_i + 0.5)$ against $\ln(1 + \text{hours})$, with slope clamped to $[0, 2]$, floored at last earned count, and capped at $0.99 \times N_{\text{pred}}$.
4. **Median Rate & Dilution:** $\text{median}_i = 100 \times E_{i,\text{pred}} / N_{\text{pred}}$ (clamped to $[0.01, 99.99]$). The 100% trophy keeps median 100.
5. **Calibrated Spread (`logit sigma`):** Incorporates random-walk variance over horizon days, fit residuals, extrapolation horizon scaling, and refresh staleness ($0.35$). Quantiles are computed via inverse logit with standard normal z-scores and rounded to 4 decimals half-up.

## Implementation

- **Language:** Python 3.13 (Standard Library only).
- **Performance:** Executes in ~0.05 seconds per case (well under the 8-second limit).
- **Robustness:** Handles empty history, flat launch states, null rates, and degenerate inputs gracefully and deterministically.
