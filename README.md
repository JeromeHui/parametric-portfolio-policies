# State-Dependent Asset Pricing via Parametric Portfolio Policies: A Penalized Regression Approach

[![R](https://img.shields.io/badge/Language-R-blue.svg)](https://www.r-project.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview
This repository contains an end-to-end quantitative research and engineering pipeline that integrates high-dimensional machine learning with empirical asset pricing. Departing from traditional two-step mean-variance optimization—which suffers from severe error-maximization—we implement a **Parametric Portfolio Policy (PPP)** framework where portfolio weights are modeled directly as a linear function of stock characteristics (Brandt, Santa-Clara, and Valkanov, 2004). 

To handle high-dimensional feature spaces and mitigate multicollinearity among hundreds of equity anomalies, we utilize **Elastic Net regularization** paired with a nested rolling-window cross-validation scheme. Furthermore, we construct conditional interaction terms by multiplying option-implied volatility smile slopes (`SmileSlope`) with baseline equity characteristics to capture state-dependent risk premiums.

---

## Methodology Architecture
1. **Data Cleaning & Preprocessing**: Cross-sectional median imputation, 1%-99% Winsorization for outlier clipping, and Z-score standardization within each monthly cross-section to ensure scale invariance.
2. **Feature Engineering**: Integration of core equity characteristics alongside state-dependent conditional interaction terms (e.g., `SmileSlope × IdioVol`).
3. **Utility Maximization & Regularization**: Optimization via constant relative risk aversion (CRRA) logarithmic utility coupled with an Elastic Net penalty ($\alpha = 0.5$) to balance $L_1$ sparsity and $L_2$ ridge shrinkage.
4. **Nested Rolling-Window Optimization**: A 60-month rolling training window combined with a 12-month inner validation horizon to dynamically select the optimal regularization parameter ($\lambda$) via BFGS quasi-Newton optimization.

---

## Key Empirical Findings

### Portfolio Performance
| Model Specification | Annualized Return | Out-of-Sample Sharpe Ratio | Benchmark Annualized Return | Benchmark Sharpe Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Elastic Net ($\alpha = 0.0$)** | 28.41% | 1.025 | 10.10% | 0.516 |
| **Elastic Net ($\alpha = 0.5$)** | 32.69% | 1.039 | 10.10% | 0.516 |
| **Elastic Net ($\alpha = 1.0$)** | 41.85% | 1.026 | 10.10% | 0.516 |
| **Original PPP Model** | 44.65% | 1.034 | 10.10% | 0.516 |

* **Key Insight**: All machine learning variants and the original parametric model achieve robust out-of-sample Sharpe ratios between **1.025 and 1.039**, substantially outperforming the equal-weight benchmark's Sharpe ratio of **0.516**.

### Top-Ranked Features & Interaction Impact
Analysis of mean absolute coefficients ($\theta$) and selection rates across rolling windows demonstrates that option-market tail-risk pricing (`SmileSlope`) and its state-dependent interactions (`SmileSlope_x_IdioVolAHT`, `SmileSlope_x_DolVol`) play a dominant role in driving optimal portfolio weights alongside traditional anomalies like `Mom12mOffSeason`.

---

## Repository Structure
```text
├── README.md               # Project presentation and summary
├── docs/                   # Full academic research paper (PDF/Word)
├── code/                   # Cleaned and modularized R scripts
└── output/                 # Generated performance tables and feature charts
