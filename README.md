# State-Dependent Asset Pricing via Parametric Portfolio Policies: A Penalized Regression Approach

[![R](https://img.shields.io/badge/Language-R-blue.svg)](https://www.r-project.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview
This repository includes an end-to-end pipeline for quantitative research and engineering. We combine high-dimensional machine learning with empirical asset pricing. Instead of following the usual two-step mean-variance optimisation which may cause large errors, we use Parametric Portfolio Policies (Brandt, Santa-Clara, and Valkanov, 2004) and model the weights of the portfolio as a linear function of the stock characteristics.

We use LASSO regression (Elastic Net) along with a nested rolling window cross-validation to control for the high dimensionality and multicollinearity in our hundreds of equity anomalies. We then create conditional interactions by multiplying SmileSlope (option implied volatility smile slope) with our base equity characteristics.

---

## Methodology
**Data Cleaning and Preprocessing:**
Impute missing values using the cross-sectional median, truncate extreme values by winsorizing at 1%, 99%, and normalize the data using Z score normalization on a monthly cross-sectional basis.

**Feature Engineering:**
Construct features using fundamental equity characteristics as well as interactions between variables conditional on the state (`SmileSlope × IdioVol`).

**Optimization:**
Maximize the utility function using a CRRA logarithmic utility function and Elastic Net penalty term ($\alpha = 0.5$). This penalises the number of non-zero coefficients in the model ($L_1$) and shrinks the coefficients in the model ($L_2$).

**Rolling Window Optimization:**
Use a 60 month rolling window with an internal 12 month validation period. Optimize the regularization parameter ($\lambda$) using a BFGS quasi Newton method.

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
The analysis of average absolute coefficients ($\theta$) and selection rates in rolling window reveals that option market tail risk variables (SmileSlope) and state-dependent interactions (SmileSlope_x_IdioVolAHT and SmileSlope_x_DolVol) play a dominant role in determining optimal weights, over traditional anomalies like Mom12mOffSeason.

---

## Repository Structure
```text
├── README.md               # Project presentation and summary
├── docs/                   # Full academic research paper (PDF/Word)
├── code/                   # Cleaned and modularized R scripts
└── output/                 # Generated performance tables and feature charts
