# Quantile-Based Spillover Analysis

This repository contains a set of Python methods for calculating **Quantile-based Spillover-Effects**.
The approach extends the well-known spillover framework developed by Francis X. Diebold and Kamil Yilmaz.

The implementation is inspired by the R package [ConnectednessApproach](https://github.com/GabauerDavid/ConnectednessApproach) by David Gabauer.

## Module Overview

### `gfevd.py`
Calculates the cumulative **Generalized Impulse Response Function** and the **forecast error covariance matrix**.
From these, it computes the **Normalized Generalized Forecast Error Variance Decomposition (GFEVD)** following:

- Gary Koop, M. Hashem Pesaran & Simon M. Potter (1996)
- H. Hashem Pesaran & Yongcheol Shin (1998)

The routines work with both standard VAR and quantile‑VAR (QVAR) coefficient matrices.

### `qvar.py`
Estimates the coefficients of a QVAR model via equation‑by‑equation quantile regressions (using `statsmodels`).
It returns:

- QVAR coefficient matrices
- residual covariance matrix
- fitted values

The file also includes helper functions for lag generation, stability checks, and covariance estimation.

### `spillover.py`
Builds the **spillover table** from a normalized GFEVD matrix according to the Diebold‑Yilmaz methodology.
Functions compute the total spillover index, directional spillovers (to/from), and net spillovers.

### `spillover_analysis.py`
Serves as a high‑level wrapper providing a closed workflow for computing spillovers.
It supports both VAR and QVAR methods and includes extra utilities such as rolling‑window analysis
and plotting routines.

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/DomSal201/Quantile-Based-Spillover-Analysis.git
   cd Quantile-Based-Spillover-Analysis
   ```

2. **Install dependencies**   There is a `requirements.txt` file listing all required packages. Install them with:
   ```bash
   pip install -r requirements.txt
   ```
   or install individually:   
   ```bash
   pip install numpy pandas statsmodels matplotlib
   ```

3. **Use the modules in your analysis**
   Import and call the functions from the modules above to estimate spillovers from your data.

## References

- Diebold, F. X. & Yilmaz, K. (2012) – *Measuring Financial Asset Return and Volatility Spillovers*
- Koop, G., Pesaran, M. H. & Potter, S. M. (1996) - *Impulse response analysis in nonlinear multivariate models*
- Pesaran, M. H. & Shin, Y. (1998) - *Generalized impulse response analysis in linear multivariate models*
- Gabauer, D. – [ConnectednessApproach (R package)](https://github.com/GabauerDavid/ConnectednessApproach)
- Ando, Tomohiro and Greenwood-Nimmo, Matthew and Shin, Yongcheol (2018) - *Quantile Connectedness: Modelling Tail Behaviour in the Topology of Financial Networks*

> **Note:** This repository is for academic and research purposes only. It is not intended for use in actual financial trading or investment decisions. The results produced by this software are based on specific econometric assumptions and should be interpreted with caution. The author assumes no responsibility for any financial losses or technical errors arising from the application of this code.