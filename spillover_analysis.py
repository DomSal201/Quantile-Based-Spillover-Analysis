###################################################
######   Importing necessary libraries   ###########
import numpy as np
import pandas as pd
from qvar import calculate_qvar
from gfevd import gfevd
from spillover import spillover_table
from var import VAR_MODEL
import matplotlib.pyplot as plt
###################################################

def calculate_spillover(data, number_of_lags: int, method: str = 'VAR', number_vma_coef = 200, forecast_horizon=10, quantile_level: float | None=None):
    '''
    Input
    -----
    method (str): Selection between VAR and QVAR Analysis, \\
    data (pd.DataFrame): Input Data excluding the DateTime column \\
    number_of_lags (int): Number of lags \\
    number_vma_coef (int): Number of VMA Coefficients
    
    Returns
    ----
    spillover_table (pd.DataFrame): Spillover Table

    Function:
    ----
    Calculate the Spillover Table for a VAR or QVAR Model
    '''
    ######### Checking Input Data #############
    non_numeric_cols = data.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        raise TypeError(f"The DataFrame contains non-numeric columns: {non_numeric_cols}.")
    if not number_of_lags > 0:
        raise ValueError(f"Number of lags must be greater than 0, but got {number_of_lags}")
    if not number_vma_coef > 0:
        raise ValueError(f"Number of VMA Coefficients must be greater than 0, but got {number_of_lags}")
    if not forecast_horizon > 0:
        raise ValueError(f"Forecast Horizon must be greater than 0, but got {number_of_lags}")
    if quantile_level is not None:
        if not (0 < quantile_level < 1):
            raise ValueError(f"Quantile must be between 0 and 1, but got {quantile_level}")
    ###########################################
    if method == "VAR":
        result_model, forecast = VAR_MODEL(data, number_of_lags)
        normalized_gfevd = gfevd(result_model.coefs, number_vma_coef, result_model.sigma_u, forecast_horizon, number_of_lags)
        spillover_values = spillover_table(normalized_gfevd, data.columns)
        return spillover_values
    elif method == "QVAR":
        if quantile_level is None:
            raise ValueError(f"Quantile must be between 0 and 1, but got None")
        else:
            qvar_coefficients, error_covariance_matrix, predicition_matrix = calculate_qvar(data, number_of_lags, quantile_level)
            normalized_gfevd = gfevd(qvar_coefficients, number_vma_coef, error_covariance_matrix, forecast_horizon, number_of_lags)
            spillover_values = spillover_table(normalized_gfevd, data.columns)
            return spillover_values
    else:
        raise ValueError(f"Unknown method: {method}")
    
'''   
def calculate_spillover_rolling_window(data, number_of_lags: int, spillover_from: str, spillover_to: str, method: str = 'VAR', number_vma_coef = 200, forecast_horizon=10, quantile_level: float | None=None, window_size: int = 200):
    # TBD!!!!
    window_results_list = []
    for start in range(len(data) - window_size + 1):
        window = data.iloc[start : start + window_size]
        print("Durchlauf 1")
        if window.isnull().values.any():
            continue
        try:
            table = calculate_spillover(window, number_of_lags, method, number_vma_coef, forecast_horizon, quantile_level)
            window_results_list.append({
                'date': window.index[-1],
                'table': table
            })
        except Exception as e:
            print(f"Fehler bei Index {start}: {e}")
            continue
        
    ##### Plot #####
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelweight": "bold",            
        "grid.alpha": 0.3
    })
        
    row_idx = data.index.get_loc(spillover_to) 
    col_idx = data.columns.get_loc(spillover_from)
    indices = [res['date'] for res in window_results_list]
    values = np.array([res['table'].iloc[row_idx, col_idx] for res in window_results_list])
    true_dates = pd.to_datetime(data['Date'].iloc[indices])
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.plot(true_dates, values, color='#1a4a73', linewidth=1.2, label='_nolegend_')
    ax.fill_between(true_dates, 0, values, color='#1a4a73', alpha=0.5, label='_nolegend_')
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Spillover", fontsize=10)
    ax.grid(True, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.autofmt_xdate()
    ax.legend(loc='best', frameon=True)
    plt.show()
'''

def calculate_tsi_all_quantiles(data, number_of_lags: int, number_vma_coef = 200, forecast_horizon=10):
    results = []
    quantiles = [q / 100 for q in range(1, 100)]
    for quantile in quantiles:
        table = calculate_spillover(data, number_of_lags, 'QVAR', number_vma_coef, forecast_horizon, quantile)
        results.append(table.iloc[-2,-1])

    plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False, 
    "axes.spines.right": False,  
    "pdf.fonttype": 42           
    })


    quantiles = np.linspace(0.01, 0.99, len(results))

    plt.figure(figsize=(10, 5))
    plt.plot(quantiles, results, 
            linestyle='-', 
            color='#1A43BF', 
            linewidth=2,
            label = 'TSI'
            )
    plt.xlabel('Quantile ($\\tau$)', fontsize=12)
    plt.ylabel('TSI', fontsize=12)
    plt.xticks([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    plt.yticks(alpha=0.7)

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(frameon=False, loc='upper center')
    plt.tight_layout()
    plt.show()  