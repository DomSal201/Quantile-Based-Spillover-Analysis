####################################################
######   Importing necessary libraries   ###########
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
###################################################
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning)

def VAR_MODEL(data, maxlags: int):
    '''
    Input
    -----
    data (pd.DataFrame): Input Data excluding the DateTime column
    
    Returns
    -----
    result_model: Estimation results
    forecast_df: Forecast values
    
    Function:
    -----
    Calculate the VAR-Coefficients
    
    '''
    ######### Checking Input Data #############
    if not maxlags > 0:
        raise ValueError(f"Number of lags must be greater than 0, but got {number_lags}")
    ###########################################
    model = VAR(data)
    result_model = model.fit(maxlags=maxlags, method='ols', verbose=True, trend='c')
    print(f"Is the model stable? {result_model.is_stable()}")
    number_lags = result_model.k_ar
    print(f"Number of lags: {number_lags}")
    forecast_df = result_model.fittedvalues

    return result_model, forecast_df