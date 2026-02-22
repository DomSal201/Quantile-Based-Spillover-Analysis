####################################################
######   Importing necessary libraries   ###########
import numpy as np
import pandas as pd
import statsmodels.api as sm
###################################################

def create_lags(data, number_lags):
    '''
    Input
    -----
    data (pd.DataFrame): Input Data excluding the DateTime column
    number_lags (int): The number of time lags used in the QVAR-Model
    
    Returns
    -------
    data_matrix (np.numpy): Input Data excluding the first 'number_lags' rows
    regressor_matrix (np.numpy): Matrix containing the regressors for the QVAR-Model
    ------------------------------------
    Function:
    Generate lagged features for the QVAR-estimation
    ------------------------------------
    '''
    data_matrix = data.to_numpy(dtype=float)
    number_of_variables = data.shape[1]
    number_of_observation = data.shape[0]
    regressor_matrix = np.zeros((data.shape[0] - number_lags, data.shape[1] * number_lags))
    for lag in range(number_lags):
        start_col = lag * number_of_variables
        end_col = (lag+1) * number_of_variables
        regressor_matrix[:, start_col:end_col] = data_matrix[number_lags-1-lag : number_of_observation-1-lag, :]
        
    return data_matrix[number_lags:, :], regressor_matrix 


def calculate_quantile_regression(data_matrix, regressor_matrix, quantile_value):
    '''
    Input
    -----
    data_matrix (np.numpy): Input Data excluding the first 'number_lags' rows
    regressor_matrix (np.numpy): Matrix containing the regressors for the QVAR-Model
    number_lags (int): The number of time lags used in the QVAR-Model
    
    Returns
    -------
    regressor_coefficients (np.numpy): Equation-by-Equation Quantile Regression Coefficients
    residual_matrix (np.numpy): Matrix containing the residuals of the Equation-by-Equation Quantile Regression
    predicition_matrix (np.numpy): Matrix containing the estimated values
    ------------------------------------
    Function:
    Estimate the Coefficients of the equation-by-equation quantile regression and residuals.
    ------------------------------------
    '''
    X = sm.add_constant(regressor_matrix) 
    number_of_columns = data_matrix.shape[1]
    regressor_coefficients = np.zeros((number_of_columns, regressor_matrix.shape[1]))
    residual_matrix = np.zeros((data_matrix.shape[0], number_of_columns))
    prediction_matrix = np.zeros((data_matrix.shape[0], number_of_columns))
    
    for i in range(number_of_columns):
        y = data_matrix[:, i]
        try:
            model = sm.QuantReg(y, X)
            result = model.fit(q=quantile_value, method = 'highs', max_iter=5000)
            regressor_coefficients[i, :] = result.params[1:]
            prediction = result.predict(X)
            residual = y - prediction
            residual_matrix[:, i] = residual
            prediction_matrix[:, i] = prediction
        except Exception as e:
            print(f"Error at variable {i}: {e}")
            
    return regressor_coefficients, residual_matrix, prediction_matrix


def calculate_qvar_coefficients(regressor_coefficients, number_lags):
    '''
    Input
    -----
    regressor_coefficients (np.numpy): Matrix containing the coefficients of equation-by-equation quantile regression
    number_lags (int): The number of time lags used in the QVAR-Model
    
    Returns
    -------
    qvar_coeffcients (np.numpy): Matrices containing the QVAR-Coefficients
    ------------------------------------
    Function:
    Construct the QVAR-Coefficient-Matrices
    ------------------------------------
    '''
    qvar_coeffcients = np.zeros((number_lags, regressor_coefficients.shape[0], regressor_coefficients.shape[0]))
    start = 0
    end = 0
    for lag in range(0, number_lags):
        start = end
        end += regressor_coefficients.shape[0]
        qvar_coeffcients[lag, :, :] = regressor_coefficients[:, start:end]
        
    return qvar_coeffcients


def calculate_error_covariance(residuals_matrix):
    '''
    Input
    -----
    residuals_matrix (np.numpy): Matrix containing the residuals of equation-by-equation quantile regression
    
    Returns
    -------
    error_covariance_matrix (np.numpy): Matrix containing the Error-Covariance-Matrix
    ------------------------------------
    Function:
    Estimate the Error-Covariance-Matrix as the second raw moment
    ------------------------------------
    ''' 
    anzahl_residuen = residuals_matrix.shape[0]
    numerator = np.dot(residuals_matrix.T, residuals_matrix)
    denominator = anzahl_residuen
    error_covariance_matrix = numerator / denominator
    
    return error_covariance_matrix

def check_qvar_stability(qvar_coefficients):
    '''
    Input
    -----
    qvar_coeffcients (np.numpy): Matrices containing the QVAR-Coefficients
    
    Returns
    -------
    
    ------------------------------------
    Function:
    Check the stability of the QVAR model using the Companion Matrix
    ------------------------------------
    ''' 
    p, n, _ = qvar_coefficients.shape
    top_row = np.hstack(qvar_coefficients)
    if p > 1:
        bottom_rows = np.hstack([np.eye(n * (p - 1)), np.zeros((n * (p - 1), n))])
        companion_matrix = np.vstack([top_row, bottom_rows])
    else:
        companion_matrix = top_row
    eigenvalues = np.linalg.eigvals(companion_matrix)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    is_stable = max_eigenvalue < 1
    return max_eigenvalue, is_stable


def calculate_qvar(data: pd.DataFrame, number_lags: int, quantile_level: int):
    '''
    Input
    -----
    data (pd.DataFrame): Input Data excluding the DateTime column
    number_lags (int): The number of time lags used in the QVAR-Model
    quantile_level (int): The target quantile
    
    Returns
    -------
    qvar_coefficients (pd.DataFrame): Inout Data excluding the first 'number_lags' rows
    error_covariance_matrix (np.numpy): Matrix containing the Error-Covariance-Matrix
    ------------------------------------
    Function:
    Start the calculation of the QVAR model
    ------------------------------------
    '''
    
    ######### Checking Input Data #############
    non_numeric_cols = data.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        raise TypeError(f"The DataFrame contains non-numeric columns: {non_numeric_cols}.")
    if not number_lags > 0:
        raise ValueError(f"Number of lags must be greater than 0, but got {number_lags}")
    if not (0<quantile_level<1):
        raise ValueError(f"Quantile must be between 0 and 1, but got {quantile_level}")
    ###########################################
    
    data, regressor_matrix = create_lags(data, number_lags)
    regressor_coefficients, residuals_matrix, prediction_matrix = calculate_quantile_regression(data, regressor_matrix, quantile_level)
    qvar_coefficients = calculate_qvar_coefficients(regressor_coefficients, number_lags)
    error_covariance_matrix = calculate_error_covariance(residuals_matrix)
    max_eigenvalue, is_stable = check_qvar_stability(qvar_coefficients)
    print(f"Max. Eigenvalue: {max_eigenvalue:.4f}")
    print(f"System is stable: {is_stable}")
    
    return qvar_coefficients, error_covariance_matrix, prediction_matrix