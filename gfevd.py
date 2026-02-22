###################################################
######   Importing necessary libraries   ###########
import numpy as np
import pandas as pd
###################################################

def calculate_vma_coefficients (coefficients, number_of_coefficients, number_lags):
    '''
    Input
    -----
    coefficients (np.numpy): Coefficient Matrices of the (Q)VAR model
    number_lags (int): The number of time lags used in the QVAR-Model
    
    Returns
    ----
    vma_coeffcients_matrices (np.numpy); Coefficient Matrices of the (Q)VMA-representation

    Function:
    ----
    Calculate the (Q)VMA-Coefficient-Matrices

    '''
    vma_coeffcients_matrices = np.zeros((number_of_coefficients, coefficients[0].shape[0], coefficients[0].shape[1]))
    vma_coeffcients_matrices[0] = np.eye(coefficients[0].shape[0])
    index_var_coef = 0
    
    for number in range(1, number_of_coefficients):      
        for i in range(0, number_lags):
            index_var_coef = i
            if index_var_coef+1 > number:
                break
            else:
                vma_coeffcients_matrices[number] += np.dot(coefficients[index_var_coef], vma_coeffcients_matrices[number-(index_var_coef+1)])
                
    return vma_coeffcients_matrices


def calculate_girf_culumative_squared (vma_coeffcients_matrices, error_covariance_matrix, H, dimension_var_coefficients):
    '''
    Input
    -----
    vma_coeffcients_matrices (np.numpy); Coefficient Matrices of the (Q)VMA-representation
    error_covariance_matrix (np.numpy): Matrix containing the Error-Covariance-Matrix
    H (int): Forecast Horizon
    dimension_var_coefficients (int): Dimension of the (Q)VAR-Coefficient-Matrices
    
    Returns
    ----
    cumulative_squared_girf (np.numpy): Cumulative squared GIRF

    Function:
    ----
    Calculate the cumulative squared GIRF

    '''
    cumulative_squared_girf = np.zeros((dimension_var_coefficients,dimension_var_coefficients))
    for i in range (0, H):
        temp = np.dot(vma_coeffcients_matrices[i], error_covariance_matrix)
        cumulative_squared_girf += np.multiply(temp,temp)
    for j in range(0, cumulative_squared_girf.shape[1]):
        cumulative_squared_girf[:, j] /= error_covariance_matrix[j,j]
    return cumulative_squared_girf


def calculate_forecast_error_covariance_matrix (vma_coeffcients_matrices, error_covariance_matrix, H, dimension_var_coefficients):
    '''
    Input
    -----
    vma_coeffcients_matrices (np.numpy); Coefficient Matrices of the (Q)VMA-representation
    error_covariance_matrix (np.numpy): Matrix containing the Error-Covariance-Matrix
    H (int): Forecast Horizon
    dimension_var_coefficients (int): Dimension of the (Q)VAR-Coefficient-Matrices
    
    Returns
    ----
    forecast_error_covariance_matrix (np.numpy): Forecast Error Covariance Matrix for Horizon H

    Function:
    ----
    Calculate the Forecast Error Covariance Matrix for Horizon H

    '''
    forecast_error_covariance_matrix = np.zeros((dimension_var_coefficients,dimension_var_coefficients))
    for index in range(0,H):
        forecast_error_covariance_matrix += np.dot(np.dot(vma_coeffcients_matrices[index], error_covariance_matrix), vma_coeffcients_matrices[index].T)
    return forecast_error_covariance_matrix


def calculate_gfevd (cumulative_squared_girf, forecast_error_covariance_matrix):
    '''
    Input
    -----
    cumulative_squared_girf (np.numpy): Cumulative squared GIRF
    forecast_error_covariance_matrix (np.numpy): Forecast Error Covariance Matrix for Horizon H
    
    Returns
    ----
    gfevd (np.numpy): Generalized Forecast Error Variance Decomposition for Horizon H

    Function:
    ----
    Calculate the Generalized Forecast Error Variance Decomposition for Horizon H

    '''
    if cumulative_squared_girf.shape == forecast_error_covariance_matrix.shape:
        gfevd = np.zeros((forecast_error_covariance_matrix.shape[0], forecast_error_covariance_matrix.shape[1]))
        for i in range(0, cumulative_squared_girf.shape[0]):
           for j in range(0, cumulative_squared_girf.shape[1]):
               gfevd[i,j] = cumulative_squared_girf[i,j] / forecast_error_covariance_matrix[i,i]
    else:
        ValueError('Dimension Error')
    return gfevd


def calculate_normalized_gfevd (gfevd):
    '''
    Input
    -----
    gfevd (np.numpy): Generalized Forecast Error Variance Decomposition for Horizon H
    
    Returns
    ----
    normalized_gfevd (np.numpy): Normalized Generalized Forecast Error Variance Decomposition for Horizon H

    Function:
    ----
    Calculate the Normalized Generalized Forecast Error Variance Decomposition for Horizon H
    ------------------------------------
    '''
    normalized_gfevd = np.zeros(gfevd.shape)
    for row in range(0, gfevd.shape[0]):
        row_sum = gfevd[row,:].sum()
        for column in range(0, gfevd.shape[1]):
            normalized_gfevd[row, column] = gfevd[row, column] / row_sum
    return normalized_gfevd


def gfevd(var_coefficients_matrices, number_of_vma_coefficients, error_covariance_matrix, H, number_lags):
    '''
    Input
    -----
    gfevd (np.numpy): Generalized Forecast Error Variance Decomposition for Horizon H
    
    Returns
    -----
    normalized_gfevd (np.numpy): Normalized Generalized Forecast Error Variance Decomposition for Horizon H
    
    Function:
    -----
    Calculate the Normalized Generalized Forecast Error Variance Decomposition for Horizon H
    
    '''
    ######### Checking Input Data #############
    if isinstance(error_covariance_matrix, (pd.DataFrame, pd.Series)):
        error_covariance_matrix = error_covariance_matrix.to_numpy()
    if not isinstance(H, int):
        raise TypeError(f"Expected int, but got {type(H).__name__}")
    if not isinstance(number_lags, int):
        raise TypeError(f"Expected int, but got {type(number_lags).__name__}")
    ###########################################
    
    dimension = var_coefficients_matrices.shape[1]
    vma_coefficients = calculate_vma_coefficients(var_coefficients_matrices, number_of_vma_coefficients, number_lags)
    girf_culumative_squared = calculate_girf_culumative_squared(vma_coefficients, error_covariance_matrix, H, dimension)
    forecast_error_covariance_matrix = calculate_forecast_error_covariance_matrix(vma_coefficients, error_covariance_matrix, H, dimension)
    gfevd = calculate_gfevd(girf_culumative_squared, forecast_error_covariance_matrix)
    normalized_gfevd = calculate_normalized_gfevd(gfevd)
    normalized_gfevd = normalized_gfevd*100
    
    return normalized_gfevd