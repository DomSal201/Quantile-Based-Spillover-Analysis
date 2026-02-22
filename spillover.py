###################################################
######   Importing necessary libraries   ###########
import numpy as np
import pandas as pd
###################################################

def calculate_total_spillover(gfevd_table):
    '''
    Input
    -----
    gfevd_table (np.numpy): Resulting table of the GFEVD
    
    Returns
    ----
    tsi (float): Total Spillover Index

    Function:
    ----
    Calculate the Total Spillover Index

    '''
    numerator = 0
    denominator = gfevd_table.sum().sum()
    
    for i in range(gfevd_table.shape[1]):
        for j in range(gfevd_table.shape[0]):
            if i == j:
                continue
            else:
                numerator += gfevd_table[i,j]
    tsi = (numerator / denominator) * 100
    
    return np.round(tsi,2)

def calculate_directional_spillover(gfevd_table):
    '''
    Input
    -----
    gfevd_table (np.numpy): Resulting table of the GFEVD
    
    Returns
    ----
    directional_spillover (float): Directional Spillover

    Function:
    ----
    Calculate the Directional Spillover

    '''
    
    denominator = 1
    directional_spillover = np.zeros((3, gfevd_table.shape[1]))
    for i in range(gfevd_table.shape[1]):
        numerator = 0
        for j in range(gfevd_table.shape[1]):
            if i == j:
                continue
            else:
                numerator += gfevd_table[i,j]
        directional_spillover[0, i] = (numerator / denominator)
    for i in range(gfevd_table.shape[1]):
        numerator = 0
        for j in range(gfevd_table.shape[1]):
            if i == j:
                continue
            else:
                numerator += gfevd_table[j,i]
        directional_spillover[1, i] = (numerator / denominator)
    directional_spillover[2,:] = directional_spillover[1,:] - directional_spillover[0,:]
        
    return np.round(directional_spillover, 2)


def spillover_table(gfevd_table, data_columns):
    '''
    Input
    -----
    gfevd_table (np.numpy): Resulting table of the GFEVD
    data_columns (pd.DataFrame columns): Columns of the Data
    
    Returns
    ----
    spillover_table (pd.DataFrame): Spillover Table

    Function:
    ----
    Calculate the Spillover Table

    '''
    data_columns = data_columns.tolist()
    total_spillover_index = calculate_total_spillover(gfevd_table)
    directional_spillover = calculate_directional_spillover(gfevd_table)
    row_index = data_columns.copy()
    column_index = data_columns.copy()
    row_index.append('Zu Anderen')
    row_index.append('Netto-Spillover')
    column_index.append('Von Anderen')
    spillover_table = pd.DataFrame(index=row_index, columns=column_index)
    spillover_table.iloc[0:gfevd_table.shape[0], 0:gfevd_table.shape[1]] = np.round(gfevd_table,2)           
    spillover_table.iloc[-2,:-1] = directional_spillover[1,:]
    for i in range(directional_spillover.shape[1]):
        spillover_table.iloc[i,-1] = directional_spillover[0, i]
    spillover_table.iloc[-2,-1] = total_spillover_index
    for i in range (spillover_table.shape[0]-2):
        spillover_table.iloc[-1,i] = float(spillover_table.iloc[-2,i]) - float(spillover_table.iloc[i,-1])
    return spillover_table