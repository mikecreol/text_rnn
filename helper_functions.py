
import numpy as np
import ipdb

def Gini(y_true, y_pred):
#    ipdb.set_trace()
    
    if(type(y_true) is list):
        y_true = np.array(y_true)
        
    if(type(y_pred) is list):
        y_pred = np.array(y_pred)
    
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1/n_samples, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true