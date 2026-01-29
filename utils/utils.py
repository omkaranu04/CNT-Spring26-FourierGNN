import os
import torch
import numpy as np

def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)
    
def slice_arrays(arrays, start=None, stop=None):
    """
        Slice an array or list of arrays.
        Args:
            arrays: A numpy array or list of numpy arrays to be sliced.
            start: Can be an integer, list of integers or None.
            stop: Can be an integer or None.
        Returns:
            Sliced array or list of arrays.
    """
    if arrays is None:
        return [None]
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    if isinstance(start, list) and stop is not None:
        raise ValueError('The stop argument has to be None is the start argument is a list')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]
        
def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    filename = os.path.join(model_dir, epoch + '_dhfm.pt')
    with open(filename, 'wb') as f:
        torch.save(model, f)
        
def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    filename = os.path.join(model_dir, epoch + '_dhfm.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(filename):
        return
    with open(filename, 'rb') as f:
        model = torch.load(f)
    return model

def masked_MAPE(v, v_, axis=None):
    """
        Mean Absolute Percentage Error
        Args:
            v: Ground truth values.
            v_: Predicted values.
            axis: Axis along which to compute the MAPE.
        Returns:
            MAPE value.     
    """
    mask = (v == 0)
    percentage = np.abs(v_ - v) / np.abs(v)
    if np.any(mask):
        masked_array = np.ma.masked_array(percentage, mask=mask)
        result = masked_array.mean(axis=axis)
        if isinstance(result, np.ma.MaskedArray):
            return result.filled(np.nan)
        else:
            return result
    else:
        return np.mean(percentage, axis=axis).astype(np.float64)
    
def MAPE(v, v_, axis=None):
    """
        Mean Absolute Percentage Error
        Args:
            v: Ground truth values.
            v_: Predicted values.
            axis: Axis along which to compute the MAPE.
        Returns:
            MAPE value.     
    """
    mape = (np.abs(v_ - v) / (np.abs(v) + 1e-5)).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis=axis)

def SMAPE(P, A):
    """
        Symmetric Mean Absolute Percentage Error
        Args:
            P: Predicted values.
            A: Ground truth values.
        Returns:
            SMAPE value.     
    """
    nz = np.where(A > 0)
    Pz = P[nz]
    Az = A[nz]
    num = 2 * np.abs(Pz - Az)
    den = np.abs(Az) + np.abs(Pz)
    smape = np.mean(num / den)
    return smape

def RMSE(v, v_, axis=None):
    """
        Root Mean Square Error
        Args:
            v: Ground truth values.
            v_: Predicted values.
            axis: Axis along which to compute the RMSE.
        Returns:
            RMSE value.     
    """
    return np.sqrt(np.mean((v_ - v) ** 2, axis=axis)).astype(np.float64)

def MAE(v, v_, axis=None):
    """
        Mean Absolute Error
        Args:
            v: Ground truth values.
            v_: Predicted values.
            axis: Axis along which to compute the MAE.
        Returns:
            MAE value.     
    """
    return np.mean(np.abs(v_ - v), axis=axis).astype(np.float64)

def evaluate(y, y_hat, by_step=False, by_node=False):
    """
        Evaluate the prediction performance using MAPE, RMSE, MAE.
        Args:
            y: Ground truth values.
            y_hat: Predicted values.
            by_step: Whether to return the results by each time step.
            by_node: Whether to return the results by each node.
        Returns:
            A dictionary containing MAPE, RMSE, MAE values.
    """
    if not by_step and not by_node:
        return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat)
    if by_step and by_node:
        return MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0)
    if by_step:
        return MAPE(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2))
    if by_node:
        return MAPE(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1))