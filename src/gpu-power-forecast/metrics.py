import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_metrics(preds, actual, delta=1.0):
    """
    Compute RMSE, MAE, and Huber loss metrics.
    
    Args:
        preds: Predicted values
        actual: Actual values
        delta: Threshold for Huber loss (default=1.0)
    
    Returns:
        rmse, mae, huber
    """
   
    # Compute Huber loss
    error = actual - preds
    abs_error = np.abs(error)
    
    # Huber loss: quadratic for small errors, linear for large errors
    huber = np.where(
        abs_error <= delta,
        0.5 * error**2,  # MSE for small errors
        delta * (abs_error - 0.5 * delta)  # MAE-like for large errors
    )
    huber_loss = np.mean(huber)
    
    return huber_loss
