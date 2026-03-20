import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.array(x)

    if x.ndim == 1:
        x_stable = x - np.max(x)
        x_exp = np.exp(x_stable)
        return x_exp / np.sum(x_exp)
    elif x.ndim == 2:
        x_stable = x - np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x_stable)
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)
    else:
        raise ValueError("Invalid input")