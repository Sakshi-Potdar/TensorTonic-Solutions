import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x)
    p = np.array(p)

    if x.shape != p.shape:
        raise ValueError("Shape unequal")
    if not np.allclose(np.sum(p), 1, atol=1e-6):
        raise ValueError("Not sum to 1")

    return np.sum(x * p)
