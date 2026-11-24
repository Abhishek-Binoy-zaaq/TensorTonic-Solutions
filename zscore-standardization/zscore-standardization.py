import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    # Write code here
    X_train = np.asarray(X, dtype=float)

    if X_train.size == 0:
        return X_train.copy()

    mean = np.mean(X_train, axis = axis, keepdims = True)
    std = np.std(X_train, axis = axis, keepdims = True)
    safe_std = np.maximum(std, eps)
    return (X_train-mean)/safe_std