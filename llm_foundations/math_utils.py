import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax over a chosen axis."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_mask(seq_len: int) -> np.ndarray:
    """Return a lower-triangular mask for causal attention."""
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    return mask


def scaled_dot_product(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Scaled dot-product scores for queries and keys."""
    dim = q.shape[-1]
    return (q @ k.transpose(0, 2, 1)) / np.sqrt(dim)
