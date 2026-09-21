import numpy as np

from .math_utils import causal_mask, softmax, scaled_dot_product


def scaled_dot_product_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Self-attention over query, key, value matrices.

    q, k, v: shape (batch, seq_len, d_k)
    """
    scores = scaled_dot_product(q, k)

    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    weights = softmax(scores, axis=-1)
    return weights @ v


def causal_self_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Causal self-attention with a lower-triangular mask."""
    seq_len = q.shape[1]
    mask = causal_mask(seq_len)
    mask = mask[None, :, :]
    return scaled_dot_product_attention(q, k, v, mask=mask)
