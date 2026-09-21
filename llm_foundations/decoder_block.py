"""A readable NumPy decoder block for studying the forward pass."""

import numpy as np

from .math_utils import causal_mask
from .multi_head_attention import MultiHeadAttention


def gelu(x: np.ndarray) -> np.ndarray:
    """Tanh approximation used by many GPT-style implementations."""
    return 0.5 * x * (
        1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))
    )


class LayerNorm:
    """Normalize the feature axis of each token independently."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * normalized + self.beta


class MLP:
    """The same two-layer feature transformation at every token position."""

    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator):
        scale = 1.0 / np.sqrt(d_model)
        self.W1 = rng.normal(0.0, scale, size=(d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.normal(0.0, 1.0 / np.sqrt(d_ff), size=(d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2


class DecoderBlock:
    """Pre-norm causal decoder block.

    Input and output shape: (batch, sequence, d_model).
    Dropout is intentionally omitted so every number is deterministic.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, seed: int = 0):
        self.norm1 = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, np.random.default_rng(seed))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, sequence, d_model)")
        mask = causal_mask(x.shape[1])[None, :, :]
        h = x + self.attention(self.norm1(x), mask=mask)
        return h + self.mlp(self.norm2(h))
