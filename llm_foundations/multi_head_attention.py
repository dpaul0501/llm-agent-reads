import numpy as np

from .self_attention import scaled_dot_product_attention


class MultiHeadAttention:
    """A minimal multi-head attention implementation.

    Input shape: (batch, seq_len, d_model)
    Output shape: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, num_heads: int):
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        rng = np.random.default_rng(0)
        self.W_q = rng.normal(0, 1 / np.sqrt(d_model), size=(d_model, d_model))
        self.W_k = rng.normal(0, 1 / np.sqrt(d_model), size=(d_model, d_model))
        self.W_v = rng.normal(0, 1 / np.sqrt(d_model), size=(d_model, d_model))
        self.W_o = rng.normal(0, 1 / np.sqrt(d_model), size=(d_model, d_model))

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch, seq_len, d_model = x.shape
        return x.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        batch, _, seq_len, head_dim = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)

    def __call__(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        out = np.empty_like(q)
        for h in range(self.num_heads):
            out[:, h, :, :] = scaled_dot_product_attention(
                q[:, h, :, :],
                k[:, h, :, :],
                v[:, h, :, :],
                mask=mask,
            )

        out = self._combine_heads(out)
        return out @ self.W_o
