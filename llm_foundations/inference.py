import numpy as np


class KVCache:
    """Minimal key/value cache for autoregressive inference.

    Stores tensors shaped as (batch, num_heads, seq_len, head_dim).
    """

    def __init__(self, max_len: int, head_dim: int, num_heads: int, batch_size: int = 1):
        self.max_len = max_len
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.k = np.zeros((batch_size, num_heads, max_len, head_dim), dtype=np.float64)
        self.v = np.zeros((batch_size, num_heads, max_len, head_dim), dtype=np.float64)
        self.length = 0

    def update(self, k: np.ndarray, v: np.ndarray) -> None:
        batch, heads, seq_len, head_dim = k.shape
        if seq_len + self.length > self.max_len:
            raise ValueError("KV cache exceeds max_len")
        if head_dim != self.head_dim or heads != self.num_heads or batch != self.batch_size:
            raise ValueError("KV cache shape mismatch")

        start = self.length
        end = self.length + seq_len
        self.k[:, :, start:end, :] = k
        self.v[:, :, start:end, :] = v
        self.length = end

    def get(self):
        return self.k[:, :, : self.length, :], self.v[:, :, : self.length, :]
