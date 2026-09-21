import numpy as np


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Rotate every pair of dimensions by 90 degrees.

    x shape: (..., d)
    """
    x = np.asarray(x)
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires even dimension size.")

    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = np.stack((-x2, x1), axis=-1).reshape(x.shape)
    return rotated


def apply_rope(x: np.ndarray, positions: np.ndarray | None = None, base: float = 10000.0) -> np.ndarray:
    """Apply rotary positional embeddings to a tensor.

    Args:
        x: (..., seq_len, d_model) or (..., d_model)
        positions: optional positions array of shape (seq_len,)
        base: RoPE base frequency

    Returns:
        x with rotary position applied.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires even hidden size.")

    if x.ndim == 1:
        x = x[None, :]

    if positions is None:
        seq_len = x.shape[-2] if x.ndim >= 2 else 1
        positions = np.arange(seq_len)

    if x.ndim == 2:
        seq_len, d = x.shape
        freqs = np.arange(0, d, 2).astype(np.float64)
        inv_freq = 1.0 / (base ** (freqs / d))
        angles = positions[:, None] * inv_freq[None, :]
        cos = np.cos(angles)
        sin = np.sin(angles)

        x_even = x[:, ::2]
        x_odd = x[:, 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return np.stack([out_even, out_odd], axis=-1).reshape(seq_len, d)

    if x.ndim == 3:
        batch, seq_len, d = x.shape
        freqs = np.arange(0, d, 2).astype(np.float64)
        inv_freq = 1.0 / (base ** (freqs / d))
        angles = positions[None, :, None] * inv_freq[None, None, :]
        cos = np.cos(angles)
        sin = np.sin(angles)

        x_even = x[:, :, ::2]
        x_odd = x[:, :, 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return np.stack([out_even, out_odd], axis=-1).reshape(batch, seq_len, d)

    raise ValueError("Input must be 2D or 3D.")
