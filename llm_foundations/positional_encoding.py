import numpy as np


def sinusoidal_positions(seq_len: int, d_model: int) -> np.ndarray:
    """Compute sinusoidal positional encodings for a sequence.

    Shape: (seq_len, d_model)
    """
    if d_model % 2 != 0:
        raise ValueError("d_model should be even for sinusoidal positional encodings.")

    positions = np.arange(seq_len)[:, None]
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    )

    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    return pe


def add_positional_encoding(x: np.ndarray, positions: np.ndarray | None = None) -> np.ndarray:
    """Add a sinusoidal positional embedding to an input tensor.

    x: shape (batch, seq_len, d_model) or (seq_len, d_model)
    """
    x = np.asarray(x, dtype=np.float64)
    if positions is None:
        positions = sinusoidal_positions(x.shape[-2], x.shape[-1])

    if x.ndim == 2:
        return x + positions
    if x.ndim == 3:
        return x + positions[None, :, :]
    raise ValueError("Input must be 2D or 3D.")
