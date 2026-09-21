"""A vectorized LSTM cell and sequence forward pass written in NumPy."""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid for the small educational examples in this package."""
    positive = x >= 0
    out = np.empty_like(x, dtype=np.float64)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


class LSTMCell:
    """One forget-gate LSTM step.

    The packed gate order is forget, input, candidate, output. Inputs are
    batches of vectors: x is (B, d_input), h and c are (B, d_hidden).
    """

    def __init__(self, d_input: int, d_hidden: int, seed: int = 0):
        self.d_input = d_input
        self.d_hidden = d_hidden
        rng = np.random.default_rng(seed)
        self.W = rng.normal(
            0.0,
            1.0 / np.sqrt(d_input + d_hidden),
            size=(d_input + d_hidden, 4 * d_hidden),
        )
        self.b = np.zeros(4 * d_hidden)

    def __call__(
        self, x: np.ndarray, state: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        h_previous, c_previous = state
        if x.ndim != 2 or h_previous.ndim != 2 or c_previous.ndim != 2:
            raise ValueError("x, h_previous, and c_previous must be rank-2 batches")
        if x.shape[0] != h_previous.shape[0] or h_previous.shape != c_previous.shape:
            raise ValueError("batch sizes and recurrent-state shapes must agree")

        gates = np.concatenate([x, h_previous], axis=-1) @ self.W + self.b
        f_logit, i_logit, candidate_logit, o_logit = np.split(gates, 4, axis=-1)
        f = sigmoid(f_logit)
        i = sigmoid(i_logit)
        candidate = np.tanh(candidate_logit)
        o = sigmoid(o_logit)
        c = f * c_previous + i * candidate
        h = o * np.tanh(c)
        return h, c

    def forward_sequence(
        self,
        x: np.ndarray,
        state: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Run a batch-major sequence x with shape (B, T, d_input)."""
        if x.ndim != 3 or x.shape[-1] != self.d_input:
            raise ValueError("x must have shape (batch, time, d_input)")
        batch, time, _ = x.shape
        if state is None:
            h = np.zeros((batch, self.d_hidden))
            c = np.zeros((batch, self.d_hidden))
        else:
            h, c = state

        outputs = np.empty((batch, time, self.d_hidden))
        for t in range(time):
            h, c = self(x[:, t, :], (h, c))
            outputs[:, t, :] = h
        return outputs, (h, c)
