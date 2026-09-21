import numpy as np


class TokenEmbedding:
    """Simple token embedding layer using a learned lookup table."""

    def __init__(self, vocab_size: int, d_model: int, rng: np.random.Generator | None = None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.rng = rng or np.random.default_rng(0)
        self.weights = self.rng.normal(0.0, 1.0 / np.sqrt(d_model), size=(vocab_size, d_model))

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        token_ids = np.asarray(token_ids, dtype=np.int64)
        return self.weights[token_ids]
