import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_foundations.math_utils import softmax
from llm_foundations.positional_encoding import sinusoidal_positions
from llm_foundations.rope import apply_rope
from llm_foundations.self_attention import scaled_dot_product_attention
from llm_foundations.multi_head_attention import MultiHeadAttention


def main() -> None:
    seq_len = 4
    d_model = 8

    pos = sinusoidal_positions(seq_len, d_model)
    print("Positional encoding shape:", pos.shape)

    x = np.arange(seq_len * d_model, dtype=np.float64).reshape(seq_len, d_model)
    rope_out = apply_rope(x)
    print("RoPE output shape:", rope_out.shape)

    q = np.random.randn(1, seq_len, d_model)
    k = np.random.randn(1, seq_len, d_model)
    v = np.random.randn(1, seq_len, d_model)
    attn = scaled_dot_product_attention(q, k, v)
    print("Self-attention output shape:", attn.shape)

    mha = MultiHeadAttention(d_model=d_model, num_heads=2)
    mha_out = mha(q)
    print("Multi-head attention output shape:", mha_out.shape)

    print("Example softmax:", softmax(np.array([1.0, 2.0, 3.0])))


if __name__ == "__main__":
    main()
