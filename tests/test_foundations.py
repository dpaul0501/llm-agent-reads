import numpy as np

from llm_foundations.embedding import TokenEmbedding
from llm_foundations.math_utils import softmax
from llm_foundations.positional_encoding import sinusoidal_positions
from llm_foundations.rope import apply_rope, rotate_half
from llm_foundations.self_attention import scaled_dot_product_attention
from llm_foundations.multi_head_attention import MultiHeadAttention
from llm_foundations.inference import KVCache
from llm_foundations.decoder_block import DecoderBlock, LayerNorm
from llm_foundations.lstm import LSTMCell


def test_softmax_basic():
    x = np.array([1.0, 2.0, 3.0])
    y = softmax(x)
    assert np.allclose(y.sum(), 1.0)
    assert np.all(y >= 0)


def test_sinusoidal_positions_shape():
    pe = sinusoidal_positions(4, 8)
    assert pe.shape == (4, 8)


def test_rope_preserves_shape():
    x = np.arange(8, dtype=np.float64).reshape(1, 8)
    y = apply_rope(x)
    assert y.shape == x.shape


def test_rope_relative_offset_and_pair_rotation():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 0.5, 3.0]])
    assert np.array_equal(rotate_half(x[0]), np.array([-2.0, 1.0, -4.0, 3.0]))
    first = apply_rope(x, positions=np.array([2, 5]))
    shifted = apply_rope(x, positions=np.array([9, 12]))
    assert np.allclose(first[0] @ first[1], shifted[0] @ shifted[1])


def test_self_attention_shape():
    q = np.random.randn(2, 4, 6)
    k = np.random.randn(2, 4, 6)
    v = np.random.randn(2, 4, 6)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 6)


def test_multi_head_attention_shape():
    x = np.random.randn(2, 4, 8)
    m = MultiHeadAttention(d_model=8, num_heads=2)
    out = m(x)
    assert out.shape == x.shape


def test_token_embedding_shape():
    emb = TokenEmbedding(vocab_size=10, d_model=6)
    x = np.array([0, 1, 3])
    out = emb(x)
    assert out.shape == (3, 6)


def test_kv_cache_updates_and_shape():
    cache = KVCache(max_len=4, head_dim=2, num_heads=2)
    k = np.ones((1, 2, 1, 2))
    v = np.ones((1, 2, 1, 2)) * 2
    cache.update(k, v)
    k_out, v_out = cache.get()
    assert k_out.shape == (1, 2, 1, 2)
    assert v_out.shape == (1, 2, 1, 2)


def test_layer_norm_normalizes_each_token():
    x = np.array([[[1.0, 2.0, 3.0, 4.0], [10.0, 12.0, 14.0, 16.0]]])
    y = LayerNorm(4)(x)
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-7)
    assert np.allclose(y.var(axis=-1), 1.0, atol=1e-4)


def test_decoder_block_preserves_residual_shape_and_causality():
    block = DecoderBlock(d_model=8, num_heads=2, d_ff=16)
    x = np.random.default_rng(1).normal(size=(2, 4, 8))
    first = block(x)
    changed_future = x.copy()
    changed_future[:, 3, :] += 100.0
    second = block(changed_future)
    assert first.shape == x.shape
    assert np.allclose(first[:, :3, :], second[:, :3, :])


def test_lstm_sequence_matches_repeated_cell_steps():
    cell = LSTMCell(d_input=3, d_hidden=2)
    x = np.random.default_rng(2).normal(size=(1, 4, 3))
    outputs, final_state = cell.forward_sequence(x)
    h = np.zeros((1, 2))
    c = np.zeros((1, 2))
    manual = []
    for t in range(4):
        h, c = cell(x[:, t, :], (h, c))
        manual.append(h.copy())
    assert np.allclose(outputs, np.stack(manual, axis=1))
    assert np.allclose(final_state[0], h)
    assert np.allclose(final_state[1], c)
