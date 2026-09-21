# LLM Foundations from Scratch

This workspace is organized around the same intellectual arc used in the strongest modern LLM courses: build the core mathematical intuition first, then implement the mechanics from scratch.

The sequence is intentionally rigorous and course-like:

0. Original Transformer paper, math foundations, then tokenization
1. Embeddings and representation
2. Positional structure and RoPE
3. Attention and multi-head attention
4. A complete decoder block
5. Training objective and optimization
6. Generation and inference efficiency

The [LSTM appendix](docs/appendix_lstm.md) is optional background linked from the embeddings chapter.

The architecture page uses **Figure 1 extracted from the 2017 research paper** and explains which parts are absent in a decoder-only language model.

Beginner technical prerequisites are taught in the [foundation chapter](docs/module_01_foundations.md), including dot and outer products, tensor shapes, frequency, and rotation. [Tokenization](docs/part_00_tokenization.md) then covers BPE and input-target construction; [Part 2](docs/part_02_position_encoding.md) applies the wave and rotation math to position encoding.

## Reference stack

This repo synthesizes four widely used references:

- Stanford CS336: Language Modeling from Scratch
  - implementation-heavy, systems-aware, and closest to how real LLMs are actually built
  - https://cs336.stanford.edu/
- IIT Delhi ELL881 / AIL821 lecture series
  - mathematically grounded, conceptually deep, and structurally course-like
  - https://lcs2-iitd.github.io/ELL881-AIL821-2401/lectures/
- Andrej Karpathy’s published Zero to Hero lessons
  - code-first intuition for building a GPT and its tokenizer
  - https://karpathy.ai/zero-to-hero.html
  - LLM101n is an archived syllabus, not a completed course: https://github.com/karpathy/LLM101n
- Sebastian Raschka’s LLMs-from-scratch
  - rigorous PyTorch-based implementation path with step-by-step teaching
  - https://github.com/rasbt/LLMs-from-scratch

This project is not using only a subset of these. It is intentionally combining all four into one coherent learning trajectory.

## Core learning objective

The right mental model is not “LLMs are magic black boxes.”

The model is built from a sequence of explicit mathematical ideas:

- tokens are mapped to vectors
- position is encoded as structure
- attention uses similarity to route information
- multi-head attention creates multiple relational views
- inference is a sequential decoding problem with engineering optimizations

## Study plan

Foundation math
- vectors, matrix shapes, dot products, and outer products
- why the attention score table is $QK^\top$

Part 0: Tokenization and training examples
- text, bytes, vocabulary, BPE, and special tokens
- shifted input-target pairs

Part 1: Embeddings and representation learning
- vocabulary, tokenization, and lookup tables
- one-hot vs dense embeddings
- embedding geometry and semantic similarity
- why learned vector spaces matter

Optional LSTM appendix
- recurrence and backpropagation through time
- forget, input, and output gates with a worked two-step example
- why attention followed recurrent encoder-decoder models

Part 2: Positional encoding
- why order matters in language
- radians, angular frequency, period, and sine/cosine pairs
- complex numbers, Euler's formula, and rotations
- original additive encoding versus RoPE

Part 3: Attention and transformer mechanics
- Q, K, V decomposition
- softmax attention and weighted context retrieval
- causal masking for autoregressive generation
- multi-head attention and parallel relational views

Part 4: The complete decoder block
- attention, residuals, normalization, and MLP
- tensor shapes and parameter count

Part 5: Training objective and optimization
- next-token prediction as the core learning objective
- teacher forcing and masked training
- cross-entropy and gradient flow
- AdamW, learning rate schedules, and optimization dynamics
- when training is different from inference

Part 6: Inference-time systems and optimization
- sampling and greedy decoding
- KV cache, FlashAttention, and speculative decoding
- deployment trade-offs

## Repository layout

- `llm_foundations/` contains the implementation modules
- `docs/` contains the course-style notes and diagrams
- `tests/` contains correctness checks for the mathematical primitives

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m mkdocs build --strict
python -m mkdocs serve
```

## Rebuild the visual explanations

The [plot script](examples/plot_foundations.py) regenerates the SVG figures from NumPy and Matplotlib:

```bash
python examples/plot_foundations.py
node examples/check_labs.mjs
```

The site includes five interactive labs: BPE merges, positional frequency and rotation, attention with causal masking, softmax and loss, and LSTM gates. Each lab also has a standalone link beside its embedded version. The Node check exercises their arithmetic and controls without adding a browser runtime dependency.

## Present the course

Run `python -m mkdocs serve`, open `http://127.0.0.1:8000`, and navigate to the chapter you want to teach. Press **P** to enter the distraction-free presentation layout and browser full screen. Press **Esc** to return to the normal course navigation. Adding `?present=1` to a chapter URL opens it directly in the distraction-free layout.

The `main` branch also deploys automatically to GitHub Pages through `.github/workflows/pages.yml`, so the same presentation can be opened at `https://dpaul0501.github.io/llm-agent-reads/` after Pages is enabled for the repository.

## Mathematical focus

The project is centered on the actual structure of modern LMs:

- embeddings as learned geometry
- positional encodings as structured phase information
- attention as softmax-weighted retrieval over context
- multi-head attention as parallel specialization
- inference as sequential decoding under memory and latency constraints

This is the foundation on which transformer-based language models are built.
