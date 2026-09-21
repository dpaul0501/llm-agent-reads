# Course resources and reading map

Use primary sources for the architecture and implementation path, then use visual articles to clarify one mechanism at a time.

| Resource | Best use in this course | Link |
| --- | --- | --- |
| Vaswani et al., *Attention Is All You Need* | The original encoder-decoder diagram and 2017 design | [Paper PDF](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf) |
| Stanford CS336 | Tokenizer/model/optimizer build, systems, scaling, data, evaluation, and alignment | [2025 course and assignments](https://cs336.stanford.edu/spring2025/) |
| IIT Delhi ELL881/AIL821 | Sequence-model history, attention, positional encoding, normalization, and later LLM topics | [Lecture index](https://lcs2-iitd.github.io/ELL881-AIL821-2401/lectures/) |
| Karpathy Zero to Hero | Published code-first GPT and tokenizer lessons | [Lesson index](https://karpathy.ai/zero-to-hero.html) |
| Karpathy LLM101n | Broad planned syllabus; the repository says the course was not completed | [Archived outline](https://github.com/karpathy/LLM101n) |
| 3Blue1Brown, *Transformers, the tech behind LLMs* | Visual companion for the full token-to-logit path and embedding geometry | [Lesson](https://www.3blue1brown.com/lessons/gpt/) |
| 3Blue1Brown, *Attention in transformers, step-by-step* | Visual companion for query/key routing, value mixing, masking, and multiple heads | [Lesson](https://www.3blue1brown.com/lessons/attention/) |
| Raschka, *Build a Large Language Model (From Scratch)* | Text preparation through GPT implementation, pretraining, and fine-tuning | [Book companion and chapter map](https://www.sebastianraschka.com/llms-from-scratch/) |
| Jay Alammar, *The Illustrated Transformer* | Visual reading of the original architecture | [Article](https://jalammar.github.io/illustrated-transformer/) |
| Raschka, *Understanding and Coding Self-Attention* | A Substack example that connects equations to code | [Article](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) |
| Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* | Why and how subword BPE handles rare words | [ACL paper](https://aclanthology.org/P16-1162/) |
| Hochreiter and Schmidhuber, *Long Short-Term Memory* | Recurrent memory and gradient motivation | [Original paper](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory) |
| Gers et al., *Learning to Forget* | The forget gate used in standard LSTM explanations | [Paper record](https://pubmed.ncbi.nlm.nih.gov/11032042/) |
| Su et al., *RoFormer* | Rotary position and relative query-key geometry | [Paper](https://arxiv.org/abs/2104.09864) |

## Reference figures used in the chapters

| Chapter | Figure | Why it is included | Source and reuse note |
| --- | --- | --- | --- |
| Foundation and decoder block | Transformer architecture, Figure 1 | Original encoder-decoder architecture, residual paths, masked decoder attention, and output head | [Vaswani et al. paper](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf); extracted from the research paper |
| Attention | Scaled dot-product and multi-head attention, Figure 2 | Standard research-paper view of scaling, masking, softmax, value mixing, parallel heads, concatenation, and output projection | [Vaswani et al. paper](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf); extracted from the research paper |
| Embeddings | CBOW and Skip-gram, Figure 1 | Original Word2Vec paper's two directions of prediction | [Mikolov et al. paper](https://arxiv.org/abs/1301.3781); extracted from the research paper |
| LSTM appendix | Modern forget-gate LSTM cell | Standard gate-and-cell-state diagram matching the equations in the appendix | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:LSTM_cell.svg), Guillaume Chevalier and Ketograff, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The visual articles are selected examples. When comparing another article or lecture, use the same criteria: correct architecture attribution, explicit tensor shapes, runnable examples, and a complete path to loss and generation.
