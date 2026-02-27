# Mini-Eng-to-Es-NMT

A Transformer-based English-to-Spanish neural machine translation system built entirely from scratch — no `nn.MultiheadAttention`, no pre-built transformer layers. Every component, from multi-head attention to beam search, is hand-implemented in PyTorch.

## What This Project Does

Takes an English sentence like *"they trusted tom"* and translates it to Spanish: *"ellos confiaban en tom"*. The model learns this mapping using the attention mechanism from "Attention Is All You Need" (Vaswani et al., 2017), trained on 50,000 sentence pairs from the [Anki Spanish-English dataset](https://huggingface.co/datasets/OscarNav/spa-eng).

## Architecture

A simplified single-layer Transformer encoder-decoder with `d_model=256` and 4 attention heads (64 dimensions per head). The encoder processes English tokens through self-attention, producing contextualized representations. The decoder generates Spanish tokens one at a time, using masked self-attention for coherence and cross-attention to look back at the English source — this Q-K-V bridge between languages is the core of the translation.

**Key components built from scratch**: Sinusoidal positional encoding, scaled dot-product multi-head attention, feed-forward networks with residual connections and layer normalization, padding masks, causal masks, greedy decoding, and beam search with length normalization.

## Ablation Study

Instead of training one model, we run six configurations to isolate what actually matters:

| Config | BLEU | spBLEU | chrF | What Changed |
|---|---|---|---|---|
| Baseline (random embeddings, standard CE) | 18.99 | 21.05 | 40.18 | — |
| + Label Smoothing (ε=0.1) | 18.77 | 21.41 | 40.61 | Loss function |
| + FastText (pre-trained 300d → 256d) | 30.72 | 32.89 | 54.45 | Embeddings |
| + BPE (8K joint vocab) | 18.15 | 20.11 | 40.04 | Tokenization |
| Full Enhanced (FastText + smoothing) | 34.88 | 36.64 | 57.33 | Combined |
| Full + Beam Search (k=3) | **36.44** | **38.08** | **58.57** | Inference |

**Key finding**: Pre-trained FastText embeddings are the dominant factor, nearly doubling BLEU from 19 to 31. Label smoothing alone barely moves the needle, but when combined with FastText it prevents overfitting and adds +4 BLEU. BPE eliminates unknown words entirely (0% vs 3% UNK rate) but doesn't improve BLEU under the 20-token length constraint. Beam search adds a clean +1.6 BLEU purely from better inference search.

## Project Structure

The notebook is organized into six phases that run sequentially:

- **Phase 0** — Environment setup, imports, reproducibility (seed=42)
- **Phase 1** — Data loading, 80/10/10 split, exploratory analysis
- **Phase 2** — Text cleaning, word-level tokenization (15K vocab), BPE tokenization (8K joint), DataLoaders with dynamic padding
- **Phase 3** — FastText embedding download (streamed, vocab-filtered to ~30MB), 300→256 projection
- **Phase 4** — Full Transformer architecture from scratch
- **Phase 5** — Training loop (20 epochs, Adam, gradient clipping, ReduceLROnPlateau), evaluation with sacrebleu metrics, experiment runner for all 6 configs
- **Phase 6** — Training curves, BLEU bar chart, qualitative translation examples

## Quick Start

Open the notebook in Google Colab (GPU runtime recommended), and run all cells. Total runtime is approximately 5 minutes for all six experiments on a modern GPU. Dependencies (`datasets`, `sacrebleu`, `fasttext`, `sentencepiece`) are installed in the first cell.

## Sample Translations

| English | Reference | Model (Full+Beam) |
|---|---|---|
| its my umbrella | es mi paraguas | es mi paraguas ✓ |
| they trusted tom | confiaban en tom | ellos confiaban en tom ✓ |
| many americans blamed spain | muchos americanos culparon a españa | muchos estadounidenses culpó de españa ≈ |

The model handles simple sentences well, struggles with complex morphology and longer dependencies — expected for a single-layer model on 50K pairs.

## Requirements

Python 3.10+, PyTorch 2.x, and a CUDA-capable GPU. All other dependencies are pip-installed within the notebook.

## Course Context

Built for the *Foundational Models and Generative AI* course, focusing on understanding cross-attention as the mechanism that bridges encoder and decoder in sequence-to-sequence translation.
