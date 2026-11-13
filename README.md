# Parle: French→English Transformer from Scratch

An encoder–decoder Transformer for French→English translation on Multi30k, written in PyTorch **without** `nn.Transformer`, `nn.MultiheadAttention` or `F.scaled_dot_product_attention`. The repo also contains hand-written greedy and beam-search decoders, evaluated with corpus-level SacreBLEU.

| | |
|---|---|
| **Model** | 2 encoder + 2 decoder layers, d_model 128, 4 heads, FFN 128→512→128, 2,465,696 parameters |
| **Data** | Multi30k (29,000 train / 1,014 val / 1,000 test), French→English |
| **Tokenization** | Separate 4,000-piece BPE vocabularies (SentencePiece) for French and English |
| **Training** | 15 epochs, ~94 seconds on a Tesla T4 |
| **Validation perplexity** | 4.80 |
| **Test BLEU** | Greedy **49.33** → Beam search (width 3) **50.31** (SacreBLEU, 1,000 sentences) |
| **Decode cost** | Beam search took ~1.3× the time of greedy |

---

## Results

All numbers below come from the executed notebook in this repo (Colab, Tesla T4, seed 42).

### Training

| Metric | Value |
|---|---|
| Trainable parameters | 2,465,696 |
| Completed epochs | 15 |
| Training time | 93.8 s (1.56 min) |
| Best validation loss | 1.5693 (epoch 14) |
| Validation perplexity | 4.803 (`exp(mean validation loss)`) |

The checkpoint with the lowest validation loss is saved and used for all decoding. The test set is touched only after that choice.

### Greedy vs beam search (test set, 1,000 sentences)

| Method | Beam width | BLEU | Total decode time (s) |
|---|---|---|---|
| Greedy | 1 | 49.33 | 2.01 |
| Beam search | 3 | **50.31** | 2.66 |

Beam search took 1.32× the time of greedy decoding, with the same model, hardware, batch size (64) and maximum length (64).

- **SacreBLEU settings:** `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`
- **n-gram precisions (1/2/3/4):** greedy 75.4 / 55.6 / 42.8 / 33.2, beam 76.1 / 56.7 / 43.8 / 34.3
- Greedy and beam outputs differ on **383 of 1,000** sentences.
- Only generation is timed; model loading, text conversion and BLEU scoring are excluded, and `torch.cuda.synchronize()` is called before starting and stopping the timer.

### Run-to-run variation

Training on GPU is not bit-for-bit deterministic even with a fixed seed. An earlier run of the same code (same seed, also a T4) gave a best checkpoint at epoch 13 (val loss 1.5765) and test BLEU of **48.34 (greedy) → 49.90 (beam)** at 1.48× decode time. So the beam-search gain is consistently positive, roughly **+1 to +1.6 BLEU**, but exact numbers shift by a few tenths between runs. The table above is from the run saved in this repo's notebook. I have not run a multi-seed study.

### Example translations

| French | Reference | Greedy | Beam (3) |
|---|---|---|---|
| Des gens réparent le toit d'une maison. | People are fixing the roof of a house. | People are fixing the roof of a house. | People are fixing the roof of a house. |
| Trois personnes sont assises dans une grotte. | Three people sit in a cave. | Three people are sitting in a groke. | Three people are sitting in a cave. |
| Une fille avec une robe en jean marche sur une poutre d'équilibre surélevée. | A girl in a jean dress is walking along a raised balance beam. | A girl in a denim dress walks down a balance beam. | A girl in a denim dress is walking on a balance beam. |

The first ten test sentences, plus eight where greedy and beam disagree, are printed at the end of the notebook.

### Error analysis

The model is small (2.47M parameters, 29k training pairs), and its failures are the kind you would expect:

- **Rare words get garbled at the subword level.** "igloo" becomes "igoint" in both decoders, and "groke" appears for "cave" under greedy. With a 4,000-piece vocabulary, rare nouns are split into fragments the decoder can recombine into non-words.
- **Rare or ambiguous vocabulary produces fluent but wrong output.** "Une fille en tenue de karaté brisant un bâton" comes out as "A girl in a bath outfit reading a stick with a kick", and "alpinisme" (mountain climbing) becomes "racing around the track". The output is grammatical but the meaning is wrong.
- **Greedy can repeat or stall.** One greedy output contains "snow-covered snow-covered mountain" and another "but hands but hands". Beam search reduces but does not remove this.

---

## Architecture

Post-norm Transformer, following the original "Attention Is All You Need" layout at a much smaller scale.

| Component | Setting |
|---|---|
| Encoder / decoder layers | 2 / 2 |
| Embedding dimension (d_model) | 128 |
| Attention heads | 4 (32 features per head) |
| Feed-forward | Linear 128→512, ReLU, dropout, Linear 512→128 |
| Dropout | 0.1 |
| Positional encoding | Fixed sinusoidal, 512 positions |
| Normalization | LayerNorm after each residual addition (post-norm) |
| Embeddings | Separate French (source) and English (target); embeddings scaled by √d_model |
| Output layer | Linear d_model → 4,000 English tokens (not weight-tied) |
| Initialization | Xavier uniform on all matrices |

**Attention is implemented by hand:** the Q/K/V projections, scaled dot-product scores, masking with `-inf`, softmax, attention dropout, head concatenation and output projection are written explicitly. The same `MultiHeadAttention` module is used for encoder self-attention, decoder masked self-attention and decoder cross-attention (queries from the decoder, keys and values from the encoder output).

**Masking:**
- Source mask `[B, 1, 1, Ls]` hides padding tokens.
- Target mask `[B, 1, Lt, Lt]` combines the padding mask with a lower-triangular causal mask so the decoder cannot see future target tokens.

**Parameter count (2,465,696 total):**

| Part | Parameters |
|---|---|
| Source + target embeddings (2 × 4000 × 128) | 1,024,000 |
| Output projection (128 × 4000 + 4000) | 516,000 |
| Encoder layers (2 × 198,272) | 396,544 |
| Decoder layers (2 × 264,576) | 529,152 |

---

## Training setup

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight decay 0.01 |
| Peak learning rate | 1e-3 (chosen using validation loss) |
| Schedule | Linear warm-up for 400 steps, then linear decay to zero |
| Gradient clipping | 1.0 (global norm) |
| Batch size | 128 (train), 64 (val/test) |
| Epochs | 15 |
| Loss | Cross-entropy summed over non-padding tokens and divided by their count |
| Teacher forcing | Decoder input is the target shifted right by one |
| Model selection | Lowest validation loss, saved to `transformer_best.pt` |
| Seed | 42 (Python, NumPy, PyTorch, CUDA) |

Training pairs longer than 64 tokens are dropped by the dataset code; validation and test sentences are kept in full. A 30-minute time-limit guard stops training early if needed, but it was not triggered here.

---

## Decoding

Both decoders are written from scratch with tensor operations and run batched on the GPU. Neither uses a KV cache, so each step re-runs the decoder over the full prefix.

**Greedy decoding:** encode the French sentence once, then repeatedly pick the argmax token. PAD and BOS are never generated, and finished sentences are padded rather than extended. Stops at EOS or 64 generated tokens.

**Beam search (width 3):**
- Keeps the 3 best partial translations per sentence, scoring candidates by `sum(log p) / L^0.6`, where `L` counts generated tokens including EOS and excluding BOS/PAD.
- Finished beams are frozen: they can only continue with PAD at log-probability 0, so their score is unchanged and they are never extended.
- At the first step only beam 0 is alive, so the 3 beams do not start as duplicates.
- The best beam per sentence is chosen by its final length-normalized score.

---

## Data

- **Dataset:** Multi30k task 1 (Flickr image captions), French source → English target. Train 29,000, validation 1,014, test (`test_2016_flickr`) 1,000.
- **Download:** The notebook downloads the six text files from a fixed revision of [multi30k/dataset](https://github.com/multi30k/dataset) and verifies each against a SHA-256 hash, so everyone gets identical data. Files are cached locally after the first run.
- **Tokenizers:** Two SentencePiece BPE models (4,000 pieces each, character coverage 1.0), fitted on **training sentences only**. Special IDs: PAD=0, UNK=1, BOS=2, EOS=3.

---

## Run it

Needs Python 3.9+ and a GPU is recommended (the whole notebook runs in a few minutes on a Colab T4; it will run on CPU but training is much slower).

```bash
pip install torch sentencepiece==0.2.1 sacrebleu numpy matplotlib
jupyter notebook parle_transformer.ipynb
```

Then run the cells from top to bottom. The notebook needs internet on the first run to download Multi30k. It also installs `sentencepiece` and `sacrebleu` automatically if they are missing.

Files produced when you run it:
- `transformer_best.pt`: best checkpoint (by validation loss), including vocabulary sizes
- `french_src_tokenizer.model`, `english_tgt_tokenizer.model`: the two SentencePiece models
- `transformer_loss_curves.png`: train/validation loss plot

### Notebook layout

1. **Data**: download, hash-check, tokenizer training, datasets and loaders
2. **Model**: masks, multi-head attention, feed-forward, positional encoding, encoder/decoder layers, full Transformer, shape check
3. **Training**: training/validation loops, checkpointing, summary and loss curves
4. **Decoding and evaluation**: greedy and beam search, SacreBLEU, timing table, example translations, greedy-vs-beam disagreements
