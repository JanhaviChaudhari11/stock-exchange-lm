# Stock Exchange Language Model

A small GPT-style (decoder-only) transformer built from scratch in TensorFlow/Keras and trained on financial text, with a Flask web app for real-time text generation and live training charts (Chart.js).

> Educational project. The generated text is not financial advice.

![Live generation](screenshots/demo-generation.png)

## Results

| Metric | Value |
|---|---|
| Parameters | 5.8M |
| Training data | 4,838 Financial PhraseBank sentences + 22 embedded paragraphs (~163k tokens) |
| Validation loss | 4.76 (from 8.29 at random initialisation) |
| Validation perplexity | ~117 (out of a 4,000-token vocabulary) |
| Validation next-token accuracy | 21.5% |
| Training time | ~3.5 min on a free Colab T4 GPU (30 epochs) |
| Generation speed | ~30–50 tokens/s |

Sample outputs:

- *Operating profit rose to EUR 3.5 mn from EUR 3.4 mn in the corresponding period in 2007.*
- *Net sales decreased by 5 % from EUR 5.3 mn in the corresponding period in 2005.*

The model learns the style and structure of financial news well, but not facts or arithmetic. It can write "decreased" next to a number that went up. That's expected at this scale and is why it must never be used for real decisions.

![Training charts](screenshots/demo-training-charts.png)

## How it works

1. **Data:** 4,838 financial news sentences from Financial PhraseBank (downloaded from HuggingFace) plus a short stock-market text corpus. 10% of sentences are held out for validation.
2. **Tokenizer:** a byte-level BPE tokenizer is trained on the training sentences. It learns 4,000 word-pieces such as " profit" and " EUR" and turns text into integer ids.
3. **Model:** the ids pass through token and position embeddings, then 6 transformer blocks. In each block, causal self-attention lets every token look only at earlier tokens, followed by a feed-forward layer. The output is a score for every token in the vocabulary.
4. **Training:** the model learns next-token prediction. At every position it predicts the following token, and the cross-entropy loss is minimised with AdamW, warmup and cosine decay, keeping the checkpoint with the best validation loss.
5. **Generation:** starting from a prompt, the model repeatedly predicts the next token, samples one using temperature, top-k and top-p, and appends it.
6. **Web app:** Flask streams each new token to the browser with Server-Sent Events, so text appears live. A Keras callback writes the training history after every epoch, and Chart.js polls it to draw the charts.

## What's inside

| Part | Details |
|---|---|
| Model | Decoder-only transformer, written layer by layer (no `keras.layers.MultiHeadAttention`) |
| Size | 6 layers · 8 heads · d_model 256 · FFN 1024 · context 128 tokens · ~5.8M parameters |
| Tricks | Pre-LayerNorm, GELU, causal masking, weight tying, dropout 0.1 |
| Tokenizer | Byte-level BPE (4,000 tokens) trained on this corpus with HuggingFace `tokenizers` |
| Data | Financial PhraseBank (~4,800 financial news sentences) + an embedded stock-market text corpus |
| Training | AdamW, warmup + cosine LR decay, gradient clipping, held-out validation set, early stopping |
| Sampling | Temperature, top-k and top-p (nucleus) sampling |
| Web app | Flask + Server-Sent Events (tokens stream in as they're generated), Chart.js dashboard that refreshes while training runs |

## Quick start

**Train on Google Colab (recommended):** open `train_on_colab.ipynb` in Colab, choose a T4 GPU runtime and run all cells.

**Or locally:**

```bash
pip install -r requirements.txt
python train.py                                   # train (≈15 min on a GPU)
python generate.py --prompt "Operating profit rose"
python app.py                                     # open http://localhost:5000
```

Quick CPU smoke test without downloading anything: `python train.py --tiny --no-phrasebank --epochs 3`

## Project layout

```
model/config.py       hyperparameters
model/tokenizer.py    byte-level BPE tokenizer
model/transformer.py  the transformer: attention, feed-forward, blocks, full model
data/corpus.py        loads Financial PhraseBank + embedded text
data/market_text.txt  embedded stock-market corpus
train.py              training pipeline (split, tokenize, window, train, log history)
generate.py           sampling / streaming generation + CLI
app.py                Flask server (SSE generation, history and model-info APIs)
templates/, static/   web UI and Chart.js dashboard
train_on_colab.ipynb  one-click training on a free Colab GPU
checkpoints/          trained weights, tokenizer, config and training history
```

## What I learned

<!-- Rewrite these in your own words before publishing. Keep only what is true for you. -->

- How causal masking turns a transformer into a text generator, and how the attention tensor shapes fit together across 8 heads.
- Why the validation split has to be made on whole sentences before cutting overlapping windows, so no validation text leaks into training.
- How to read training curves: loss, perplexity and the gap between train and validation as a sign of overfitting.
- Debugging a Flask 500 error by reading the server traceback (a missing `templates` folder).

## Limitations and future work

- **Small scale:** 5.8M parameters and ~160k tokens of text. The model learns the style of financial news but not facts or arithmetic.
- **More data:** add earnings-call transcripts or SEC filings to broaden the vocabulary beyond PhraseBank's company news.
- **Faster generation:** a KV cache would avoid recomputing attention over the whole context for every new token.
- **Sentiment control:** PhraseBank labels each sentence positive, negative or neutral. Prefixing sentences with their label would let the user ask for positive or negative news.
- **Modern components:** rotary position embeddings and a larger context window.

## Data credit

Financial PhraseBank: P. Malo, A. Sinha, P. Korhonen, J. Wallenius, P. Takala. *Good debt or bad debt: Detecting semantic orientations in economic texts.* JASIST, 2014. (CC BY-NC-SA 3.0)

---

Built by Janhavi Chaudhari · 

<!-- Replace the two links above with your own profile URLs. -->
