"""
Training pipeline:

    documents -> train/validation split -> train BPE tokenizer -> encode to token ids
    -> slice into (input, target) windows -> train FinGPT -> save best weights + history

Usage:
    python train.py                    # full run (needs internet for Financial PhraseBank)
    python train.py --epochs 2 --tiny  # quick smoke test on a CPU
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data.corpus import get_documents
from model import Config, FinGPT, FinTokenizer


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def encode_documents(docs, tok):
    """Concatenate all documents into one long token stream, with <eos> after each."""
    ids = []
    for d in docs:
        ids.extend(tok.encode(d))
        ids.append(tok.eos_id)
    return np.array(ids, dtype=np.int32)


def make_windows(ids, seq_len, stride):
    """
    Next-token prediction pairs. For the window ids[i : i+seq_len+1]:
        input  = ids[i     : i+seq_len]
        target = ids[i + 1 : i+seq_len+1]   (the same tokens shifted left by one)
    so at every position the model learns to predict the token that comes next.
    """
    starts = range(0, len(ids) - seq_len - 1, stride)
    x = np.stack([ids[s:s + seq_len] for s in starts])
    y = np.stack([ids[s + 1:s + seq_len + 1] for s in starts])
    return x, y


def to_dataset(x, y, batch_size, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup, then cosine decay
# ---------------------------------------------------------------------------
class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, min_lr, warmup_steps, total_steps):
        self.peak_lr, self.min_lr = peak_lr, min_lr
        self.warmup_steps, self.total_steps = warmup_steps, total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.peak_lr * step / max(self.warmup_steps, 1)
        progress = tf.clip_by_value(
            (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1), 0.0, 1.0)
        cosine = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + tf.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, warm, cosine)

    def get_config(self):
        return dict(peak_lr=self.peak_lr, min_lr=self.min_lr,
                    warmup_steps=self.warmup_steps, total_steps=self.total_steps)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class LiveHistory(keras.callbacks.Callback):
    """
    Writes training_history.json after EVERY epoch. The Flask dashboard polls this file,
    so the Chart.js graphs update live while training is still running.
    """

    def __init__(self, path, meta):
        super().__init__()
        self.path, self.meta = path, meta
        self.hist = {k: [] for k in ("loss", "val_loss", "accuracy", "val_accuracy",
                                     "perplexity", "val_perplexity", "lr", "epoch_time")}
        self.t0 = None

    def _write(self, status):
        with open(self.path, "w") as f:
            json.dump({"status": status, **self.meta, **self.hist}, f, indent=2)

    def on_train_begin(self, logs=None):
        self._write("training")

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in ("loss", "val_loss", "accuracy", "val_accuracy"):
            self.hist[k].append(float(logs.get(k, float("nan"))))
        # Perplexity = e^loss. Roughly "how many tokens is the model choosing between".
        self.hist["perplexity"].append(math.exp(min(logs["loss"], 20)))
        self.hist["val_perplexity"].append(math.exp(min(logs["val_loss"], 20)))
        lr = self.model.optimizer.learning_rate
        self.hist["lr"].append(float(lr.numpy() if hasattr(lr, "numpy") else lr))
        self.hist["epoch_time"].append(time.time() - self.t0)
        self._write("training")

    def on_train_end(self, logs=None):
        self._write("done")


class PrintSamples(keras.callbacks.Callback):
    """Print a couple of generations every few epochs so you can watch the model improve."""

    def __init__(self, tokenizer, every=5, prompts=("The company said", "Shares of")):
        super().__init__()
        self.tok, self.every, self.prompts = tokenizer, every, prompts

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every:
            return
        from generate import Generator
        gen = Generator(self.model, self.tok)
        print(f"\n--- samples after epoch {epoch + 1} ---")
        for p in self.prompts:
            print(" >", gen.generate(p, max_new_tokens=40))
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--no-phrasebank", action="store_true", help="embedded text only (offline test)")
    ap.add_argument("--tiny", action="store_true", help="very small model for a quick CPU test")
    args = ap.parse_args()

    cfg = Config()
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.learning_rate = args.lr
    if args.tiny:
        cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff = 64, 4, 2, 256
        cfg.seq_len, cfg.stride, cfg.vocab_size, cfg.warmup_steps = 64, 32, 1000, 20

    random.seed(42); np.random.seed(42); tf.random.set_seed(42)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # 1. Documents, split BEFORE tokenizing. Splitting whole documents (not overlapping windows)
    #    guarantees no validation sentence ever appears in the training data.
    docs = get_documents(use_phrasebank=not args.no_phrasebank)
    random.shuffle(docs)
    n_val = max(1, int(len(docs) * cfg.val_fraction))
    val_docs, train_docs = docs[:n_val], docs[n_val:]

    # 2. Tokenizer — trained on training documents only.
    tok = FinTokenizer.train(train_docs, vocab_size=cfg.vocab_size)
    cfg.vocab_size = tok.vocab_size
    tok.save(os.path.join(cfg.checkpoint_dir, "tokenizer.json"))
    sample = "Operating profit rose to EUR 13.1 mn."
    print(f"[tokenizer] vocab={tok.vocab_size}  '{sample}' -> {len(tok.encode(sample))} tokens")

    # 3. Encode and window.
    train_ids, val_ids = encode_documents(train_docs, tok), encode_documents(val_docs, tok)
    x_tr, y_tr = make_windows(train_ids, cfg.seq_len, cfg.stride)
    x_va, y_va = make_windows(val_ids, cfg.seq_len, cfg.seq_len)   # no overlap for validation
    print(f"[data] train tokens={len(train_ids):,} windows={len(x_tr)} | "
          f"val tokens={len(val_ids):,} windows={len(x_va)}")
    train_ds = to_dataset(x_tr, y_tr, cfg.batch_size, shuffle=True)
    val_ds = to_dataset(x_va, y_va, cfg.batch_size, shuffle=False)

    # 4. Model.
    model = FinGPT(cfg).build_model()
    print(f"[model] {cfg.n_layers} layers, {cfg.n_heads} heads, d_model={cfg.d_model}, "
          f"params={model.num_params():,}")
    cfg.save(os.path.join(cfg.checkpoint_dir, "config.json"))

    steps_per_epoch = math.ceil(len(x_tr) / cfg.batch_size)
    schedule = WarmupCosine(cfg.learning_rate, cfg.min_lr, cfg.warmup_steps,
                            steps_per_epoch * cfg.epochs)
    model.compile(
        optimizer=keras.optimizers.AdamW(schedule, weight_decay=cfg.weight_decay, clipnorm=1.0),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    meta = {"config": {k: v for k, v in vars(cfg).items()}, "params": model.num_params(),
            "train_tokens": int(len(train_ids)), "val_tokens": int(len(val_ids))}
    callbacks = [
        LiveHistory(cfg.history_path, meta),
        keras.callbacks.ModelCheckpoint(os.path.join(cfg.checkpoint_dir, "best.weights.h5"),
                                        monitor="val_loss", save_best_only=True,
                                        save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience),
        PrintSamples(tok, every=5),
    ]

    # 5. Train.
    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=callbacks)

    # 6. Re-save the best checkpoint WITHOUT the optimizer state (Adam keeps 2 extra numbers per
    #    weight). Only the model weights are needed for generation: ~70 MB -> ~23 MB.
    best_path = os.path.join(cfg.checkpoint_dir, "best.weights.h5")
    slim = FinGPT(cfg).build_model()
    slim.load_weights(best_path)
    slim.save_weights(best_path)
    print(f"[done] {time.time() - t0:.0f}s. Best weights: {best_path} "
          f"({os.path.getsize(best_path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
