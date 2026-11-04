"""
Text generation (inference) for FinGPT.

Autoregressive generation: feed the prompt, take the logits for the LAST position,
turn them into probabilities, sample one token, append it, repeat.

Usage:
    python generate.py --prompt "Operating profit rose"
"""
import argparse
import os

import numpy as np
import tensorflow as tf

from model import Config, FinGPT, FinTokenizer


class Generator:
    def __init__(self, model: FinGPT, tokenizer: FinTokenizer):
        self.model = model
        self.tok = tokenizer
        self.seq_len = model.cfg.seq_len

        # Compile the forward pass once. We always feed a FIXED shape (1, seq_len), padding on
        # the right. That's safe because of the causal mask: padding *after* position t can never
        # affect the prediction at position t. A fixed shape means TensorFlow traces the graph
        # only once, which makes generation much faster than calling the model eagerly.
        @tf.function(input_signature=[tf.TensorSpec((1, self.seq_len), tf.int32)])
        def forward(ids):
            return model(ids, training=False)[0]   # (seq_len, vocab)

        self._forward = forward

    def _next_token(self, ids, temperature, top_k, top_p):
        context = ids[-self.seq_len:]                       # keep only the last seq_len tokens
        padded = np.zeros((1, self.seq_len), dtype=np.int32)
        padded[0, :len(context)] = context
        logits = self._forward(padded)[len(context) - 1].numpy().astype(np.float64)

        # Temperature: <1 makes the distribution sharper (safer), >1 flatter (more creative).
        logits /= max(temperature, 1e-6)

        # Top-k: keep only the k most likely tokens.
        if top_k > 0:
            kth = np.sort(logits)[-top_k]
            logits[logits < kth] = -np.inf

        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        # Top-p (nucleus): keep the smallest set of tokens whose probabilities add up to p.
        if top_p < 1.0:
            order = np.argsort(probs)[::-1]
            cumulative = np.cumsum(probs[order])
            cutoff = order[cumulative > top_p][1:]       # always keep at least one token
            probs[cutoff] = 0
            probs /= probs.sum()

        return int(np.random.choice(len(probs), p=probs))

    def stream(self, prompt, max_new_tokens=120, temperature=0.8, top_k=40, top_p=0.95,
               stop_at_eos=True):
        """Yield the generated text piece by piece (used by the web app for live output)."""
        ids = self.tok.encode(prompt)
        if not ids:
            ids = [self.tok.eos_id]
        text_so_far = self.tok.decode(ids)
        for _ in range(max_new_tokens):
            nxt = self._next_token(ids, temperature, top_k, top_p)
            if stop_at_eos and nxt == self.tok.eos_id:
                break
            ids.append(nxt)
            # Decode the whole sequence and send only the new part. Decoding token by token
            # can break multi-byte characters, so this is the safe way to stream.
            full = self.tok.decode(ids)
            yield full[len(text_so_far):]
            text_so_far = full

    def generate(self, prompt, **kwargs):
        return prompt + "".join(self.stream(prompt, **kwargs))


def load_generator(checkpoint_dir="checkpoints"):
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    tok_path = os.path.join(checkpoint_dir, "tokenizer.json")
    w_path = os.path.join(checkpoint_dir, "best.weights.h5")
    for p in (cfg_path, tok_path, w_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found — run train.py first.")
    cfg = Config.load(cfg_path)
    tok = FinTokenizer.load(tok_path)
    model = FinGPT(cfg).build_model()
    model.load_weights(w_path)
    return Generator(model, tok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The stock market")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    args = ap.parse_args()

    gen = load_generator(args.checkpoint_dir)
    print(args.prompt, end="", flush=True)
    for piece in gen.stream(args.prompt, args.max_tokens, args.temperature, args.top_k, args.top_p):
        print(piece, end="", flush=True)
    print()
