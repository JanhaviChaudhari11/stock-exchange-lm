"""
FinGPT — a GPT-style, decoder-only transformer written from scratch in TensorFlow/Keras.

Data flow for one batch (B = batch size, T = sequence length, D = d_model, V = vocab size):

    token ids (B, T)
      -> token embedding + position embedding          (B, T, D)
      -> N x TransformerBlock                           (B, T, D)
            x = x + CausalSelfAttention(LayerNorm(x))     "look at earlier tokens"
            x = x + FeedForward(LayerNorm(x))             "think about each token"
      -> final LayerNorm
      -> multiply by the embedding matrix (weight tying) (B, T, V)  = logits

The logits at position t are the model's guess for token t+1.
"""
import tensorflow as tf
from tensorflow import keras

from .config import Config


class CausalSelfAttention(keras.layers.Layer):
    """Multi-head self-attention where each token may only look at itself and earlier tokens."""

    def __init__(self, cfg: Config, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model
        # One dense layer produces Q, K and V together (3 x D outputs) — cheaper than three layers.
        self.qkv = keras.layers.Dense(3 * cfg.d_model, name="qkv")
        self.proj = keras.layers.Dense(cfg.d_model, name="proj")
        self.attn_drop = keras.layers.Dropout(cfg.dropout)
        self.resid_drop = keras.layers.Dropout(cfg.dropout)

    def _split_heads(self, x, B, T):
        # (B, T, D) -> (B, T, H, Dh) -> (B, H, T, Dh)
        x = tf.reshape(x, (B, T, self.n_heads, self.head_dim))
        return tf.transpose(x, (0, 2, 1, 3))

    def call(self, x, training=False):
        B, T = tf.shape(x)[0], tf.shape(x)[1]

        q, k, v = tf.split(self.qkv(x), 3, axis=-1)
        q, k, v = (self._split_heads(t, B, T) for t in (q, k, v))

        # Attention scores: how much should token i pay attention to token j?
        # Divide by sqrt(Dh) so the scores don't grow with head size (keeps softmax from saturating).
        scores = tf.matmul(q, k, transpose_b=True)                      # (B, H, T, T)
        scores /= tf.math.sqrt(tf.cast(self.head_dim, scores.dtype))

        # Causal mask: lower-triangular matrix of ones. Positions above the diagonal are the
        # "future" — we set them to a huge negative number so softmax gives them weight 0.
        mask = tf.linalg.band_part(tf.ones((T, T), dtype=tf.bool), -1, 0)
        scores = tf.where(mask, scores, tf.constant(-1e9, dtype=scores.dtype))

        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.attn_drop(weights, training=training)

        out = tf.matmul(weights, v)                                       # (B, H, T, Dh)
        out = tf.transpose(out, (0, 2, 1, 3))                             # (B, T, H, Dh)
        out = tf.reshape(out, (B, T, self.d_model))                       # concat heads
        return self.resid_drop(self.proj(out), training=training)


class FeedForward(keras.layers.Layer):
    """Two-layer MLP applied to every position independently: D -> 4D -> D."""

    def __init__(self, cfg: Config, **kwargs):
        super().__init__(**kwargs)
        self.up = keras.layers.Dense(cfg.d_ff, activation="gelu", name="up")
        self.down = keras.layers.Dense(cfg.d_model, name="down")
        self.drop = keras.layers.Dropout(cfg.dropout)

    def call(self, x, training=False):
        return self.drop(self.down(self.up(x)), training=training)


class TransformerBlock(keras.layers.Layer):
    """Pre-norm block (as in GPT-2): LayerNorm *before* each sub-layer, residual add after."""

    def __init__(self, cfg: Config, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.ff = FeedForward(cfg)

    def call(self, x, training=False):
        x = x + self.attn(self.ln1(x), training=training)   # residual connection 1
        x = x + self.ff(self.ln2(x), training=training)     # residual connection 2
        return x


class FinGPT(keras.Model):
    def __init__(self, cfg: Config, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.tok_emb = keras.layers.Embedding(cfg.vocab_size, cfg.d_model, name="tok_emb")
        self.pos_emb = keras.layers.Embedding(cfg.seq_len, cfg.d_model, name="pos_emb")
        self.drop = keras.layers.Dropout(cfg.dropout)
        self.blocks = [TransformerBlock(cfg, name=f"block_{i}") for i in range(cfg.n_layers)]
        self.ln_f = keras.layers.LayerNormalization(epsilon=1e-5, name="ln_f")

    def call(self, ids, training=False):
        T = tf.shape(ids)[1]
        x = self.tok_emb(ids) + self.pos_emb(tf.range(T))
        x = self.drop(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.ln_f(x)
        # Weight tying: reuse the input embedding matrix as the output layer.
        # Saves vocab_size x d_model parameters and usually trains better on small data.
        return tf.matmul(x, self.tok_emb.embeddings, transpose_b=True)  # (B, T, V) logits

    def build_model(self):
        """Run one dummy batch so Keras creates all the weights."""
        self(tf.zeros((1, self.cfg.seq_len), dtype=tf.int32))
        return self

    def num_params(self) -> int:
        return int(sum(tf.size(w).numpy() for w in self.trainable_weights))
