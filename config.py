"""
All hyperparameters in one place.

The defaults are sized for a free Google Colab T4 GPU (~15 min of training).
"""
import json
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- model architecture ----
    vocab_size: int = 4000      # overwritten with the real size after the tokenizer is trained
    seq_len: int = 128          # context window: how many tokens the model can "see" at once
    d_model: int = 256          # size of each token's vector (embedding dimension)
    n_heads: int = 8            # attention heads; each head works on d_model / n_heads = 32 dims
    n_layers: int = 6           # number of stacked transformer blocks
    d_ff: int = 1024            # hidden size of the feed-forward network (4 x d_model, as in GPT)
    dropout: float = 0.1        # regularisation: randomly zero 10% of activations while training

    # ---- training ----
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 3e-4  # peak learning rate after warmup
    min_lr: float = 3e-5         # learning rate at the end of cosine decay
    warmup_steps: int = 200
    weight_decay: float = 0.01   # AdamW weight decay (another regulariser)
    val_fraction: float = 0.1    # last 10% of the token stream is held out for validation
    stride: int = 64             # step between training windows (seq_len // 2 = 50% overlap)
    patience: int = 4            # early stopping: stop if val loss doesn't improve for 4 epochs

    # ---- paths ----
    checkpoint_dir: str = "checkpoints"
    history_path: str = "checkpoints/training_history.json"

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(**json.load(f))
