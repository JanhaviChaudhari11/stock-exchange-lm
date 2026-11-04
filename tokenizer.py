"""
Custom byte-level BPE tokenizer, trained on our own financial corpus.

Why byte-level BPE (the same family GPT-2 uses)?
  * Character-level tokens make sequences very long, so the model sees less text per window.
  * Word-level tokens can't handle new words ("EBITDA-adjusted") -> <unk>.
  * BPE starts from bytes and repeatedly merges the most frequent adjacent pair
    ("p"+"r" -> "pr", "pr"+"ofit" -> "profit"), so common financial words become one token
    and rare words are split into pieces. Starting from bytes means there is never an
    unknown character, and decoding gives back exactly the original text (spaces included).
"""
import os
from typing import List

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<eos>"]


class FinTokenizer:
    def __init__(self, tokenizer: Tokenizer = None):
        self.tk = tokenizer

    # ---------- training ----------
    @classmethod
    def train(cls, texts: List[str], vocab_size: int = 4000) -> "FinTokenizer":
        tk = Tokenizer(models.BPE())
        # Work on raw bytes. Spaces stay attached to the following word (shown as "Ġ"),
        # so " profit" is one token and the space survives decoding.
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        # The decoder turns byte tokens back into normal text. Without it you get "pro fit".
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tk.train_from_iterator(texts, trainer=trainer)
        return cls(tk)

    # ---------- encode / decode ----------
    def encode(self, text: str) -> List[int]:
        return self.tk.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    @property
    def eos_id(self) -> int:
        return self.tk.token_to_id("<eos>")

    # ---------- save / load ----------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.tk.save(path)

    @classmethod
    def load(cls, path: str) -> "FinTokenizer":
        return cls(Tokenizer.from_file(path))
