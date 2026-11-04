"""
Builds the training corpus from two sources:

  1. Financial PhraseBank (Malo et al., 2014) from HuggingFace — ~4,800 real sentences
     from financial news, e.g. "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn".
     The dataset has sentiment labels; we ignore them because we're doing language
     modelling (predict the next token), not classification.
  2. A small embedded corpus (data/market_text.txt) of explanatory stock-market text,
     so the model also learns longer, textbook-style sentences.

Each document ends with an <eos> token so the model learns where a sentence stops.
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
EMBEDDED_PATH = os.path.join(HERE, "market_text.txt")
CACHE_PATH = os.path.join(HERE, "phrasebank_cache.txt")


def _load_phrasebank_via_datasets():
    """Newer `datasets` versions can't run loading scripts, so read the auto-converted parquet."""
    from datasets import load_dataset

    ds = load_dataset(
        "takala/financial_phrasebank",
        "sentences_50agree",
        revision="refs/convert/parquet",
        split="train",
    )
    return [row["sentence"] for row in ds]


def _load_phrasebank_via_zip():
    """Fallback: download the original zip from the HuggingFace repo and read the .txt file."""
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        "takala/financial_phrasebank",
        "data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith("Sentences_50Agree.txt"))
        raw = z.read(name).decode("latin-1")
    # Each line looks like: "sentence text@positive"
    return [line.rsplit("@", 1)[0].strip() for line in raw.splitlines() if "@" in line]


def load_phrasebank():
    """Return a list of PhraseBank sentences (cached to disk after the first download)."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            return [line for line in f.read().splitlines() if line.strip()]

    errors = []
    for loader in (_load_phrasebank_via_datasets, _load_phrasebank_via_zip):
        try:
            sentences = loader()
            break
        except Exception as e:  # try the next method
            errors.append(f"{loader.__name__}: {e}")
    else:
        # Fail loudly — training silently without PhraseBank would make the results misleading.
        raise RuntimeError(
            "Could not download Financial PhraseBank.\n  " + "\n  ".join(errors) +
            "\nCheck your internet connection, or run train.py --no-phrasebank for a quick test."
        )

    sentences = list(dict.fromkeys(s for s in sentences if s))  # de-duplicate, keep order
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))
    return sentences


def load_embedded():
    """Paragraphs of the embedded corpus (separated by blank lines)."""
    with open(EMBEDDED_PATH, encoding="utf-8") as f:
        return [p.strip().replace("\n", " ") for p in f.read().split("\n\n") if p.strip()]


def get_documents(use_phrasebank: bool = True):
    docs = load_embedded()
    print(f"[corpus] embedded text: {len(docs)} paragraphs")
    if use_phrasebank:
        pb = load_phrasebank()
        print(f"[corpus] Financial PhraseBank: {len(pb)} sentences")
        docs += pb
    print(f"[corpus] total: {len(docs)} documents, {sum(map(len, docs)):,} characters")
    return docs
