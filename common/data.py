"""Loading the training corpus and slicing it into train/val splits."""

from pathlib import Path

from common.tokenizer import CharTokenizer


def load_corpus(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Training corpus not found at {path}. Point --data at a plain-text "
            f"file (any UTF-8 text works for this char-level model)."
        )
    return path.read_text(encoding="utf-8")


def train_val_split(ids, val_fraction: float = 0.1):
    n = int((1 - val_fraction) * len(ids))
    return ids[:n], ids[n:]


def prepare(path: str | Path, val_fraction: float = 0.1):
    """Convenience wrapper: load text, build tokenizer, encode, split."""
    text = load_corpus(path)
    tokenizer = CharTokenizer.from_text(text)
    ids = tokenizer.encode(text)
    train_ids, val_ids = train_val_split(ids, val_fraction)
    return tokenizer, train_ids, val_ids
