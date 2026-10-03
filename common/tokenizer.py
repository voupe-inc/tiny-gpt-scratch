"""Character-level tokenizer: the vocab is just the set of characters in the corpus."""

from dataclasses import dataclass


@dataclass
class CharTokenizer:
    stoi: dict
    itos: dict

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)
