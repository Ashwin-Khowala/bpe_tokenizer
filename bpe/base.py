import unicodedata
import regex as re

def render_token(t: bytes) -> str:
    s = t.decode("utf-8", errors="replace")
    chars = []

    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")

    return "".join(chars)

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """

    counts = counts if counts is not None else {}
    for pair in zip(ids,ids[1:]):
        counts[pair] = counts.get(pair,0)+1
    return counts

def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """

    newids=[]
    i=0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i+=2
        else:
            newids.append(ids[i])
            i+=1
    return newids

class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.special_tokens = {} 
        self.pattern = ""
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        return NotImplementedError

    def encode(self, text):
        return NotImplementedError

    def decode(self, ids):
        return NotImplementedError

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        return vocab

    
    def save(self, file_prefix):
        # 1. Save the model needed to reconstruct the tokenizer.
        model_file = file_prefix + ".model"

        with open(model_file, "w", encoding="utf-8") as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")

            # Save special tokens.
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")

            # Save merges in the order they were learned.
            for (p0, p1), idx in sorted(
                self.merges.items(), key=lambda item: item[1]
            ):
                f.write(f"{p0} {p1}\n")

        # 2. Save a human-readable vocabulary file.
        vocab_file = file_prefix + ".vocab"

        inverted_merges = {
            idx: pair for pair, idx in self.merges.items()
        }

        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in sorted(self.vocab.items()):
                s = render_token(token)

                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")


    def load(self, model_file):
        assert model_file.endswith(".model")

        merges = {}
        special_tokens = {}

        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "minbpe v1", f"Unsupported version: {version}"

            self.pattern = f.readline().rstrip("\n")

            num_special = int(f.readline().strip())

            for _ in range(num_special):
                special, idx = f.readline().rstrip("\n").rsplit(" ", 1)
                special_tokens[special] = int(idx)

            # Reconstruct merge IDs in learned order.
            idx = 256
            for line in f:
                p0, p1 = map(int, line.split())
                merges[(p0, p1)] = idx
                idx += 1

        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

        # Rebuild regex-specific state if this tokenizer uses it.
        if hasattr(self, "compiled_pattern"):
            self.compiled_pattern = re.compile(self.pattern)

        # Rebuild the reverse special-token mapping if present.
        if hasattr(self, "inverse_special_tokens"):
            self.inverse_special_tokens = {
                idx: special for special, idx in self.special_tokens.items()
            }

