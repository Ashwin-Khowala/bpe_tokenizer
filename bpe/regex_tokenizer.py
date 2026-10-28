import regex as re

from .base import Tokenizer, get_stats, merge

GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+"""
    r"""|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)

class RegexTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

        self.pattern = GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)

        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_chunks = self.compiled_pattern.findall(text)
        ids = [list(text_chunk.encode("utf-8")) for text_chunk in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            pairs = {}
            for chunks in ids:
                pairs = get_stats(chunks, pairs)

            if not pairs:
                raise ValueError("No mergeable pairs remain in the training text.")

            pair = max(pairs, key= pairs.get)
            newidx = i + 256 

            ids = [merge(chunks, pair, newidx) for chunks in ids]
            merges[pair] = newidx
            vocab[newidx] = vocab[pair[0]] + vocab[pair[1]]

        #used while encoding
        self.merges = merges
        #used while decoding 
        self.vocab = self._build_vocab()

    def decode(self, ids):
        # modified to handle special tokens
        part_bytes =[]
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
            
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode_ordinary(self, text):
        text_chunks = self.compiled_pattern.findall(text)
        ids = [list(text_chunk.encode("utf-8")) for text_chunk in text_chunks]

        while True:
            pairs = {}
            for chunks in ids:
                pairs = get_stats(chunks, pairs)

            if not pairs:
                break

            pair = min(pairs, key= lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            ids = [merge(chunks, pair, self.merges[pair]) for chunks in ids]

        return [idx for idx_chunks in ids for idx in idx_chunks]

    def encode(self, text, allowed_special="none_raise"):

        special = None
        if allowed_special =="all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        
        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids


    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens.copy()
        self.inverse_special_tokens = {idx: token for token, idx in self.special_tokens.items()}
        self.vocab = self._build_vocab()
