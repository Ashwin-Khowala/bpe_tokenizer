import regex as re
class Tokenizer:

    @staticmethod
    def get_stats(ids, pairs=None):
        pairs = {} if pairs is None else pairs
        for pair in zip(ids,ids[1:]):
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs
    
    @staticmethod
    def merge(ids, pair, newid):
        new_ids=[]
        i=0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(newid)
                i+=2
            else:
                new_ids.append(ids[i])
                i+=1
        return new_ids

    GPT4_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        compiled_pattern = re.compile(self.GPT4_SPLIT_PATTERN)
        text_chunks = compiled_pattern.findall(text)

        tokens = [list(ch.encode("utf-8")) for ch in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            pairs = {}
            for chunks in tokens:
                Tokenizer.get_stats(chunks, pairs)

            if not pairs:
                break

            max_pair = max(pairs, key=pairs.get)
            new_id = 256+i

            tokens = [Tokenizer.merge(chunks, max_pair, new_id) for chunks in tokens]

            merges[max_pair] = new_id   
            vocab[new_id] = vocab[max_pair[0]] + vocab[max_pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {max_pair} -> {new_id} ({vocab[new_id]}) had {pairs[max_pair]} occurrences")
        
        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        # tokens = list(text.encode("utf-8"))

        compiled_pattern = re.compile(self.GPT4_SPLIT_PATTERN)
        text_chunks = compiled_pattern.findall(text)
        chunks = [list(ch.encode("utf-8")) for ch in text_chunks]

        while True:
            pairs ={}
            for chunk_id in chunks:
                pairs = Tokenizer.get_stats(chunk_id,pairs)

            if not pairs:
                break   

            pair = min(pairs, key= lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break

            new_idx = self.merges[pair]
            chunks = [Tokenizer.merge(chunk_ids, pair, new_idx) for chunk_ids in chunks] 


        return [token_id for chunk_ids in chunks for token_id in chunk_ids]

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)   
        text = text_bytes.decode("utf-8", errors="replace")
        return text



tokenizer = Tokenizer()

with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

vocab_size = 500

# Train
tokenizer.train(text, vocab_size, verbose=True)

# Encode
ids = tokenizer.encode(text)
print("\nEncoded IDs:", ids)
print("Number of tokens:", len(ids))

# Decode
decoded = tokenizer.decode(ids)
print("Decoded text:", decoded)

# Verify
assert decoded == text, f"Round-trip failed: {decoded!r}"
print("Round-trip test passed!")