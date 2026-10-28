
from pathlib import Path

from bpe.regex_tokenizer import RegexTokenizer

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset.txt"
SAVE_DIR = ROOT / "saved"
MODEL_PREFIX = SAVE_DIR / "swift_tokenizer"

# 1. Load the training dataset.
text = DATASET_PATH.read_text(encoding="utf-8")
print(f"Loaded {len(text):,} characters from {DATASET_PATH.name}")

# 2. Train the tokenizer.
tokenizer = RegexTokenizer()
tokenizer.train(text, vocab_size=1000)

# 3. Register special tokens.
base_id = len(tokenizer.vocab)

tokenizer.register_special_tokens({
    "<|endoftext|>": base_id,
    "<|im_start|>": base_id + 1,
    "<|im_end|>": base_id + 2,
})

# 4. Save the model.
SAVE_DIR.mkdir(parents=True, exist_ok=True)
tokenizer.save(str(MODEL_PREFIX))

print(f"Vocabulary size: {len(tokenizer.vocab)}")
print(f"Saved model: {MODEL_PREFIX}.model")
