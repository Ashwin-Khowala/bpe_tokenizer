
from pathlib import Path

from bpe.regex_tokenizer import RegexTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "saved" / "swift_tokenizer.model"

# 1. Create a fresh tokenizer and load the saved model.
tokenizer = RegexTokenizer()
tokenizer.load(str(MODEL_PATH))

print("Tokenizer loaded successfully.")

# 2. Encode new text.
text = "Taylor Swift released a new album."

ids = tokenizer.encode(text)
print("Token IDs:", ids)

# 3. Decode the IDs back to text.
decoded = tokenizer.decode(ids)
print("Decoded:", decoded)

assert decoded == text
print("Round-trip test passed.")

# 4. Test a special token.
text_with_special = "Hello<|endoftext|>"

ids = tokenizer.encode(
    text_with_special,
    allowed_special="all",
)

print("Special-token IDs:", ids)
print("Decoded:", tokenizer.decode(ids))

assert tokenizer.decode(ids) == text_with_special
print("Special-token test passed.")
