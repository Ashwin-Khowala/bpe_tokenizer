import unittest

from bpe.regex_tokenizer import RegexTokenizer


class TestRegexTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = RegexTokenizer()

        training_text = (
            "Hello world! The quick brown fox jumps over the lazy dog. "
            "123456789. Testing punctuation, whitespace, and newlines.\n"
            "你好 😀 café"
        )

        self.tokenizer.train(training_text, vocab_size=300)

        self.tokenizer.register_special_tokens({
            "<|endoftext|>": 100257,
            "<|im_start|>": 100258,
            "<|im_end|>": 100259,
        })

    def test_ordinary_round_trip(self):
        cases = [
            "Hello world!",
            "你好 😀 café",
            "123456789",
            " spaces  and\nnewlines ",
            "",
        ]

        for text in cases:
            with self.subTest(text=text):
                ids = self.tokenizer.encode(text)
                self.assertEqual(self.tokenizer.decode(ids), text)

    def test_special_token_encoding(self):
        ids = self.tokenizer.encode(
            "<|endoftext|>",
            allowed_special="all",
        )
        self.assertEqual(ids, [100257])

    def test_multiple_special_tokens(self):
        text = "A<|im_start|>B<|im_end|>C<|endoftext|>"
        ids = self.tokenizer.encode(text, allowed_special="all")

        self.assertIn(100257, ids)
        self.assertIn(100258, ids)
        self.assertIn(100259, ids)
        self.assertEqual(self.tokenizer.decode(ids), text)

    def test_special_token_as_ordinary_text(self):
        text = "Hello<|endoftext|>world"
        ids = self.tokenizer.encode(text, allowed_special="none")

        self.assertNotIn(100257, ids)
        self.assertEqual(self.tokenizer.decode(ids), text)

    def test_none_raise_rejects_special_token(self):
        with self.assertRaises(AssertionError):
            self.tokenizer.encode(
                "<|endoftext|>",
                allowed_special="none_raise",
            )

    def test_invalid_token_id(self):
        with self.assertRaises(ValueError):
            self.tokenizer.decode([999999])

    def test_save_load(self):
        import tempfile
        from pathlib import Path

        training_text = "Hello world! This is a tokenizer test."

        tokenizer = RegexTokenizer()
        tokenizer.train(training_text, vocab_size=280)
        tokenizer.register_special_tokens({
            "<|endoftext|>": 100257,
        })

        test_text = "Hello world!<|endoftext|>"

        original_ids = tokenizer.encode(
            test_text,
            allowed_special="all",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = str(Path(tmpdir) / "tokenizer")
            tokenizer.save(prefix)

            loaded = RegexTokenizer()
            loaded.load(prefix + ".model")

            loaded_ids = loaded.encode(
                test_text,
                allowed_special="all",
            )

            self.assertEqual(loaded_ids, original_ids)
            self.assertEqual(loaded.decode(loaded_ids), test_text)


if __name__ == "__main__":
    unittest.main()