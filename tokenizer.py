import json


class WordTokenizer:
    """
        A simple word tokenizer that converts text to indices and back.
    """
    def __init__(self, vocab_path: str) -> None:
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        self.word_to_index = vocab
        self.word_to_index["<pad>"] = len(self.word_to_index)  # Add padding token
        self.unk_token = "<unk>"

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert a string to a list of indices."""
        return [self.word_to_index.get(t, self.word_to_index[self.unk_token]) for t in tokens]

    def get_padding_index(self) -> int:
        """Get the index of the padding token."""
        return self.word_to_index["<pad>"]

    def __len__(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.word_to_index)
