class VocabIndex:
    def __init__(self, vocab_path: str):
        self.vocab_path = vocab_path
        self.vocab = self.load_vocab()
        self.token_to_index = {token: index for index, token in enumerate(self.vocab)}
        self.index_to_token = {index: token for index, token in enumerate(self.vocab)}

    def load_vocab(self):
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f.readlines()]
        return vocab

    def token_to_id(self, token: str) -> int:
        return self.token_to_index.get(token, -1)

    def id_to_token(self, index: int) -> str:
        return self.index_to_token.get(index, "<UNK>")
