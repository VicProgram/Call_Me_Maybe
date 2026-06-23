import json
from llm_sdk import Small_LLM_Model


class VocabIndex:
    def __init__(self, model: Small_LLM_Model) -> None:
        vocab_path = model.get_path_to_vocab_file()

        with open(vocab_path, "r", encoding="utf-8") as f:
            raw: dict[str, str] = json.load(f)

            self.id_to_token: dict[int, str] = {
                int(v): k for k, v in raw.items()
            }

            self.token_to_ids: dict[str, list[int]] = {}
            for token_id, token_str in self.id_to_token.items():
                if token_str not in self.token_to_ids:
                    self.token_to_ids[token_str] = []
                self.token_to_ids[token_str].append(token_id)

            self.vocab_size = len(self.id_to_token)
            self.all_ids = set(self.id_to_token.keys())

    def get_ids_start_with(self, prefix: str) -> list[int]:

        res = []

        for token_str, ids in self.token_to_ids.items():
            if token_str.startswith(prefix):
                res.extend(ids)
        return res
    

    def get_ids_exact(self, text: str) -> list[int]:
        return self.token_to_ids.get(text, [])
    
    def get_ids_chars(self, char: str) -> list[int]:
        res = []

        for token_str, ids in self.token_to_ids.items():
            if token_str and all(c in char for c in token_str):
                res.extend(ids)
        return res
    


