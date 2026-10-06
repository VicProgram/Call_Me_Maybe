import json
from typing import Any

class VocabIndex:
    def __init__(self, json_path: str):

        self.json_path = json_path
        data = self.load_json(self.json_path)
        # key_type = self.detect_key(data)
        # self.create_dict(data, key_type)
        self.create_dict(data)

    def load_json(self, json_path: str) -> dict[str, Any]:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        except Exception:
            raise FileNotFoundError("JSON not founded")

        return data

    # region
    # def detect_key(self, data: dict):

    #     first_key = next(iter(data))

    #     try:
    #         int(first_key)
    #         return True

    #     except (ValueError, TypeError):
    #         return False
    # endregion

    # def create_dict(self, data: dict, key_type: bool):
    def create_dict(self, data: dict) -> None:

        # if key_type:
        self.token_to_id = data
        self.id_to_token = {v: k for k, v in data.items()}

        # else:
        #     self.id_to_token = data
        #     self.token_to_id = {v: k for k, v in data.items()}

    def search_exact(self, token: str) -> list[int]:

        token_id = self.token_to_id.get(token)

        if token_id is not None:
            return [token_id]

        return []

    def search_prefix(self, prefix: str) -> list[int]:

        valid_tokens = []

        for token in self.token_to_id:
            if token.startswith(prefix):
                valid_tokens.append(self.token_to_id[token])

        return valid_tokens

    def search_characters(self, valid_chars: str) -> list[int]:

        token_chars = []

        for token in self.token_to_id:
            if all(c in valid_chars for c in token):
                token_chars.append(self.token_to_id[token])

        return token_chars
