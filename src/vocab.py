import json


class VocabIndex:
    def __init__(self, json_path: str):

        self.json_path = json_path
        data = self.load_json(self.json_path)
        # key_type = self.detect_key(data)
        self.create_dict(data, key_type)

    def load_json(self, json_path: str):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        except Exception:
            raise FileNotFoundError("JSON not founded")

        return data

    # def detect_key(self, data: dict):

    #     first_key = next(iter(data))

    #     try:
    #         int(first_key)
    #         return True

    #     except (ValueError, TypeError):
    #         return False

    def create_dict(self, data: dict, key_type: bool):

        # if key_type:
        self.token_to_id = data
        self.id_to_token = {v: k for k, v in data.items()}

        # else:
        #     self.id_to_token = data
        #     self.token_to_id = {v: k for k, v in data.items()}

    def search_exact(self, token: str):

        token_id = self.token_to_id.get(token)

        if token_id is not None:
            return [token_id]

        return []

    def search_prefix(self, prefix: str):

        valid_tokens = []

        for token in self.token_to_id:
            if token.startswith(prefix):
                valid_tokens.append(self.token_to_id[token])

        return valid_tokens

    def search_characters(self, valid_chars: str):

        token_chars = []

        for token in self.token_to_id:
            if all(c in valid_chars for c in token):
                token_chars.append(self.token_to_id[token])

        return token_chars
