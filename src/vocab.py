import json
from typing import Any


class VocabIndex:
    """Index for fast token-to-id and id-to-token lookups.

    Loads a vocabulary JSON file and provides methods to search
    tokens by exact match, prefix, or character composition.

    Attributes:
        json_path: Path to the vocabulary JSON file.
        token_to_id: Dictionary mapping tokens to their IDs.
        id_to_token: Dictionary mapping IDs to their tokens.
    """

    def __init__(self, json_path: str):
        """Initialize the vocabulary index from a JSON file.

        Args:
            json_path: Path to the vocabulary JSON file.
        """
        self.json_path = json_path
        data = self.load_json(self.json_path)
        self.create_dict(data)

    def load_json(self, json_path: str) -> dict[str, Any]:
        """Load and parse a JSON file.

        Args:
            json_path: Path to the JSON file.

        Returns:
            The parsed JSON data as a dictionary.

        Raises:
            FileNotFoundError: If the file does not exist or cannot be parsed.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        except Exception:
            raise FileNotFoundError("JSON not founded")

        return data  # type: ignore[no-any-return]

    def create_dict(self, data: dict[str, Any]) -> None:
        """Create bidirectional token-to-id and id-to-token mappings.

        Args:
            data: The vocabulary data as a dictionary.
        """
        self.token_to_id = data
        self.id_to_token = {v: k for k, v in data.items()}

    def search_exact(self, token: str) -> list[int]:
        """Search for a token by exact match.

        Args:
            token: The token to search for.

        Returns:
            A list containing the token ID if found, otherwise an empty list.
        """
        token_id = self.token_to_id.get(token)

        if token_id is not None:
            return [token_id]

        return []

    def search_characters(self, valid_chars: str) -> list[int]:
        """Search for all tokens composed only of the given characters.

        Args:
            valid_chars: A string of allowed characters.

        Returns:
            A list of token IDs whose tokens are composed
            only of the allowed characters.
        """
        token_chars = []

        for token in self.token_to_id:
            if all(c in valid_chars for c in token):
                token_chars.append(self.token_to_id[token])

        return token_chars

    def search_any_character(self, valid_chars: str) -> list[int]:
        """Search for tokens containing any of the given characters.

        Args:
            valid_chars: A string of characters to look for.

        Returns:
            A list of token IDs whose tokens contain at least
            one of the allowed characters.
        """
        token_chars = []

        for token in self.token_to_id:
            if any(c in valid_chars for c in token):
                token_chars.append(self.token_to_id[token])

        return token_chars
