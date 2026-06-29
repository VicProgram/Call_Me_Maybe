import math
import numpy as np
from llm_sdk import Small_LLM_Model
from src.vocab import VocabIndex


def get_next_token_logits(
    model: Small_LLM_Model,
    input_ids: list[int]
) -> np.ndarray:
    raw_logits = model.get_logits_from_input_ids(input_ids)
    return np.array(raw_logits, dtype=np.float32)


def apply_mask(
    logits: np.ndarray,
    valid_ids: list[int]
) -> np.ndarray:
    masked = np.full(len(logits), -math.inf)
    for tid in valid_ids:
        if 0 <= tid < len(logits):
            masked[tid] = logits[tid]
    return masked


def select_best_token(masked_logits: np.ndarray) -> int | None:
    best = int(np.argmax(masked_logits))
    if masked_logits[best] == -math.inf:
        return None
    return best


class JSONGenerator:

    def __init__(self, model: Small_LLM_Model, vocab: VocabIndex) -> None:
        self.model = model
        self.vocab = vocab

    def select_function_name(
        self,
        prompt_ids: list[int],
        candidate_names: list[str]
    ) -> str:
        logits = get_next_token_logits(self.model, prompt_ids)
        log_probs = self._log_softmax(logits)

        best_name = candidate_names[0]
        best_score = -math.inf

        for name in candidate_names:
            name_token_ids = self.model.encode(name).flatten().tolist()
            score = self._score_sequence(prompt_ids, name_token_ids, log_probs)
            if score > best_score:
                best_score = score
                best_name = name

        return best_name

    def _log_softmax(self, logits: np.ndarray) -> np.ndarray:
        max_logit = np.max(logits)
        shifted = logits - max_logit
        log_sum_exp = np.log(np.sum(np.exp(shifted)))
        return shifted - log_sum_exp

    def _score_sequence(
        self,
        context_ids: list[int],
        target_ids: list[int],
        first_logprobs: np.ndarray
    ) -> float:
        if not target_ids:
            return -math.inf

        first_id = target_ids[0]
        if first_id >= len(first_logprobs):
            return -math.inf

        score = float(first_logprobs[first_id])

        current_ids = context_ids + [first_id]
        for next_target_id in target_ids[1:]:
            logits = get_next_token_logits(self.model, current_ids)
            log_probs = self._log_softmax(logits)
            if next_target_id < len(log_probs):
                score += float(log_probs[next_target_id])
            current_ids = current_ids + [next_target_id]

        return score

    def extract_number(
        self,
        prompt_ids: list[int],
    ) -> float:
        digit_ids = self.vocab.get_ids_chars("0123456789")
        dot_ids = self.vocab.get_ids_exact(".")
        minus_ids = self.vocab.get_ids_exact("-")

        accumulated = ""
        current_ids = list(prompt_ids)

        for _ in range(20):
            logits = get_next_token_logits(self.model, current_ids)

            if not accumulated:
                valid_ids = digit_ids + minus_ids
            elif accumulated == "-":
                valid_ids = digit_ids
            elif "." in accumulated:
                valid_ids = digit_ids
            else:
                valid_ids = digit_ids + dot_ids

            terminator_ids = (
                self.vocab.get_ids_exact(",")
                + self.vocab.get_ids_exact("}")
                + self.vocab.get_ids_exact("\n")
                + self.vocab.get_ids_exact(" ")
            )

            all_valid = valid_ids + terminator_ids
            masked = apply_mask(logits, all_valid)
            chosen_id = select_best_token(masked)

            if chosen_id is None:
                break

            token_str = self.vocab.id_to_token.get(chosen_id, "")

            if token_str in [",", "}", "\n", " "] and accumulated:
                break

            accumulated += token_str
            current_ids = current_ids + [chosen_id]

        try:
            return float(accumulated) if accumulated else 0.0
        except ValueError:
            return 0.0

    def extract_string(
        self,
        prompt_ids: list[int],
    ) -> str:
        blocked_tokens = {"{", "}", "[", "]"}
        valid_ids = [
            tid for tid, tstr in self.vocab.id_to_token.items()
            if tstr not in blocked_tokens
        ]

        end_tokens = {"\n", '",', '"}', '",\n'}

        accumulated = ""
        current_ids = list(prompt_ids)

        for _ in range(50):
            logits = get_next_token_logits(self.model, current_ids)
            masked = apply_mask(logits, valid_ids)
            chosen_id = select_best_token(masked)

            if chosen_id is None:
                break

            token_str = self.vocab.id_to_token.get(chosen_id, "")

            if any(token_str.endswith(end) for end in end_tokens):
                break

            accumulated += token_str
            current_ids = current_ids + [chosen_id]

        return accumulated.strip()

    def extract_boolean(
        self,
        prompt_ids: list[int],
    ) -> bool:
        logits = get_next_token_logits(self.model, prompt_ids)
        log_probs = self._log_softmax(logits)

        true_ids = self.model.encode("true").flatten().tolist()
        false_ids = self.model.encode("false").flatten().tolist()

        true_score = self._score_sequence(prompt_ids, true_ids, log_probs)
        false_score = self._score_sequence(prompt_ids, false_ids, log_probs)

        return true_score > false_score
