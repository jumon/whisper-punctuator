import re
import string
from dataclasses import dataclass
from typing import Optional, Union

import torch
import whisper
from whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from whisper.tokenizer import get_tokenizer


@dataclass
class DecodeOptions:
    initial_tokens: torch.Tensor
    punctuation_tokens: torch.Tensor
    punctuation_suppressing_tokens: torch.Tensor
    min_punctuation_probability: float = 0.0
    min_token_probability: float = 0.0
    beam_size: int = 1
    truecase_search: bool = False
    truecase_first_character: bool = True
    truecase_after_period: bool = True
    periods: str = ".?!。"


class Punctuator:
    def __init__(
        self,
        model_name: str = "small",
        language: str = "en",
        device: Optional[str] = None,
        punctuations: str = ",.?",
        punctuation_suppressing_chars: str = "",
        initial_prompt: str = "",
        min_punctuation_probability: float = 0.0,
        min_token_probability: float = 0.0,
        beam_size: int = 1,
        truecase_search: bool = False,
        truecase_first_character: bool = True,
        truecase_after_period: bool = True,
        periods: str = ".?!。",
    ):
        self.device = self._get_device(device)
        self.fp16 = self.device == "cuda"
        self.tokenizer = get_tokenizer(
            multilingual=".en" not in model_name, language=language, task="transcribe"
        )
        self.model = whisper.load_model(model_name, device=self.device)

        try:
            punctuation_tokens = self._get_tokens(punctuations)
        except ValueError:
            raise ValueError("Each character in `punctuations` must be a single token")

        try:
            punctuation_suppressing_tokens = self._get_tokens(punctuation_suppressing_chars)
        except ValueError:
            raise ValueError(
                "Each character in `punctuation_suppressing_chars` must be a single token"
            )

        self.decode_options = DecodeOptions(
            initial_tokens=self._get_initial_tokens(initial_prompt),
            punctuation_tokens=punctuation_tokens,
            punctuation_suppressing_tokens=punctuation_suppressing_tokens,
            min_punctuation_probability=min_punctuation_probability,
            min_token_probability=min_token_probability,
            beam_size=beam_size,
            truecase_search=truecase_search,
            truecase_first_character=truecase_first_character,
            truecase_after_period=truecase_after_period,
            periods=periods,
        )

    def _get_device(self, device: Optional[str]) -> str:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _get_tokens(self, characters: Optional[str]) -> torch.Tensor:
        tokens = []
        for char in characters:
            encoded = self.tokenizer.encode(char)
            if len(encoded) > 1:
                raise ValueError(f"Character {char} is not a single token")
            tokens.append(encoded[0])
        return torch.tensor(tokens, device=self.device)

    def _get_initial_tokens(self, initial_prompt: str) -> torch.Tensor:
        initial_tokens = []
        if initial_prompt != "":
            initial_tokens.append(self.tokenizer.sot_prev)
            initial_prompt = self.tokenizer.encode(" " + initial_prompt.strip())
            initial_tokens.extend(initial_prompt[-(self.model.dims.n_text_ctx // 2 - 1) :])

        initial_tokens.extend(self.tokenizer.sot_sequence_including_notimestamps)
        return torch.tensor(initial_tokens, device=self.device)

    def _includes_lowercase_alphabet(self, text: str) -> bool:
        return any(c in string.ascii_lowercase for c in text)

    def _get_same_spelling_tokens(
        self, token_id: int, probabilities: torch.Tensor, max_num_tokens: int
    ) -> torch.Tensor:
        target_spelling = self.tokenizer.decode([token_id]).lower()
        if not self._includes_lowercase_alphabet(target_spelling):
            return torch.tensor([token_id], device=self.device)

        token_probability = probabilities[token_id]

        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        same_spelling_tokens = []
        for probability, index in zip(sorted_probabilities, sorted_indices):
            if (
                probability < self.decode_options.min_token_probability
                and probability < token_probability
            ):
                break

            if self.tokenizer.decode(index).lower() == target_spelling:
                same_spelling_tokens.append(index)
                if len(same_spelling_tokens) >= max_num_tokens:
                    break

        return torch.tensor(same_spelling_tokens, device=self.device)

    def post_truecasing(
        self,
        text: str,
        truecase_first_character: Optional[bool] = None,
        truecase_after_period: Optional[bool] = None,
        periods: Optional[str] = None,
    ) -> str:
        if truecase_first_character is None:
            truecase_first_character = self.decode_options.truecase_first_character
        if truecase_after_period is None:
            truecase_after_period = self.decode_options.truecase_after_period
        if periods is None:
            periods = self.decode_options.periods

        if truecase_first_character and len(text) >= 1:
            text = text[0].upper() + text[1:]

        if truecase_after_period:
            text = re.sub(
                f"([{periods}] )([a-z])",
                lambda m: m.group(1) + m.group(2).upper(),
                text,
            )
        return text

    def _load_mel(self, audio_path: str) -> torch.Tensor:
        mel = log_mel_spectrogram(audio_path)
        mel = pad_or_trim(mel, N_FRAMES)
        if self.fp16:
            mel = mel.half()
        return mel.unsqueeze(0).to(self.device)

    def _encode_text(self, text: str) -> torch.Tensor:
        if self.tokenizer.language in ["ja", "zh"]:
            text = text.strip()
        else:
            text = " " + text.strip()
        tokens = self.tokenizer.encode(text)
        tokens = torch.tensor(tokens + [self.tokenizer.eot], dtype=torch.long)
        return tokens

    def _get_candidate_punctuations(self, probabilities: torch.Tensor) -> torch.Tensor:
        return self.decode_options.punctuation_tokens[
            probabilities[self.decode_options.punctuation_tokens]
            > self.decode_options.min_punctuation_probability
        ]

    def _should_skip_punctuation_insertion(self, i: int, next_token: torch.Tensor) -> bool:
        return i == 0 or next_token in self.decode_options.punctuation_suppressing_tokens

    def _get_next_tokens(
        self, next_token: torch.Tensor, probabilities: torch.Tensor
    ) -> torch.Tensor:
        if self.decode_options.truecase_search:
            return self._get_same_spelling_tokens(
                token_id=next_token.item(),
                probabilities=probabilities,
                max_num_tokens=self.decode_options.beam_size,
            )
        else:
            return next_token.unsqueeze(0)

    @torch.no_grad()
    def punctuate(self, audio: Union[str, torch.Tensor], text: Union[str, torch.Tensor]) -> str:
        if isinstance(audio, str):
            audio = self._load_mel(audio)

        if isinstance(text, str):
            original_text_tokens = self._encode_text(text)
        else:
            original_text_tokens = text

        audio_features = self.model.embed_audio(audio)
        beams = [BeamNode(tokens=self.decode_options.initial_tokens, sum_log_probs=0.0, length=0)]

        for i in range(len(original_text_tokens)):
            next_token = original_text_tokens[i]
            skip_punctuation_insertion = self._should_skip_punctuation_insertion(i, next_token)
            new_beams = []

            for beam in beams:
                logits = self.model.decoder(beam.tokens.unsqueeze(0), audio_features)
                next_probabilities = logits[0, -1, :].softmax(dim=0)
                next_tokens = self._get_next_tokens(next_token, next_probabilities)

                for next_token in next_tokens:
                    next_token_log_prob = torch.log(next_probabilities[next_token]).item()
                    no_punctuation_beam = BeamNode(
                        tokens=torch.cat([beam.tokens, next_token.unsqueeze(0)], dim=0),
                        sum_log_probs=beam.sum_log_probs + next_token_log_prob,
                        length=beam.length + 1,
                    )
                    new_beams.append(no_punctuation_beam)

                if skip_punctuation_insertion:
                    continue

                candidate_punctuations = self._get_candidate_punctuations(next_probabilities)
                for punctuation_token in candidate_punctuations:
                    punctuation_log_prob = torch.log(next_probabilities[punctuation_token]).item()

                    input_tokens = torch.cat([beam.tokens, punctuation_token.unsqueeze(0)], dim=0)
                    logits = self.model.decoder(input_tokens.unsqueeze(0), audio_features)
                    token_probabilities = logits[0, -1, :].softmax(dim=0)
                    next_tokens = self._get_next_tokens(next_token, token_probabilities)

                    for next_token in next_tokens:
                        next_token_log_prob = torch.log(token_probabilities[next_token]).item()
                        new_beam = BeamNode(
                            tokens=torch.cat([input_tokens, next_token.unsqueeze(0)], dim=0),
                            sum_log_probs=beam.sum_log_probs
                            + punctuation_log_prob
                            + next_token_log_prob,
                            length=beam.length + 2,
                        )
                        new_beams.append(new_beam)

            beams = sorted(new_beams, reverse=True)[: self.decode_options.beam_size]

        best_beam = beams[0]
        punctuated_tokens = best_beam.tokens[self.decode_options.initial_tokens.shape[0] : -1]
        punctuated_text = self.tokenizer.decode(punctuated_tokens.tolist()).strip()
        punctuated_text = self.post_truecasing(punctuated_text)

        return punctuated_text


class BeamNode:
    def __init__(self, tokens: torch.Tensor, sum_log_probs: float, length: int) -> None:
        self.tokens = tokens
        self.sum_log_probs = sum_log_probs
        self.length = length

    def beam_score(self) -> float:
        return self.sum_log_probs / self.length

    def __lt__(self, other: "BeamNode") -> bool:
        return self.beam_score() < other.beam_score()

    def __gt__(self, other: "BeamNode") -> bool:
        return self.beam_score() > other.beam_score()

    def __eq__(self, other: "BeamNode") -> bool:
        return self.beam_score() == other.beam_score()

    def __le__(self, other: "BeamNode") -> bool:
        return self.beam_score() <= other.beam_score()

    def __ge__(self, other: "BeamNode") -> bool:
        return self.beam_score() >= other.beam_score()
