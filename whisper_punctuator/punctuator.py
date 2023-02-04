import re
import string
from dataclasses import dataclass
from functools import lru_cache
from logging import getLogger
from typing import List, Optional, Union

import torch
import whisper
from whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from whisper.tokenizer import get_tokenizer

logger = getLogger(__name__)


@dataclass
class DecodeOptions:
    initial_tokens: torch.Tensor
    punctuation_tokens: torch.Tensor
    no_punctuation_before: List[str]  # do not insert punctuation before these patterns
    beam_size: int = 1
    truecase_search: bool = False
    truecase_first_character: bool = True
    truecase_after_period: bool = True
    periods: str = ".?!。"
    # whether to allow punctuation insertion other than before spaces
    allow_punctuation_within_word: bool = False


class BeamNode:
    def __init__(
        self, tokens: torch.Tensor, sum_log_probs: float, length: int, pos: int = 0
    ) -> None:
        self.tokens = tokens
        self.sum_log_probs = sum_log_probs
        self.length = length
        self.pos = pos  # how many bytes of the original text have been processed

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


class Punctuator:
    def __init__(
        self,
        model_name: str = "small",
        language: str = "en",
        device: Optional[str] = None,
        punctuations: str = ",.?",
        no_punctuation_before: str = "",
        initial_prompt: str = "",
        beam_size: int = 1,
        truecase_search: bool = False,
        truecase_first_character: bool = True,
        truecase_after_period: bool = True,
        periods: str = ".?!。",
        allow_punctuation_within_word: Optional[bool] = None,
    ):
        self.NOSPACE_LANGUAGES = ["zh", "ja", "th", "lo", "my"]

        self.device = self._get_device(device)
        self.fp16 = self.device == "cuda"
        self.language = language
        self.model = whisper.load_model(model_name, device=self.device)
        self.whisper_tokenizer = get_tokenizer(
            multilingual=".en" not in model_name, language=language, task="transcribe"
        )
        # employ Whisper Tokenizer's internal GPT2TokenizerFast to use `convert_ids_to_tokens``
        self.tokenizer = self.whisper_tokenizer.tokenizer

        try:
            punctuation_tokens = self._get_tokens(punctuations)
        except ValueError:
            raise ValueError("Each character in `punctuations` must be a single token")

        self.allow_punctuation_within_word = self._get_allow_punctuation_within_word(
            allow_punctuation_within_word
        )

        self.options = DecodeOptions(
            initial_tokens=self._get_initial_tokens(initial_prompt),
            punctuation_tokens=punctuation_tokens,
            no_punctuation_before=self._get_no_punctuation_before(no_punctuation_before),
            beam_size=beam_size,
            truecase_search=truecase_search,
            truecase_first_character=truecase_first_character,
            truecase_after_period=truecase_after_period,
            periods=periods,
            allow_punctuation_within_word=self.allow_punctuation_within_word,
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
            initial_tokens.append(self.whisper_tokenizer.sot_prev)
            initial_prompt = self._add_leading_space(initial_prompt)
            initial_prompt = self.tokenizer.encode(initial_prompt)
            initial_tokens.extend(initial_prompt[-(self.model.dims.n_text_ctx // 2 - 1) :])

        initial_tokens.extend(self.whisper_tokenizer.sot_sequence_including_notimestamps)
        return torch.tensor(initial_tokens, device=self.device)

    def _add_leading_space(self, text: str) -> str:
        """
        Add a leading space to the text if the language uses spaces to separate words.
        For languages that do not use spaces, namely Chinese, Japanese, Thai, Lao, and
        Burmese, return the text as is.
        """
        if self.language in self.NOSPACE_LANGUAGES:
            return text.strip()
        else:
            return " " + text.strip()

    def _get_no_punctuation_before(self, punctuation_suppressing_chars: str) -> List[str]:
        return [self._byte_encode(char) for char in punctuation_suppressing_chars]

    def _get_allow_punctuation_within_word(
        self, allow_punctuation_within_word: Optional[bool]
    ) -> bool:
        if not allow_punctuation_within_word and self.language in self.NOSPACE_LANGUAGES:
            logger.warning(
                f"`allow_punctuation_within_word` is set False, but the language {self.language}"
                "does not use spaces to separate words, which results in no punctuation insertion."
            )

        if allow_punctuation_within_word is None:
            allow_punctuation_within_word = self.language in self.NOSPACE_LANGUAGES

        return allow_punctuation_within_word

    def _load_mel(self, audio_path: str) -> torch.Tensor:
        mel = log_mel_spectrogram(audio_path)
        mel = pad_or_trim(mel, N_FRAMES)
        if self.fp16:
            mel = mel.half()
        return mel.unsqueeze(0).to(self.device)

    def _should_skip_punctuation_insertion(self, beam: BeamNode, byte_encoded_text: str) -> bool:
        # do not insert punctuation at the beginning of the text
        if beam.pos == 0:
            return True

        # do not insert punctuation in a row
        if len(beam.tokens) > 0 and beam.tokens[-1] in self.options.punctuation_tokens:
            return True

        # do not insert punctuation within a word
        if not self.options.allow_punctuation_within_word:
            # Ġ corresponds to a space in the GP2TokenizerFast's representation
            if beam.pos < len(byte_encoded_text) and byte_encoded_text[beam.pos] != "Ġ":
                return True

        # do not insert punctuation before a character that suppresses it
        for pattern in self.options.no_punctuation_before:
            if byte_encoded_text[beam.pos :].startswith(pattern):
                return True

        return False

    @lru_cache(maxsize=1000)
    def _get_next_tokens(self, byte_encoded_text: str, pos: int) -> torch.Tensor:
        if pos == len(byte_encoded_text):
            return torch.tensor([self.tokenizer.eos_token_id], device=self.device)

        next_tokens = []
        if self.options.truecase_search:
            byte_encoded_text = self._lowercase_ascii(byte_encoded_text)

        # TODO: employ a more efficient search algorithm (probably a trie)
        # eos_token_id is the first special token in the vocabulary
        for index in range(self.tokenizer.eos_token_id):
            index_token_text = self.tokenizer.convert_ids_to_tokens(index)
            if self.options.truecase_search:
                index_token_text = self._lowercase_ascii(index_token_text)

            if byte_encoded_text[pos:].startswith(index_token_text):
                next_tokens.append(index)

        return torch.tensor(next_tokens, device=self.device)

    def _lowercase_ascii(self, text: str) -> str:
        lowercased = []
        for char in text:
            if char in string.ascii_uppercase:
                lowercased.append(char.lower())
            else:
                lowercased.append(char)
        return "".join(lowercased)

    def _byte_encode(self, text: str) -> str:
        token_ids = self.tokenizer.encode(text)
        return "".join(self.tokenizer.convert_ids_to_tokens(token_ids))

    @torch.no_grad()
    def punctuate(self, audio: Union[str, torch.Tensor], text: str) -> str:
        if isinstance(audio, str):
            audio = self._load_mel(audio)

        text = self._add_leading_space(text)
        byte_encoded_text = self._byte_encode(text)
        audio_features = self.model.embed_audio(audio)
        beams = [BeamNode(tokens=self.options.initial_tokens, sum_log_probs=0.0, length=0, pos=0)]
        finished_beams = []

        while True:
            new_beams = []
            for beam in beams:
                logits = self.model.decoder(beam.tokens.unsqueeze(0), audio_features)
                next_probabilities = logits[0, -1, :].softmax(dim=0)
                next_tokens = self._get_next_tokens(byte_encoded_text, beam.pos)

                for next_token in next_tokens:
                    log_prob = torch.log(next_probabilities[next_token]).item()
                    no_punctuation_beam = BeamNode(
                        tokens=torch.cat([beam.tokens, next_token.unsqueeze(0)], dim=0),
                        sum_log_probs=beam.sum_log_probs + log_prob,
                        length=beam.length + 1,
                        pos=beam.pos + len(self.tokenizer.convert_ids_to_tokens(next_token.item())),
                    )
                    new_beams.append(no_punctuation_beam)

                if self._should_skip_punctuation_insertion(beam, byte_encoded_text):
                    continue

                for punctuation_token in self.options.punctuation_tokens:
                    log_prob = torch.log(next_probabilities[punctuation_token]).item()
                    punctuation_beam = BeamNode(
                        tokens=torch.cat([beam.tokens, punctuation_token.unsqueeze(0)], dim=0),
                        sum_log_probs=beam.sum_log_probs + log_prob,
                        length=beam.length + 1,
                        pos=beam.pos,
                    )
                    new_beams.append(punctuation_beam)

            new_beams = sorted(new_beams, reverse=True)

            # construct beams for the next iteration
            beams = []
            for new_beam in new_beams:
                if new_beam.tokens[-1] == self.tokenizer.eos_token_id:
                    finished_beams.append(new_beam)
                    if len(finished_beams) == self.options.beam_size:
                        break
                elif new_beam.tokens.size(0) < self.model.dims.n_text_ctx:
                    beams.append(new_beam)
                    if len(beams) == self.options.beam_size:
                        break

            if len(finished_beams) == self.options.beam_size or len(beams) == 0:
                break

        if len(finished_beams) == 0:
            raise RuntimeError(
                "No beams finished. Probably the input text is too long to be processed."
            )

        best_beam = sorted(finished_beams, reverse=True)[0]
        punctuated_tokens = best_beam.tokens[self.options.initial_tokens.shape[0] : -1]
        punctuated_text = self.tokenizer.decode(punctuated_tokens.tolist()).strip()
        punctuated_text = self.post_truecasing(punctuated_text)

        return punctuated_text

    def post_truecasing(
        self,
        text: str,
        truecase_first_character: Optional[bool] = None,
        truecase_after_period: Optional[bool] = None,
        periods: Optional[str] = None,
    ) -> str:
        if truecase_first_character is None:
            truecase_first_character = self.options.truecase_first_character
        if truecase_after_period is None:
            truecase_after_period = self.options.truecase_after_period
        if periods is None:
            periods = self.options.periods

        if truecase_first_character and len(text) >= 1:
            text = text[0].upper() + text[1:]

        if truecase_after_period:
            text = re.sub(
                f"([{periods}] )([a-z])",
                lambda m: m.group(1) + m.group(2).upper(),
                text,
            )
        return text
