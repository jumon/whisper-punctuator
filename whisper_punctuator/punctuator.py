import argparse
import json
import re
import string
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from whisper import Whisper
from whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from whisper.tokenizer import Tokenizer


@dataclass
class DecodeOptions:
    initial_tokens: torch.Tensor
    punctuation_tokens: torch.Tensor
    punctuation_suppressing_tokens: torch.Tensor
    min_punctuation_probability: float = 0.0
    min_token_probability: float = 0.0
    beam_size: int = 1
    truecasing: bool = False
    truecase_first_character: bool = True
    truecase_after_period: bool = True
    periods: str = ".?!。"


@dataclass
class Record:
    audio_path: str
    text: str


def create_records(args: argparse.Namespace) -> List[Record]:
    if args.audio is not None and args.text is not None:
        records = [Record(args.audio, args.text)]
    elif args.json is not None:
        records = read_json(args.json)
    else:
        raise ValueError("Either --audio and --text or --json must be specified")

    if args.unicode_normalize:
        records = [
            Record(record.audio_path, unicodedata.normalize("NFKC", record.text))
            for record in records
        ]
    return records


def read_json(path: str) -> List[Record]:
    records = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            records.append(Record(audio_path=data["audio_path"], text=data["text"]))
    return records


def write_json(records: List[Record], output: str):
    with open(output, "w") as f:
        for record in records:
            data = {"audio_path": record.audio_path, "text": record.text}
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


class AudioDataset(Dataset):
    def __init__(self, records: List[Record], tokenizer: Tokenizer, fp16: bool = True) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.fp16 = fp16

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        mel = log_mel_spectrogram(record.audio_path)
        mel = pad_or_trim(mel, N_FRAMES)
        if self.fp16:
            mel = mel.half()

        if self.tokenizer.language in ["ja", "zh"]:
            text = record.text.strip()
        else:
            text = " " + record.text.strip()
        tokens = self.tokenizer.encode(text)
        tokens = torch.tensor(tokens + [self.tokenizer.eot], dtype=torch.long)

        return mel, tokens


def get_dataloader(
    records: List[Record], tokenizer: Tokenizer, batch_size: int = 1, fp16: bool = True
) -> DataLoader:
    dataset = AudioDataset(records, tokenizer, fp16)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


def get_tokens(tokenizer: Tokenizer, characters: Optional[str]) -> torch.Tensor:
    if characters is None:
        return torch.tensor([])

    tokens = []
    for char in characters:
        encoded = tokenizer.encode(char)
        if len(encoded) > 1:
            raise ValueError(f"Character {char} is not a single token")
        tokens.append(encoded[0])
    return torch.tensor(tokens)


def get_initial_tokens(tokenizer: Tokenizer, model: Whisper, initial_prompt: str) -> torch.Tensor:
    initial_tokens = []
    if initial_prompt is not None:
        initial_tokens.append(tokenizer.sot_prev)
        initial_prompt = tokenizer.encode(" " + initial_prompt.strip())
        initial_tokens.extend(initial_prompt[-(model.dims.n_text_ctx // 2 - 1) :])

    initial_tokens.extend(tokenizer.sot_sequence_including_notimestamps)
    return torch.tensor(initial_tokens, device=model.device)


def cleanup_hooks(hooks: List[torch.utils.hooks.RemovableHandle]) -> None:
    for hook in hooks:
        hook.remove()


@dataclass
class BeamNode:
    tokens: torch.Tensor
    sum_log_probs: float
    length: int


def beam_score(node: BeamNode) -> float:
    return node.sum_log_probs / node.length


def includes_lowercase_alphabet(text: str) -> bool:
    return any(c in string.ascii_lowercase for c in text)


def get_same_spelling_tokens(
    token_id: int,
    probabilities: torch.Tensor,
    tokenizer: Tokenizer,
    min_probability: float,
    max_num_tokens: int,
    device: str,
) -> torch.Tensor:
    target_spelling = tokenizer.decode([token_id]).lower()
    if not includes_lowercase_alphabet(target_spelling):
        return torch.tensor([token_id], device=device)

    token_probability = probabilities[token_id]

    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    same_spelling_tokens = []
    for probability, index in zip(sorted_probabilities, sorted_indices):
        if probability < min_probability and probability < token_probability:
            break

        if tokenizer.decode(index).lower() == target_spelling:
            same_spelling_tokens.append(index)
            if len(same_spelling_tokens) >= max_num_tokens:
                break

    return torch.tensor(same_spelling_tokens, device=device)


def post_truecasing(
    text: str,
    truecase_first_character: bool = True,
    truecase_after_period: bool = True,
    periods: str = ".?!。",
) -> str:
    if truecase_first_character:
        text = text[0].upper() + text[1:]

    if truecase_after_period:
        text = re.sub(f"([{periods}] )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)

    return text


@torch.no_grad()
def predict_punctuations(
    mel: torch.Tensor,
    original_text_tokens: torch.Tensor,
    model: Whisper,
    tokenizer: Tokenizer,
    decode_options: DecodeOptions,
) -> str:
    audio_features = model.embed_audio(mel)
    beams = [BeamNode(tokens=decode_options.initial_tokens.clone(), sum_log_probs=0.0, length=0)]

    for i in range(len(original_text_tokens)):
        next_token = original_text_tokens[i]
        skip_punctuation_insertion = (
            i == 0 or next_token in decode_options.punctuation_suppressing_tokens
        )
        new_beams = []

        for beam in beams:
            logits = model.decoder(beam.tokens.unsqueeze(0), audio_features)
            next_probabilities = logits[0, -1, :].softmax(dim=0)

            if decode_options.truecasing:
                next_tokens = get_same_spelling_tokens(
                    token_id=next_token.item(),
                    probabilities=next_probabilities,
                    tokenizer=tokenizer,
                    min_probability=decode_options.min_token_probability,
                    max_num_tokens=decode_options.beam_size,
                    device=model.device,
                )
            else:
                next_tokens = next_token.unsqueeze(0)

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

            candidate_punctuation_tokens = decode_options.punctuation_tokens[
                next_probabilities[decode_options.punctuation_tokens]
                > decode_options.min_punctuation_probability
            ]
            for punctuation_token in candidate_punctuation_tokens:
                punctuation_log_prob = torch.log(next_probabilities[punctuation_token]).item()

                input_tokens = torch.cat([beam.tokens, punctuation_token.unsqueeze(0)], dim=0)
                logits = model.decoder(input_tokens.unsqueeze(0), audio_features)
                token_probabilities = logits[0, -1, :].softmax(dim=0)

                if decode_options.truecasing:
                    next_tokens = get_same_spelling_tokens(
                        token_id=next_token.item(),
                        probabilities=token_probabilities,
                        tokenizer=tokenizer,
                        min_probability=decode_options.min_token_probability,
                        max_num_tokens=decode_options.beam_size,
                        device=model.device,
                    )
                else:
                    next_tokens = next_token.unsqueeze(0)

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

        beams = sorted(new_beams, key=lambda beam: beam_score(beam), reverse=True)
        beams = beams[: decode_options.beam_size]

    best_beam = beams[0]
    punctuated_text = tokenizer.decode(
        best_beam.tokens[decode_options.initial_tokens.shape[0] : -1].tolist()
    ).strip()

    if decode_options.truecasing:
        punctuated_text = post_truecasing(
            text=punctuated_text,
            truecase_first_character=decode_options.truecase_first_character,
            truecase_after_period=decode_options.truecase_after_period,
            periods=decode_options.periods,
        )

    return punctuated_text


def construct_decode_options(
    tokenizer: Tokenizer, model: Whisper, args: argparse.Namespace
) -> DecodeOptions:
    try:
        punctuation_tokens = get_tokens(tokenizer, args.punctuations)
    except ValueError:
        raise ValueError("Punctuations must be single tokens")

    try:
        punctuation_suppressing_tokens = get_tokens(tokenizer, args.punctuation_suppressing_chars)
    except ValueError:
        raise ValueError("punctuation-suppressing-chars must be single tokens")

    punctuation_tokens = punctuation_tokens.to(args.device)
    punctuation_suppressing_tokens = punctuation_suppressing_tokens.to(args.device)

    initial_tokens = get_initial_tokens(tokenizer, model, args.initial_prompt)

    return DecodeOptions(
        initial_tokens=initial_tokens,
        punctuation_tokens=punctuation_tokens,
        punctuation_suppressing_tokens=punctuation_suppressing_tokens,
        min_punctuation_probability=args.min_punctuation_probability,
        min_token_probability=args.min_token_probability,
        beam_size=args.beam_size,
        truecasing=args.truecasing,
        truecase_first_character=args.truecase_first_character,
        truecase_after_period=args.truecase_after_period,
        periods=args.periods,
    )
