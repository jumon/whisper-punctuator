import argparse
import json
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple

import torch
import whisper
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from whisper import Whisper
from whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, Tokenizer, get_tokenizer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-shot punctuation insertion using Whisper")
    parser.add_argument("--audio", type=str, help="Path to an audio file")
    parser.add_argument("--text", type=str, help="Text to be punctuated")
    parser.add_argument(
        "--json", type=str, help="Path to a jsonl file containing audio paths and texts"
    )
    parser.add_argument(
        "--punctuations", type=str, default=",.?", help="List of punctuations to insert"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help="Language of the text",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.2,
        help="Minimum probability to insert a punctuation",
    )
    parser.add_argument(
        "--initial-prompt", type=str, default=None, help="Optional text to provide as a prompt"
    )
    parser.add_argument(
        "--punctuation-suppressing-chars",
        type=str,
        default="ー",
        help=(
            "Do not insert punctuations `before` these characters."
            "Default: ー (Japanese long vowel). This is useful to prevent typical Japanese"
            "punctuation insertion errors such as まーはい -> ま、ーはい。"
        ),
    )
    parser.add_argument("--unicode-normalize", action="store_true", help="Normalize unicode")
    parser.add_argument("--device", default="cuda", help="Device to use for inference")
    parser.add_argument(
        "--model",
        default="large",
        choices=whisper.available_models(),
        help="Name of the Whisper model to use",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--output", type=str, default="output/prediction.json", help="Path to the output file"
    )
    return parser


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

        tokens = self.tokenizer.encode(record.text)
        tokens = torch.tensor(tokens, dtype=torch.long)
        return mel, tokens


def get_dataloader(
    json: str, tokenizer: Tokenizer, batch_size: int = 1, fp16: bool = True
) -> DataLoader:
    records = read_json(json)
    dataset = AudioDataset(records, tokenizer, fp16)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


def get_tokens(tokenizer: Tokenizer, characters: str) -> torch.Tensor:
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
    return torch.tensor(initial_tokens).unsqueeze(0).to(model.device)


def cleanup_hooks(hooks: List[torch.utils.hooks.RemovableHandle]) -> None:
    for hook in hooks:
        hook.remove()


@torch.no_grad()
def predict_punctuations(
    mel: torch.Tensor,
    original_text_tokens: torch.Tensor,
    model: Whisper,
    tokenizer: Tokenizer,
    initial_tokens: torch.Tensor,
    punctuation_tokens: torch.Tensor,
    punctuation_suppressing_tokens: torch.Tensor,
    min_probability: float,
) -> str:
    audio_features = model.embed_audio(mel)
    kv_cache, hooks = model.install_kv_cache_hooks()
    input_tokens = initial_tokens
    punctuated_tokens = []

    for i in range(len(original_text_tokens)):
        token = original_text_tokens[i]
        punctuated_tokens.append(token.item())
        if (
            i < len(original_text_tokens) - 1
            and original_text_tokens[i + 1] in punctuation_suppressing_tokens
        ):
            continue

        input_tokens = torch.cat([input_tokens, token.view(1, 1)], dim=1)
        logits = model.decoder(input_tokens, audio_features, kv_cache=kv_cache)
        probabilities = logits.squeeze(0)[-1].softmax(dim=0)
        max_punctuation_probability, max_punctuation_index = torch.max(
            probabilities[punctuation_tokens], dim=0
        )

        if max_punctuation_probability > min_probability:
            selected_punctuation = punctuation_tokens[max_punctuation_index]
            punctuated_tokens.append(selected_punctuation.item())
            # update kv_cache
            model.decoder(selected_punctuation.view(1, 1), audio_features, kv_cache=kv_cache)

        input_tokens = torch.tensor([[]], dtype=torch.long).to(model.device)

    cleanup_hooks(hooks)
    punctuated_text = tokenizer.decode(punctuated_tokens)
    return punctuated_text


def main():
    args = get_parser().parse_args()
    records = create_records(args)

    tokenizer = get_tokenizer(
        multilingual=".en" not in args.model, language=args.language, task="transcribe"
    )
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

    model = whisper.load_model(args.model, args.device)
    initial_tokens = get_initial_tokens(tokenizer, model, args.initial_prompt)
    # We currently only support batch size 1
    data_loader = get_dataloader(args.json, tokenizer, batch_size=1, fp16=args.model == "cuda")

    punctuated_records = []
    for record, (mel, tokens) in tqdm(zip(records, data_loader), total=len(records)):
        mel, tokens = mel.to(args.device), tokens[0].to(args.device)
        punctuated_text = predict_punctuations(
            mel=mel,
            original_text_tokens=tokens,
            model=model,
            tokenizer=tokenizer,
            initial_tokens=initial_tokens,
            punctuation_tokens=punctuation_tokens,
            punctuation_suppressing_tokens=punctuation_suppressing_tokens,
            min_probability=args.min_probability,
        )
        punctuated_records.append(Record(record.audio_path, punctuated_text))

        if args.verbose:
            tqdm.write(record.audio_path)
            tqdm.write(f"  Original:   {record.text}")
            tqdm.write(f"  Punctuated: {punctuated_text}")

    write_json(punctuated_records, args.output)


if __name__ == "__main__":
    main()
