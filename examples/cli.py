import argparse
import json
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import whisper
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

from whisper_punctuator import Punctuator


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
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size for beam search")
    parser.add_argument(
        "--initial-prompt", type=str, default="", help="Optional text to provide as a prompt"
    )
    parser.add_argument(
        "--no-punctuation-before",
        type=str,
        default="",
        help=(
            "Do not insert punctuations `before` these characters."
            "For example, if you set this to `ー` (Japanese long vowel), then `ー` will not be "
            "preceded by a punctuation."
        ),
    )
    parser.add_argument("--unicode-normalize", action="store_true", help="Normalize unicode")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )
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
    parser.add_argument(
        "--truecase-search",
        action="store_true",
        help=(
            "Search for the best truecasing. The current implementation is very slow, but we plan "
            "to improve the implementation in the future."
        ),
    )
    parser.add_argument(
        "--truecase-first-character",
        action="store_true",
        help="Always truecase the first character.",
    )
    parser.add_argument(
        "--no-truecase-first-character", action="store_false", dest="truecase-first-character"
    )
    parser.set_defaults(truecase_first_character=True)
    parser.add_argument(
        "--truecase-after-period",
        action="store_true",
        help=(
            "Always truecase the first character after a period. `.`, `?`, `!`, and `。` are "
            "considered periods by default. To use other characters as periods, use the --periods "
            "option."
        ),
    )
    parser.add_argument(
        "--no-truecase-after-period", action="store_false", dest="truecase-after-period"
    )
    parser.set_defaults(truecase_after_period=True)
    parser.add_argument(
        "--periods",
        type=str,
        default=".?!。",
        help="Period characters for truecasing by the --truecase-after-period option",
    )
    parser.add_argument(
        "--allow-punctuation-within-word",
        action="store_true",
        help=(
            "Allow punctuation insertion within a word. If False, punctuation will only be "
            "inserted at the end of a word."
        ),
    )
    parser.add_argument(
        "--no-allow-punctuation-within-word",
        action="store_false",
        dest="allow-punctuation-within-word",
    )
    parser.set_defaults(allow_punctuation_within_word=False)
    return parser


@dataclass
class Record:
    audio_path: str
    text: str


class AudioDataset(Dataset):
    def __init__(self, records: List[Record], fp16: bool = True) -> None:
        self.records = records
        self.fp16 = fp16

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        mel = log_mel_spectrogram(record.audio_path)
        mel = pad_or_trim(mel, N_FRAMES)
        if self.fp16:
            mel = mel.half()
        return mel


def get_dataloader(records: List[Record], batch_size: int = 1, fp16: bool = True) -> DataLoader:
    dataset = AudioDataset(records, fp16)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


def create_records(
    audio: Optional[str], text: Optional[str], json: Optional[str], unicode_normalize: bool = False
) -> List[Record]:
    if audio is not None and text is not None:
        records = [Record(audio, text)]
    elif json is not None:
        records = read_json(json)
    else:
        raise ValueError("Either --audio and --text or --json must be specified")

    if unicode_normalize:
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


def main():
    args = get_parser().parse_args()

    records = create_records(args.audio, args.text, args.json, args.unicode_normalize)

    punctuator = Punctuator(
        model_name=args.model,
        language=args.language,
        device=args.device,
        punctuations=args.punctuations,
        no_punctuation_before=args.no_punctuation_before,
        initial_prompt=args.initial_prompt,
        beam_size=args.beam_size,
        truecase_search=args.truecase_search,
        truecase_first_character=args.truecase_first_character,
        truecase_after_period=args.truecase_after_period,
        periods=args.periods,
        allow_punctuation_within_word=args.allow_punctuation_within_word,
    )

    # We currently only support batch size 1
    data_loader = get_dataloader(records, batch_size=1, fp16=args.device == "cuda")

    punctuated_records = []
    for record, mel in tqdm(zip(records, data_loader), total=len(records)):
        mel = mel.to(args.device)
        punctuated_text = punctuator.punctuate(audio=mel, text=record.text)
        punctuated_records.append(Record(record.audio_path, punctuated_text))

        if args.verbose:
            tqdm.write(record.audio_path)
            tqdm.write(f"  Original:   {record.text}")
            tqdm.write(f"  Punctuated: {punctuated_text}")

    write_json(punctuated_records, args.output)


if __name__ == "__main__":
    main()
