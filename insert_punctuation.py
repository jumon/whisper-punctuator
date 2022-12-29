import argparse

import torch
import whisper
from tqdm import tqdm
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer

from whisper_punctuator.punctuator import (
    Record,
    construct_decode_options,
    create_records,
    get_dataloader,
    predict_punctuations,
    write_json,
)


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
        "--min-punctuation-probability",
        type=float,
        default=0.0,
        help=(
            "Minimum probability for a punctuation to be a candidate."
            "Increasing this value will speed up the process, but may result in less punctuation "
            "being inserted."
        ),
    )
    parser.add_argument(
        "--min-token-probability",
        type=float,
        default=0.0,
        help=(
            "Minimum probability for a token with the same spelling, when lowercased, to be a "
            "candidate. Increasing this value will speed up the process, but may cause incorrect "
            "capitalization."
        ),
    )
    parser.add_argument(
        "--initial-prompt", type=str, default=None, help="Optional text to provide as a prompt"
    )
    parser.add_argument(
        "--punctuation-suppressing-chars",
        type=str,
        default=None,
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
    return parser


def main():
    args = get_parser().parse_args()
    records = create_records(args)

    tokenizer = get_tokenizer(
        multilingual=".en" not in args.model, language=args.language, task="transcribe"
    )
    model = whisper.load_model(args.model, args.device)
    decode_options = construct_decode_options(tokenizer, model, args)

    # We currently only support batch size 1
    data_loader = get_dataloader(records, tokenizer, batch_size=1, fp16=args.device == "cuda")

    punctuated_records = []
    for record, (mel, tokens) in tqdm(zip(records, data_loader), total=len(records)):
        mel, tokens = mel.to(args.device), tokens[0].to(args.device)
        punctuated_text = predict_punctuations(
            mel=mel,
            original_text_tokens=tokens,
            model=model,
            tokenizer=tokenizer,
            decode_options=decode_options,
        )
        punctuated_records.append(Record(record.audio_path, punctuated_text))

        if args.verbose:
            tqdm.write(record.audio_path)
            tqdm.write(f"  Original:   {record.text}")
            tqdm.write(f"  Punctuated: {punctuated_text}")

    write_json(punctuated_records, args.output)


if __name__ == "__main__":
    main()
