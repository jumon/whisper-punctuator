# whisper-punctuator
Zero-shot punctuation insertion using the [Whisper](https://github.com/openai/whisper) speech recognition model:
* No additional training required
* Works on any language that Whispers supports
* Can change the style of punctuation by using a prompt

Have you ever wanted to fine-tune a Whisper model using public data, but the data doesn't have punctuation? If so, this script is for you! It allows you to insert punctuation into unpunctuated text using the Whisper model in a zero-shot fashion, using a pair of unpunctuated text and audio files.

## Quick Start
To use the script, first install [Whisper](https://github.com/openai/whisper) and its dependencies. See the instructions [here](https://github.com/openai/whisper#setup) for more details.

Run the following command to insert punctuation into text using a text and audio file pair as input:
```
python insert_punctuation.py --audio <path-to-audio-file> --text <text-to-be-punctuated>
```
Note that the audio needs to be shorter than 30 seconds; if it is longer, the first 30 seconds will be used.

The default setting treats `,` `.` `?` as punctuation. To change the punctuation characters, use the `--punctuations` flag and specify a list of characters. For example, to treat `,` `.` `?` `!` as punctuation, run:
```
python insert_punctuation.py --audio <path-to-audio-file> --text <text-to-be-punctuated> --punctuations ",.?!"
```
To handle languages other than English, you can use the `--language` flag to specify the language. For example, to insert punctuation for a Japanese text and treat `。` `、` as punctuation, run:
```
python insert_punctuation.py --audio <path-to-audio-file> --text <text-to-be-punctuated> --language ja --punctuations "。、"
```
To change the style of punctuation, use the `--initial-prompt` flag to specify a prompt. This will make the model more likely insert punctuation in the style of the prompt. For example, to insert punctuation after every word (though this is not recommended), run:
```
python insert_punctuation.py --audio <path-to-audio-file> --text <text-to-be-punctuated> --initial-prompt "hello, how, are, you, today?"
```

For all available options, run:
```
python insert_punctuation.py --help
```

## How does it work?
Whisper is an automatic speech recognition (ASR) model trained on a massive amount of labeled audio data collected from the internet.
The data used in its training contains punctuation, so the model learns to recognize punctuation as well.
This allows Whisper to insert punctuation into a text given an audio and text pair.
To insert punctuation, the audio is first input into the encoder of the model to generate the encoder hidden states.
Then, the decoder of the model processes each token in the text one by one, in an autoregressive fashion, along with the encoder hidden states. After each token, the model predicts the output probability for each punctuation character. If the highest probability is above a specified threshold (specified using the `--min-probability` flag), the punctuation is inserted after the token.

## Limitations
- The results are dependent on the dataset used to train Whisper, which is not publicly available. Punctuation marks that are rare in the training data may not be recognized well.
- Since the Whisper decoder generates tokens in a left-to-right fashion, it cannot see future tokens when predicting the punctuation after the current token. This can lead to problems in some cases, such as when `まーはい` (uh yes, where まー means uh and はい means yes) in Japanese is often punctuated as `ま、ーはい` instead of `まー、はい` This is because the model cannot see the future `ー` (Japanese long vowel) when predicting the punctuation after `ま`, and `ま` itself also means `uh` in Japanese. We circumvent this problem by preventing the model from inserting punctuation after `ー` that is specified by the `--punctuation-suppressing-chars` flag. However, this is not a perfect solution and the model may still suffer from problems due to its left-to-right decoding nature.
- If the model fails to insert punctuation when it should, it may enter a "no-punctuation" mode and not insert any further punctuation in the text. This is another issue caused by the left-to-right decoding nature. To mitigate this problem, you can use the `--initial-prompt` flag to induce the model to enter a "punctuation" mode.
- The `--min-probability` and `--initial-prompt` flags may need to be fine-tuned to get the best results, depending on the data.
- The current implementation does not allow for punctuation marks that consist of multiple tokens according to the Whisper tokenizer.
- If you want to fine-tune a Whisper model using publiclly available data, you may want to ensure that the data is not only punctuated but also truecased. This script does not perform truecasing, but it may be possible to achievet this using similar (but probably more complicated) techinques.
