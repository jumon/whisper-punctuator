# whisper-punctuator
Zero-shot punctuation insertion using the [Whisper](https://github.com/openai/whisper) speech recognition model:
* No additional training required
* Works on ANY language that Whispers supports
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
Then, the decoder of the model processes each token in the text one by one, in an autoregressive fashion, along with the encoder hidden states.
The model calculates the probability of each token being generated with and without being preceded by each punctuation character.
It then uses beam search to select the final prediction based on the average log probabilities of the generated tokens.

## Limitations
- The results are dependent on the dataset used to train Whisper, which is not publicly available. Punctuation marks that are rare in the training data may not be recognized well.
- Since the Whisper decoder generates tokens in a left-to-right fashion, it cannot see all future tokens when predicting the punctuation before the current token, which may lead to incorrect punctuation insertion. Using beam search with a large beam-size should mitigate this issue.
- If the model fails to insert punctuation correctly, it may keep doing similar mistakes due to the autoregressive decoding. Using beam search or a prompt can mitigate the problem.
- The `--beam-size` and `--initial-prompt` flags may need to be fine-tuned to get the best results, depending on the data.
- The current implementation does not allow for punctuation marks that consist of multiple tokens according to the Whisper tokenizer.
- If you want to fine-tune a Whisper model using publiclly available data, you may want to ensure that the data is not only punctuated but also truecased. This script does not perform truecasing, but it may be possible to achievet this using similar (but probably more complicated) techinques.
- This script cannot insert punctuation "inside" a token. This would be a problem for languages that do not use spaces between words such as Japanese and Chinese. For example, the Japanese text `えールール` (which means "uh, a rule", where "えー" means "uh" and "ルール" means "rule") is tokenized into `え` `ール` `ール` by the Whisper tokenizer, thus the script cannot insert punctuation between `ー` and `ル` to get the correct punctuated text `えー、ルール`. This is a limitation of tokenizers in general, and it is not clear how to solve this problem.
