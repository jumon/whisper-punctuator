# whisper-punctuator

Zero-shot punctuation insertion and truecasing using the [Whisper](https://github.com/openai/whisper) speech recognition model:

- No additional training required
- Works on ANY language supported by Whisper
- Can change the style of punctuation by using a prompt

Have you ever wanted to fine-tune a Whisper model using public data, but the data doesn't have punctuation? If so, this repository is for you! It allows you to insert punctuation into unpunctuated text using the Whisper model in a zero-shot fashion using a pair of unpunctuated text and audio files, and it also supports truecasing.

## Setup

Run the following commands to install:

```
git clone https://github.com/jumon/whisper-punctuator.git
cd whisper-punctuator
pip install -e .
```

If you see an error message like `ERROR: Directory '.' is not installable. Neither 'setup.py' nor 'pyproject.toml' found.`, you need to update pip to the latest version:

```
pip install --upgrade pip
```

You may also need to install [ffmpeg](https://ffmpeg.org/) and [rust](https://www.rust-lang.org/) depending on your environment. See the [instructions](https://github.com/openai/whisper#setup) in the Whisper repository for more details if you encounter any errors.

## Usage

To insert punctuation into text using a pair of text and an audio file as input, you can use the following code (the example audio and text are taken from the TEDLIUM2 corpus (https://www.openslr.org/19/, CC BY-NC-ND 3.0):

```python
from whisper_punctuator import Punctuator

punctuator = Punctuator(language="en", punctuations=",.?", initial_prompt="Hello, everyone.")
punctuated_text = punctuator.punctuate(
    "tests/test.wav",
    "and do you know what the answer to this question now is the answer is no it is not possible to buy a cell phone that doesn't do too much so"
)
print(punctuated_text) # -> "And do you know what the answer to this question now is? The answer is no. It is not possible to buy a cell phone that doesn't do too much. So"
```

Note that the audio needs to be shorter than 30 seconds; if it is longer, the first 30 seconds will be used.

For a command line example and more options, see the [examples](examples) directory.

## How does it work?

Whisper is an automatic speech recognition (ASR) model trained on a massive amount of labeled audio data collected from the internet.
The data used in its training contains punctuation, so the model learns to recognize punctuation as well.
This allows Whisper to insert punctuation into a text given an audio and text pair.
To insert punctuation, Whisper first processes the audio through its encoder, generating encoder hidden states.
Given the generated hidden states, the decoder generates tokens, restricting the output to the input text with the addition of punctuation marks.
The final prediction is determined using beam search, based on the average log probabilities of the generated tokens.

## Limitations

- The results are dependent on the dataset used to train Whisper, which is not publicly available. Punctuation marks that are rare in the training data may not be recognized well.
- Since the Whisper decoder generates tokens in a left-to-right fashion, it cannot see all future tokens when predicting the punctuation before the current token, which may lead to incorrect punctuation insertion. Using beam search with a large beam-size should mitigate this issue.
- If the model fails to insert punctuation correctly, it may keep doing similar mistakes due to the autoregressive decoding. Using beam search or a prompt can mitigate the problem.
- The `--beam-size` and `--initial-prompt` flags may need to be fine-tuned to get the best results, depending on the data.
- The current implementation does not allow for punctuation marks that consist of multiple tokens according to the Whisper tokenizer.
