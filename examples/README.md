# Command-line examples

To insert punctuation into text using a pair of text and an audio file as input, run the following command:

```
python cli.py --audio <path-to-audio-file> --text <text-to-be-punctuated>
```

Note that the audio needs to be shorter than 30 seconds; if it is longer, the first 30 seconds will be used.
The script above also conducts rule-based truecasing by default, which capitalizes the first letter of the input text and first letters after periods (`.`, `?`, `!`, and `。` by default).
These can be turned off by using the `--no-truecase-first-character` and `--no-truecase-after-period` flags, respectively.

The default setting treats `,` `.` `?` as punctuation. To change the punctuation characters, use the `--punctuations` flag and specify a list of characters. For example, to treat `,` `.` `?` `!` as punctuation, run:

```
python cli.py --audio <path-to-audio-file> --text <text-to-be-punctuated> --punctuations ",.?!"
```

To handle languages other than English, you can use the `--language` flag to specify the language. For example, to insert punctuation for a Japanese text and treat `。` `、` as punctuation, run:

```
python cli.py --audio <path-to-audio-file> --text <text-to-be-punctuated> --language ja --punctuations "。、"
```

To change the style of punctuation, use the `--initial-prompt` flag to specify a prompt. This will make the model more likely insert punctuation in the style of the prompt. For example, to insert punctuation after every word (though this is not recommended), run:

```
python cli.py --audio <path-to-audio-file> --text <text-to-be-punctuated> --initial-prompt "hello, how, are, you, today?"
```

To conduct search-based truecasing, use the `--truecase-search` flag. This allows you to capitalize letters that are not the first letter of a sentence.

```
python cli.py --audio <path-to-audio-file> --text <text-to-be-punctuated> --truecase-search
```

For other available options, run:

```
python cli.py --help
```
