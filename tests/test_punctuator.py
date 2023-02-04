import os

import pytest

from whisper_punctuator import Punctuator

# Testcases are taken from the TEDLIUM2 corpus (https://www.openslr.org/19/, CC BY-NC-ND 3.0)


@pytest.fixture
def punctuator():
    return Punctuator(model_name="tiny", device="cpu")


@pytest.fixture
def audio(punctuator: Punctuator):
    audio_path = os.path.join(os.path.dirname(__file__), "test.wav")
    return punctuator._load_mel(audio_path)


@pytest.fixture
def text():
    return (
        "and do you know what the answer to this question now is the answer is no it "
        "is not possible to buy a cell phone that doesn't do too much so"
    )


# The "tiny" model is used for testing purposes and results are not very accurate
@pytest.mark.parametrize(
    (
        "punctuations",
        "initial_prompt",
        "truecase_search",
        "punctuated_text",
    ),
    [
        (
            ",.?",
            "",
            False,
            (
                "and do you know what the answer to this question now is? the answer is no. it is "
                "not possible to buy a cell phone that doesn't do too much. so"
            ),
        ),
        (
            ",?",
            "",
            False,
            (
                "and do you know what the answer to this question now is? the answer is no, it is "
                "not possible to buy a cell phone that doesn't do too much, so"
            ),
        ),
        (
            ",.?",
            "hello everyone i'm gonna talk without using any punctuation this is a test",
            False,
            (
                "and do you know what the answer to this question now is the answer is no it is "
                "not possible to buy a cell phone that doesn't do too much so"
            ),
        ),
        (
            ",.?",
            "Hello, everyone.",
            True,
            (
                "And do you know what the answer to this question now is? The answer is no. It is "
                "not possible to buy a cell phone that doesn't do too much. So."
            ),
        ),
    ],
)
def test_punctuate(
    audio,
    text,
    punctuations: str,
    initial_prompt: str,
    truecase_search: bool,
    punctuated_text: str,
):
    punctuator = Punctuator(
        model_name="tiny",
        language="en",
        device="cpu",
        punctuations=punctuations,
        initial_prompt=initial_prompt,
        truecase_search=truecase_search,
        truecase_after_period=False,
        truecase_first_character=False,
    )
    assert punctuator.punctuate(audio=audio, text=text) == punctuated_text


@pytest.mark.parametrize(
    ("text", "punctuated_text", "truecase_first_character", "truecase_after_period", "periods"),
    [
        (
            "but, i felt worse. why? i wrote a whole book to try and explain this to myself. the",
            "But, i felt worse. Why? I wrote a whole book to try and explain this to myself. The",
            True,
            True,
            ".?!。",
        ),
        (
            "but, i felt worse. why? i wrote a whole book to try and explain this to myself. the",
            "but, i felt worse. Why? I wrote a whole book to try and explain this to myself. The",
            False,
            True,
            ".?!。",
        ),
        (
            "but, i felt worse. why? i wrote a whole book to try and explain this to myself. the",
            "But, i felt worse. why? i wrote a whole book to try and explain this to myself. the",
            True,
            False,
            ".?!。",
        ),
        (
            "but, i felt worse. why? i wrote a whole book to try and explain this to myself. the",
            "but, i felt worse. why? i wrote a whole book to try and explain this to myself. the",
            False,
            False,
            ".?!。",
        ),
        (
            "but, i felt worse. why? i wrote a whole book to try and explain this to myself. the",
            "But, i felt worse. Why? i wrote a whole book to try and explain this to myself. The",
            True,
            True,
            ".!。",
        ),
    ],
)
def test_post_truecasing(
    punctuator: Punctuator,
    text: str,
    punctuated_text: str,
    truecase_first_character: bool,
    truecase_after_period: bool,
    periods: str,
):
    assert (
        punctuator.post_truecasing(
            text=text,
            truecase_first_character=truecase_first_character,
            truecase_after_period=truecase_after_period,
            periods=periods,
        )
        == punctuated_text
    )
