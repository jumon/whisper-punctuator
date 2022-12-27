import pytest

from whisper_punctuator.punctuator import post_truecasing


# Testcases are take from the TEDLIUM2 corpus (https://www.openslr.org/19/, CC BY-NC-ND 3.0)
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
    text: str,
    punctuated_text: str,
    truecase_first_character: bool,
    truecase_after_period: bool,
    periods: str,
):
    assert (
        post_truecasing(
            text=text,
            truecase_first_character=truecase_first_character,
            truecase_after_period=truecase_after_period,
            periods=periods,
        )
        == punctuated_text
    )
