import pandas as pd

from prepare_data import (
    remove_urls,
    remove_html,
    remove_spoiler_alert,
    preprocess_text,
)


def test_remove_urls():
    text_with_urls = "Check out this link: https://www.example.com"
    expected_result = "Check out this link: "
    assert remove_urls(text_with_urls) == expected_result


def test_remove_html():
    text_with_html = "<p>This is <b>bold</b> text</p>"
    expected_result = "This is bold text"
    assert remove_html(text_with_html) == expected_result


def test_remove_spoiler_alert():
    text_with_spoiler = "Text before spoiler. (view spoiler)[ Long text containing book spoilers... (hide spoiler)] Text after spoiler."
    expected_result = "Text before spoiler.  Text after spoiler."
    assert remove_spoiler_alert(text_with_spoiler) == expected_result


def test_preprocess_text():
    df = pd.DataFrame(
        {
            "review_text": [
                "Check out this link: https://www.example.com",
                "<p>This is <b>bold</b> text</p>",
                "Text before spoiler. (view spoiler)[ Long text containing book spoilers... (hide spoiler)] Text after spoiler.",
            ]
        }
    )
    expected_result = pd.DataFrame(
        {
            "review_text": [
                "Check out this link: https://www.example.com",
                "<p>This is <b>bold</b> text</p>",
                "Text before spoiler. (view spoiler)[ Long text containing book spoilers... (hide spoiler)] Text after spoiler.",
            ],
            "text": [
                "Check out this link: ",
                "This is bold text",
                "Text before spoiler.  Text after spoiler.",
            ],
        }
    )
    pd.testing.assert_frame_equal(preprocess_text(df), expected_result)
