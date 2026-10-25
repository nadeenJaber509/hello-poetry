from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_app_loads():
    at = AppTest.from_file(APP_PATH).run()

    assert not at.exception
    assert at.title[0].value == "📊 Data Explorer"
    assert len(at.file_uploader) == 1


def test_sample_dataset():
    at = AppTest.from_file(APP_PATH).run()

    assert not at.exception
    assert at.metric[0].value == "5"
    assert at.metric[1].value == "4"
    assert len(at.selectbox) == 2