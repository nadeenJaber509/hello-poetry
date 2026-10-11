from pathlib import Path
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_app_loads():
    at = AppTest.from_file(APP_PATH).run()

    assert not at.exception
    assert len(at.text_input) == 1
    assert len(at.selectbox) == 1
    assert len(at.button) == 1


def test_generate_english_greeting():
    at = AppTest.from_file(APP_PATH).run()

    at.text_input[0].set_value("Alice").run()
    at.button[0].click().run()

    assert not at.exception
    assert len(at.success) == 1
    assert "Hello, Alice!" in at.success[0].value


def test_generate_arabic_greeting():
    at = AppTest.from_file(APP_PATH).run()

    at.text_input[0].set_value("أحمد").run()
    at.selectbox[0].set_value("Arabic").run()
    at.button[0].click().run()

    assert not at.exception
    assert len(at.success) == 1
    assert "مرحباً, أحمد!" in at.success[0].value