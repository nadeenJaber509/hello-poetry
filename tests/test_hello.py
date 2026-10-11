from hello_poetry.hello import hello, personalized_hello


def test_hello():
    assert hello() == "Hello, World!"


def test_personalized_english():
    assert personalized_hello("Nadeen", "English") == "Hello, Nadeen!"


def test_personalized_arabic():
    assert personalized_hello("Nadeen", "Arabic") == "مرحباً, Nadeen!"


def test_empty_name():
    assert personalized_hello("", "English") == "Hello, World!"


def test_arabic_empty_name():
    assert personalized_hello("", "Arabic") == "مرحباً, بالعالم!"


def test_name_whitespace():
    assert personalized_hello("  Nadeen  ") == "Hello, Nadeen!"