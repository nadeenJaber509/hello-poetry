from hello_poetry.hello import hello


def test_hello():
    assert hello() == "Hello, World!"