def hello():
    return "Hello, World!"


def personalized_hello(name: str, language: str = "English") -> str:
    greetings = {
        "English": "Hello",
        "Arabic": "مرحباً",
    }

    default_names = {
        "English": "World",
        "Arabic": "بالعالم",
    }

    greeting = greetings.get(language, "Hello")
    clean_name = name.strip() or default_names.get(language, "World")

    return f"{greeting}, {clean_name}!"