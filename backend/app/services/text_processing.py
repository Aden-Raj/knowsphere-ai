import re
import unicodedata


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    filtered = []
    preserved_formatting = {"\n", "\t", "\u200c", "\u200d"}
    for character in text:
        category = unicodedata.category(character)
        if character in preserved_formatting or category not in {"Cc", "Cf"}:
            filtered.append(character)
    text = "".join(filtered).replace("\u00a0", " ")
    text = re.sub(r"[\t ]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
