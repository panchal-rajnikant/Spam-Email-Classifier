import re
import string

def lower(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(
        str.maketrans('', '', string.punctuation)
    )

def remove_numbers(text):
    return re.sub(r"\d", "", text)

def remove_extra_space(text):
    return " ".join(text.split())

def clean_text(text):
    text = lower(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_extra_space(text)
    return text



