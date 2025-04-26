import re
from lightspeech.utils.number2text import lex_of_number


def normalize_contractions(text):

    contractions = {
        # To be contractions
        r"(\w+)'re": r"\1 are",
        r"(\w+)'m": r"\1 am",
        r"(\w+)'s": r"\1 is",
        r"(\w+)'ll": r"\1 will",
        # To have contractions
        r"(\w+)'ve": r"\1 have",
        # Negative contractions
        r"can't": "cannot",
        r"won't": "will not",
        r"n't": " not",
        # Other common contractions
        r"(\w+)'d": r"\1 would",
        # Possessive apostrophes (optional, remove if not needed)
        r"(\w+)'s\b": r"\1's",
    }

    # Apply each contraction replacement
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


special_character = open("/data/khanhnd65/special_character.txt").read().splitlines()


def remove_special_char(x):
    x = x.lower()
    for i in special_character:
        if x in ["'"]:
            x = x.replace(i, "")
        else:
            x = x.replace(i, " ")
    x = " ".join(x.split())
    return x


valid_char = open("/data/khanhnd65/valid_character.txt").read().splitlines() + [
    " ",
    "'",
]


def keep_valid_char(x):
    lst = list(x)
    res = []
    for i in lst:
        if i in valid_char:
            res.append(i)
    return "".join(res)


def convert_num2txt(x):
    x = x.split()
    res = []
    for i in x:
        if i.isdigit():
            res.append(lex_of_number(i))
        else:
            res.append(i)
    return " ".join(res)


if __name__ == "__main__":
    print(keep_valid_char("hello's"))
