from typing import Tuple, List, Union, Optional
import re


def tokenize(sentence: str, vocab: List[str]) -> List[str]:
    sentence = re.sub(r"\s+", "|", sentence)
    sentence = sentence.strip("|")

    patterns = "|".join(map(re.escape, sorted(vocab, reverse=True)))
    tokens = re.findall(patterns, sentence)
    return tokens


VOWELS = "aăâeêioôơuưy"
TONE_CHARS = "àằầèềìòồờùừỳáắấéếíóốớúứýảẳẩẻểỉỏổởủửỷạặậẹệịọộợụựỵãẵẫẽễĩõỗỡũữỹ"


def check_end_word(token: str, vocab: List[str]):
    """_summary_

    Args:
        token (str): _description_
        vocab (List[str]): _description_

    Returns:
        _type_: _description_
    """
    if token == "":
        return False

    elif token[0] in VOWELS or token[0] in TONE_CHARS and token in vocab:
        return True
    elif token in ["[", "]", "<f>"]:
        return True
    else:
        return False


def number_to_vietnamese_words(number):
    """
    Convert a number to Vietnamese spoken form.
    Handles integers up to 999,999,999,999 (trillions)
    """
    if number == 0:
        return "không"

    # Vietnamese digits
    digits = ["", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

    # Special cases for tens
    special_tens = {
        "linh",
        "mười",
        "lăm",  # Used when 5 is in the ones place after 10
    }

    # Function to convert a group of 3 digits
    def convert_group(num):
        result = ""

        # Hundreds place
        hundreds = num // 100
        if hundreds > 0:
            result += digits[hundreds] + " trăm "

        # Tens and ones place
        remainder = num % 100
        if remainder > 0:
            # Special case for numbers 1-9 when preceded by hundred
            if remainder < 10 and hundreds > 0:
                result += "linh "

            # Tens digit
            tens = remainder // 10
            ones = remainder % 10

            if tens == 1:
                result += "mười "
                if ones == 5:
                    result += "lăm"
                elif ones > 0:
                    result += digits[ones]
            elif tens > 1:
                result += digits[tens] + " mươi "
                if ones == 1:
                    result += "mốt"
                elif ones == 5:
                    result += "lăm"
                elif ones > 0:
                    result += digits[ones]
            else:  # tens == 0
                result += digits[ones]

        return result.strip()

    # Process number by groups of 3 digits
    units = ["", " nghìn", " triệu", " tỷ", " nghìn tỷ", " triệu tỷ"]
    result = ""
    unit_index = 0

    while number > 0:
        group = number % 1000
        if group > 0:
            group_text = convert_group(group)
            result = group_text + units[unit_index] + " " + result

        number //= 1000
        unit_index += 1

    return result.strip()


def convert_numbers_in_text_vi(text):
    """
    Find numeric patterns in text and replace them with Vietnamese spoken form.
    """
    import re

    # Function to convert matched numbers
    def replace_match(match):
        num_str = match.group(0)
        # Remove commas and spaces
        clean_num = num_str.replace(",", "").replace(".", "").replace(" ", "")
        try:
            number = int(clean_num)
            return number_to_vietnamese_words(number)
        except ValueError:
            return num_str  # Return original if not convertible

    # Pattern to match numbers (with optional commas/periods as thousand separators)
    pattern = r"\b\d{1,3}(?:[,. ]?\d{3})*\b"

    # Replace all matched numbers
    return re.sub(pattern, replace_match, text)


special_symbol = [",", ".", "?", "$", "@", "-", "_", "(", ")", "*", "!", "#", "/"]
# Cài đặt: pip install num2words
import re
from num2words import num2words


def convert_numbers_in_text_en(text):
    # Tìm tất cả các số trong văn bản
    def replace_num(match):
        number = int(match.group(0))
        return num2words(number)

    # Sử dụng regex để tìm và thay thế các số
    return re.sub(r"\b\d+\b", replace_num, text)


def normalize(x):

    for i in special_symbol:
        x = x.replace(i, " ")
    x = " ".join(x.split())
    return x
