import re
from typing import List, Dict, Optional
from collections import defaultdict
from importlib_resources import files

DELIMITER = "▁"
VOWELS = "aăâeêioôơuưy"
TONE_CHARS = "àằầèềìòồờùừỳáắấéếíóốớúứýảẳẩẻểỉỏổởủửỷạặậẹệịọộợụựỵãẵẫẽễĩõỗỡũữỹ"
TONE_MARKS = ["1_", "2_", "3_", "4_", "5_"]
SPECIAL_SUBWORDS = [
    "uôc",
    "uych",
    "uyn",
    "uynh",
    "uyp",
    "uyt",
    "uyên",
    "uyêt",
    "i",
    "in",
    "iêt",
    "iêu",
    "iêng",
    "uôc_",
    "uych_",
    "uyn_",
    "uynh_",
    "uyp_",
    "uyt_",
    "uyên_",
    "uyêt_",
    "i_",
    "in_",
    "iêt_",
    "iêu_",
    "iêng_",
]
import json
import random
from tqdm import tqdm
def load_dataset(filepaths):
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    dataset = []
    for filepath in filepaths:
        with open(filepath, encoding="utf-8") as datas:
            dataset += [json.loads(d) for d in tqdm(datas,desc="Loading metadata file..")]

    return dataset

def build_vocab(filepath=None) -> List[str]:
    if filepath==None:
        vocab = files("lightspeech.corpus").joinpath("vocab.txt")
        vocab = vocab.read_text("utf-8").split("\n")
    else:
        vocab=open(filepath).read().split("\n")[:-1]
    return vocab


def build_lexicon(lexicon_file=None) -> Dict[str, List[str]]:
    if lexicon_file == None:
        lexicon = files("lightspeech.corpus").joinpath("all_lexicon.txt")
        lexicon = lexicon.read_text("utf-8").split("\n")[:-1]
    else:
        lexicon = open(lexicon_file).read().split("\n")[:-1]

    lexicon = [line.split("\t", 1) for line in lexicon]
    lexicon = {item[0]: item[1].split(" ") for item in lexicon}
    return lexicon


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


lexicon_build = dict()


def tokenize(sentence: str, vocab: List[str]) -> List[str]:
    sentence = re.sub(r"\s+", "|", sentence)
    sentence = sentence.strip("|")

    patterns = "|".join(map(re.escape, sorted(vocab, reverse=True)))
    tokens = re.findall(patterns, sentence)
    return tokens


# def decode(x):
#     token_lst = x.split()
#     s = ""
#     res = []
#     for i in token_lst:
#         s += i
#         if check_end_word(i, build_vocab()):

#             res.append(s)
#             s = ""
#     return " ".join(res)


def gen_lexicon():
    vocab = build_vocab()
    lexicon = build_lexicon()
    consonants = open("consonants.txt").read().splitlines()
    # consonants=[" ".join(list(i)) for i in consonants]
    with open("temp.txt", "w") as f:
        for word in lexicon.keys():
            tokens = tokenize(word, vocab)
            if len(tokens) == 1:
                for consonant in ["w", "j", "z", "f", "p"]:
                    word_ = consonant + word
                    spell_ = " ".join([" ".join(list(consonant))] + tokens)
                    f.write(f"{word_}\t{spell_}\n")
            else:
                for consonant in consonants:
                    if tokens[0] == consonant[-len(tokens[0]) :]:
                        word_ = consonant + tokens[1]
                        spell_ = " ".join([" ".join(list(consonant))] + tokens[1:])
                        f.write(f"{word_}\t{spell_}\n")


from tqdm import tqdm


class TextCorpus:
    def __init__(self, corpus: str, vocab: List[str] = []) -> None:
        self.corpus = corpus
        self.vocab = vocab

        self.word_freqs = self.create_word_dict(open(corpus).read().split("\n")[:-1])

    def create_new_vocab(self, vocab_size: int) -> List[int]:
        if vocab_size < len(self.vocab):
            raise Exception("New vocab size is too small")

        # Split alphabet
        vocab = self.vocab.copy()
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in vocab:
                    vocab.append(letter)

        splits = dict()
        for word in tqdm(self.word_freqs.keys()):
            spell = tokenize(word, build_vocab())
            if len(spell) > 0:
                splits[word] = spell
        while len(vocab) < vocab_size:
            scores = self.compute_pair_scores(splits)
            best_pair, max_score = "", None

            for pair, score in scores.items():
                if max_score is None or max_score < score:
                    best_pair = pair
                    max_score = score

            splits = self.merge_pair(*best_pair, splits)
            new_token = best_pair[0] + best_pair[1]
            if new_token not in vocab:
                print("\r" + str(len(vocab)))
                vocab.append(new_token)

        return vocab

    def create_word_dict(self, corpus: List[str]) -> Dict:
        # Create word dicts
        word_freqs = defaultdict(int)

        for text in corpus:
            text = text.lower().strip()
            words = text.split()
            for word in words:
                if word != "":
                    word_freqs[word] += 1

        return word_freqs

    def compute_pair_scores(self, splits: Dict) -> float:
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = splits[word]

            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue

            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        scores = {pair: freq for pair, freq in pair_freqs.items()}
        # scores = {
        #     pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]]) ** 0.5
        #     for pair, freq in pair_freqs.items()
        # }
        return scores

    def merge_pair(self, a: str, b: str, splits: Dict) -> Dict:
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1

            splits[word] = split

        return splits


import multiprocessing


def multiprocess_apply(input_list, function, num_processes=None):

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the function to the input list
        results = list(
            tqdm(
                pool.imap(function, input_list),
                total=len(input_list),
                desc="Processing",
                unit="item",
            )
        )

    return results

def save_dataset(lst,filepaths):
    with open(filepaths, 'w', encoding='utf-8') as f:
        for item in lst:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
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
special_symbol=[",",".","?","$","@","-","_","(",")","*","!","#","/"]
# Cài đặt: pip install num2words
import re
from num2words import num2words

def convert_numbers_in_text_en(text):
    # Tìm tất cả các số trong văn bản
    def replace_num(match):
        number = int(match.group(0))
        return num2words(number)

    # Sử dụng regex để tìm và thay thế các số
    return re.sub(r'\b\d+\b', replace_num, text)


def normalize(transcript,lan_id):
    x=transcript
    if lan_id=="en":
        x=x.replace("%"," percent")
        x=convert_numbers_in_text_en(x)
    if lan_id=="vi":
        x=x.replace("%"," phần trăm")
        x=convert_numbers_in_text_vi(x)

    for i in special_symbol:
        x=x.replace(i," ")
    x=" ".join(x.split())
    return x
import regex as re

def is_english_vietnamese_only(text):
    # Loại bỏ dấu cách và dấu câu
    # text_without_punctuation = re.sub(r'[\s\p{P}0-9]', '', text)

    # Mẫu regex cho ký tự tiếng Anh và tiếng Việt
    # \p{Latin} bao gồm tất cả các ký tự Latin (bao gồm cả tiếng Việt)
    pattern = r'^[\p{Latin}]+$'

    return bool(re.match(pattern, text))
if __name__ == "__main__":
    # import fireducks.pandas as pd
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file='/raid/voice/khanhnd65/lightspeech_khanhnd/resources/tokenizer/en_vi_2048.model')
    vocab=build_vocab("/raid/voice/khanhnd65/lightspeech_khanhnd/src/lightspeech/corpus/vocab_code_switching.txt")
    # for i in range(1000):


    # a = load_dataset(
    #     "/raid/voice/khanhnd65/dataset_everything/train_010425_hf.jsonl"
    # )
    # for df in tqdm(a):
    #     df["frame_length"]=df["duration"]/0.04
    #     df["raw_transcript"]=df["transcript"]
    #     df["transcript"]=normalize(df["transcript"],df["lan_id"])
    #     df['check_var'] = is_english_vietnamese_only(df["transcript"])
    # # df["token_length"]=df["transcript"].progress_apply(lambda x: len(tokenize(x,vocab)))
    # # df.to_json('/raid/voice/khanhnd65/dataset_everything/train_010425_hf_1.jsonl', orient='records', lines=True)
    # save_dataset(a,"/raid/voice/khanhnd65/dataset_everything/train_010425_hf_1.jsonl")
    print(is_english_vietnamese_only("Ü"))
