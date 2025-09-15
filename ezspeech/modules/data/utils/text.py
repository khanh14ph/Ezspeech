import re
from typing import List, Optional, Tuple, Union

import sentencepiece as spm


def tokenize(sentence: str, vocab: List[str]) -> List[str]:
    # sentence = re.sub(r"\s+", "|", sentence)
    # sentence = sentence.strip("|")
    sentence = sentence.replace(" ", "_") + "_"

    patterns = "|".join(map(re.escape, sorted(vocab, reverse=True)))
    tokens = re.findall(patterns, sentence)
    return tokens


class Tokenizer:
    def __init__(self,spe_file=None):
        super().__init__()
        self.spe_file = spe_file
    
        self.spe_model = spm.SentencePieceProcessor(model_file=spe_file)
        self.vocab = self.spe_model.id_to_piece(
                [i for i in range(len(self.spe_model))]
            )

    def encode(self, sentence):
        token_idx = self.spe_model.encode(sentence)
        return token_idx

    def decode(self, idx_lst):
        return self.spe_model.decode(idx_lst)

    def __len__(self):
        return len(self.vocab)


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




def normalize(x):

    for i in special_symbol:
        x = x.replace(i, " ")
    x = " ".join(x.split())
    return x


if __name__ == "__main__":
    vocab = (
        open("/home4/khanhnd/Ezspeech/ezspeech/resource/vocab/vi_en.txt")
        .read()
        .splitlines()
    )
    res = tokenize("xin chào tôi là người đẳng cấp PRO VIP ENTERTAINMENT", vocab)
    print(res)
