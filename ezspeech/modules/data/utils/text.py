import re
from typing import List, Optional, Tuple, Union
import sentencepiece as spm

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
    def id_to_token(self,idx):
        return self.spe_model.id_to_piece(idx)
    def decode_list(self, idx_lst):
        res=[]
        for i in idx_lst:
            if i!=0:
                res.append(self.id_to_token(i))
        return res
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


import re
try:
    # UCS-4
    highpoints = re.compile(u'[\U00010000-\U0010ffff]')
except re.error:
    # UCS-2
    highpoints = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
special_symbol=open("ezspeech/resource/special_char.txt").read().splitlines()
def normalize(x):
    x=x.replace('\u200b', '').replace("°c","độ c")
    for i in special_symbol:
        
        x = x.replace(i, " ")
    x = " ".join(x.split())
    x=highpoints.sub(u'\u25FD', x)
    return x



