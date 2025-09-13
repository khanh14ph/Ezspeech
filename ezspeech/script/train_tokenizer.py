import json
import os
import random
import re

import orjson
import sentencepiece as spm
from tqdm import tqdm

# import glob
from ezspeech.utils.common import load_dataset

# i = re.sub('<[^>]*>', '', i)

a = load_dataset("/home4/khanhnd/Ezspeech/data/vlsp2020.jsonl")
with open("temp.txt", "w") as f:
    for i in tqdm(a):
        f.write(i["text"].strip() + "\n")
os.makedirs("/home4/khanhnd/Ezspeech/ezspeech/resource/tokenizer/vi", exist_ok=True)
spm.SentencePieceTrainer.train(
    input="temp.txt",
    model_prefix="/home4/khanhnd/Ezspeech/ezspeech/resource/tokenizer/vi/tokenizer",
    model_type="bpe",
    user_defined_symbols=[],
    vocab_size=1024,
    input_sentence_size=10000000,
    train_extremely_large_corpus=True,
    shuffle_input_sentence=True,
    character_coverage=0.999,
    treat_whitespace_as_suffix=True,
    # unk_surface="<unk>",
    # pad_id=0,
    # unk_id=1,
    # bos_id=2,
    # eos_id=3,
)
import os

os.remove("temp.txt")
