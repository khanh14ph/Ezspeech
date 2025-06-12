import sentencepiece as spm
import orjson
import random
import json
from tqdm import tqdm
import re

# i = re.sub('<[^>]*>', '', i)

# import glob
# from ezspeech.modules.dataset.utils.text import normalize
# lst=glob.glob("/home4/khanhnd/librispeech-lm-corpus/corpus/*/*")
# final=[]
# for i in tqdm(lst):
#     final.extend(open(i).read().split("\n"))

# final=[i for i in final if i!=""]
# with open("/home4/khanhnd/Ezspeech/ezspeech/resource/corpus/librispeech_upper.txt","w") as f:
#     for i in tqdm(final):
#         f.write(normalize(i).upper()+"\n")
# spm.SentencePieceTrainer.train(
#     input="/home4/khanhnd/Ezspeech/ezspeech/resource/corpus/librispeech_upper.txt",
#     model_prefix="/home4/khanhnd/Ezspeech/ezspeech/resource/vocab/bpe/en_upper",
#     model_type="bpe",
#     user_defined_symbols=[],
#     vocab_size=1024,
#     input_sentence_size=10000000,
#     train_extremely_large_corpus=True,
#     shuffle_input_sentence=True,
#     character_coverage=0.985,
#     treat_whitespace_as_suffix=True,
#     # unk_surface="<unk>",
#     # pad_id=0,
#     # unk_id=1,
#     # bos_id=2,
#     # eos_id=3,
# )
lst = (
    open("/home4/khanhnd/Ezspeech/ezspeech/resource/vocab/en_upper.vocab")
    .read()
    .splitlines()
)
lst = [i.split()[0] for i in lst]
with open("/home4/khanhnd/Ezspeech/ezspeech/resource/vocab/en_upper.txt", "w") as f:
    for i in lst:
        f.write(i + "\n")
