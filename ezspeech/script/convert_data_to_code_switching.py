from ezspeech.moduels.dataset.utils.text import tokenize
from ezspeech.utils.common import load_dataset, save_dataset

vn_word_lst = set(open("ezspeech/resource/vn_word_lst.txt").read().splitlines())
a = load_dataset()


def text_to_codeswitch(x):
    x = x.split()
    res = []
    for i in x:
        if i not in vn_word_lst:
            i = i.upper()
        res.append(i)
    return " ".join(res)


for i in a:
    if i["lan_id"] == "en":
        i["transcript"] = i["transcript"].upper()
    else:
        i["transcript"] = text_to_codeswitch(i["transcript"])
save_dataset(a, "path_to_dest")
