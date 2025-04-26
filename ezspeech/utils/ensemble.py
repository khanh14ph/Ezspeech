import pandas as pd
from tqdm import tqdm
from kaldialign import align
import unicodedata

vn_word_lst = set(open("/data/khanhnd65/eval/vn_word_lst.txt").read().split("\n")[:-1])
sure_vn_word_lst = set(open("/data/khanhnd65/sure_vn_word.txt").read().split("\n")[:-1])
en_words = set(
    open("/data/khanhnd65/transcription-quality-assesment/all_en_word.txt")
    .read()
    .split("\n")[:-1]
)
oov_hallucinate = ["subscribe", "like", "share", "lalaschool"]

special_character = (
    open("/data/khanhnd65/special_character.txt").read().split("\n")[:-1]
)


def normalize(x):
    x = x.lower()
    for i in special_character:
        x = x.replace(i, " ")
    x = " ".join(x.split())
    return x


def check_hallucinate(x):
    x = x.split()
    for i in x:
        if i in oov_hallucinate:
            return True
    return False


def remove_accents(input_str):
    normalized_str = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in normalized_str if not unicodedata.combining(c)])


def remove_accent_align(lst):
    res = []
    for i in lst:
        if i[0] != remove_accents(i[1]):
            res.append(i)
    return res


def no_elements_in_common(list1, list2):
    return all(item not in list2 for item in list1)


def handle_overlap_align(lst):
    index2word_dict = dict()
    for i in lst:
        temp = index2word_dict.get(i[2], [])
        temp.append(i)
        index2word_dict[i[2]] = temp
    new_lst = []
    for i in index2word_dict.keys():
        choose = None
        two_choice = index2word_dict[i]
        if len(two_choice) > 1:
            if is_better(two_choice[0], two_choice[1]):
                choose = two_choice[0]
            else:
                choose = two_choice[1]
        else:
            choose = two_choice[0]
        new_lst.append(choose)
    return new_lst


def filter_bad_word(new_lst):
    filter_lst = []
    for word, spell, sindex, _ in new_lst:
        word_lst = word.split()
        if len(word_lst) == 1:
            if word not in sure_vn_word_lst:
                filter_lst.append((word.strip(), spell.strip(), sindex, _))

        else:
            if no_elements_in_common(word_lst, vn_word_lst):
                new_word = "-".join(word_lst)
                filter_lst.append((new_word.strip(), spell.strip(), sindex, _))
    return filter_lst


def remove_too_close_align(filter_lst):
    final_lst = []
    if len(filter_lst) > 0:
        final_lst = [filter_lst[0]]
        for i in range(len(filter_lst))[1:]:
            if filter_lst[i][2] >= filter_lst[i - 1][2] + len(filter_lst[i - 1][1]):
                final_lst.append(filter_lst[i])
    return final_lst


def compute_score(x):
    score = 0
    if x[0] in en_words:
        score += 5
    if x[3] == "whisper":
        score += 5
    return score


def is_better(x, y):
    if compute_score(x) > compute_score(y):
        return True
    elif compute_score(x) < compute_score(y):
        return False
    else:
        if x[0] in y[0]:
            return False
        if y[0] in x[0]:
            return True
        else:
            return len(y[0].split()) < len(x[0].split())


def align_two_string(x, y, model):
    EPS = "*"
    x = x.split()
    y = y.split()
    ali = align(x, y, EPS)
    lst = []
    word = ""
    spell = ""
    offset = 0
    ali.append((".", "."))
    for pair in ali:
        fix = 0
        if pair[1] == "*":
            fix = 2
        if pair[0] == pair[1]:

            if len(word) == 0:
                offset = offset + len(pair[1] + " ") - fix
                continue
            else:
                word = word.replace("*", "").strip()
                spell = spell.replace("*", "").strip()
                if word != "" and spell != "":
                    lst.append(
                        [
                            word.strip(),
                            spell.strip(),
                            offset - len(spell.strip()),
                            model,
                        ]
                    )
                word = ""
                spell = ""
        else:
            word += " " + pair[0]
            spell += " " + pair[1]
        offset = offset + len(pair[1] + " ") - fix
    return lst


new_transcript = []
new_pred = []


# df = df.loc[
#     df["audio_filepath"]
#     == "/data/datbt7/dataset/speech/16k/gigaspeech2/train/31/324/31-324-18.wav"
# ]


def merge(best_vn, best_en, second_best_en=None):
    b = best_vn
    a = best_en
    b = " ".join(b.replace("]", "").replace("[", "").split())
    c = second_best_en
    lst_whisper = align_two_string(a, b, "whisper")
    if second_best_en != None:
        lst_ctc = align_two_string(c, b, "ctc")
    else:
        lst_ctc = []

    if check_hallucinate(a):

        lst = lst_ctc
    else:
        lst = lst_whisper + lst_ctc

    # Handle case with same index
    lst = filter_bad_word(lst)
    lst = handle_overlap_align(lst)

    lst = remove_accent_align(lst)

    lst = sorted(lst, key=lambda x: x[2])
    lst = remove_too_close_align(lst)
    offset = 0
    pred = b
    for word, spell, index, _ in lst:
        move_offset = offset + index
        b = b[: move_offset - 1] + word + b[move_offset - 1 + len(spell) :]

        offset = offset + len(word) - len(spell)
    b = " ".join(b.split())
    return b
