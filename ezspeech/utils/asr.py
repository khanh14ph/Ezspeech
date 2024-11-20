import json
def load_vocab(path):
    vocab_dict=json.load(open(path))
    vocab_lst = sorted(vocab_dict.keys(), key=lambda x: vocab_dict[x])
    return vocab_lst