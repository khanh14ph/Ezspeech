import json
from typing import Union, List
def save_json(dictionary,path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)
def to_jsonl(x: List[dict], filepath: str):

    with open(filepath, "w", encoding="utf8") as outfile:

        for entry in x:

            json.dump(entry, outfile, ensure_ascii=False)

            outfile.write("\n")
def load_dataset(filepaths: Union[str, List[str]]) -> List[dict]:

    if isinstance(filepaths, str):

        filepaths = [filepaths]

    dataset = []

    for filepath in filepaths:

        with open(filepath, encoding="utf-8") as datas:

            dataset += [json.loads(d) for d in datas]



    return dataset