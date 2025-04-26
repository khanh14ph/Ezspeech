import re
from typing import List, Dict


import torch

from lightspeech.datas.text import build_vocab, build_lexicon


class OOVRecognizer(object):
    def __init__(self, oov_filepath: str):
        self.vocab = build_vocab()
        self.vocab_size = len(self.vocab)

        self.lexicon = build_lexicon()

        # Initialize OOV Sound-Like Capture
        self.oov_soundlikes = self.parse_oov_data(oov_filepath)

    def parse_oov_data(self, oov_filepath: str) -> Dict[str, str]:
        with open(oov_filepath) as f:
            oov_datas = f.read().splitlines()

        oov_soundlikes = {}
        for data in oov_datas:
            columns = data.split("\t")
            oov_word = columns[1].strip()

            if len(columns) == 2:
                soundlike = columns[0].split("<>")
                soundlike = [sound.strip() for sound in soundlike]
                oov_soundlike = {sound: oov_word for sound in soundlike}
                oov_soundlikes.update(oov_soundlike)

        oov_soundlikes = dict(
            sorted(oov_soundlikes.items(), key=lambda item: len(item[0].split(" ")))
        )

        return oov_soundlikes

    def capture_soundlike(self, sentence: str) -> str:
        for soundlike, oov in self.oov_soundlikes.items():
            sentence = re.sub(rf"\b{soundlike}\b", oov, sentence)
        return sentence


import json


class ReweightEmission(object):
    def __init__(self):
        self.vocab = (
            open("/data/khanhnd65/lightspeech/src/lightspeech/corpus/vocab.txt")
            .read()
            .splitlines()
        )
        self.vocab_size = len(self.vocab)
        self.lexicon = build_lexicon()

        self.tag = ["<<", ">>"]
        self.character = "abcdefghijklmnopqrstuvwxyz"
        self.blank = ["-"]
        self.slient = ["|"]
        with open(
            "/data/khanhnd65/rescore-asr/analyze/token_level/frequency_correct.json",
            "r",
        ) as f:
            data = list(json.load(f).keys())
        correct_tokens = data[0:20]
        with open(
            "/data/khanhnd65/rescore-asr/analyze/token_level/frequency_insert.json",
            "r",
        ) as f:
            self.boost_token = list(json.load(f).keys())
        with open(
            "/data/khanhnd65/rescore-asr/analyze/token_level/frequency_replace.json",
            "r",
        ) as f:
            data = json.load(f)
            replace_tokens = list(data.keys())[0:100]
        replace_all = [(i.split("_")[0], i.split("_")[1]) for i in replace_tokens]
        self.replace_dict = dict()
        for i in replace_all:
            if i[0] not in correct_tokens:
                temp = self.replace_dict.get(self.vocab.index(i[0]), [])
                temp.append(self.vocab.index(i[1]))
                self.replace_dict[self.vocab.index(i[0])] = temp
        print(self.replace_dict)

    def create_mask(self, tokens: List[str]) -> torch.Tensor:
        tokens = torch.tensor([self.vocab.index(_) for _ in tokens])
        mask = torch.zeros(self.vocab_size)
        mask[tokens] = True
        return mask.bool()

    def boost_replace(self, emissions: torch.Tensor, temperature: int) -> torch.Tensor:

        # Convert to probability
        probs = emissions.exp()
        # print(sum(probs[0][0]))
        argmax = torch.argmax(probs, dim=-1)
        for idx_batch, batch in enumerate(probs):
            for idx_frame, frame in enumerate(batch):

                get_replace_candidate = self.replace_dict.get(
                    int(argmax[idx_batch][idx_frame]), None
                )
                probs[idx_batch][idx_frame] = probs[idx_batch][idx_frame] * temperature
                probs[idx_batch][idx_frame][int(argmax[idx_batch][idx_frame])] = (
                    probs[idx_batch][idx_frame][int(argmax[idx_batch][idx_frame])]
                    / temperature
                )
                # if get_replace_candidate:
                #     for v in get_replace_candidate:
                #         # print(probs[idx_batch][idx_frame][v])

                #         probs[idx_batch][idx_frame][v] = (
                #             probs[idx_batch][idx_frame][v] * 1.2
                #         )
        probs = probs * (1 / torch.sum(probs, dim=-1).unsqueeze(-1))
        probs = probs.log()
        return probs

    def boost(
        self,
        emissions: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:

        # Convert to probability
        probs = emissions.exp()

        # Split probability to two parts
        mask = self.create_mask(self.boost_token)
        diminish_part = probs.masked_fill(mask, value=0.0)
        hold_part = probs.masked_fill(~mask, value=0.0)

        diminish_part /= temperature
        probs = diminish_part + hold_part
        emissions = probs.log()

        return emissions

    def smooth(
        self,
        emissions: torch.Tensor,
        tokens: List[str],
        temperature: float,
    ):
        # Split probability to two parts
        mask = self.create_mask(tokens)
        hold_part = emissions.masked_fill(mask, value=0.0)
        smooth_part = emissions.masked_fill(~mask, value=0.0)

        # Get new emissions
        smooth_part /= temperature
        emissions = smooth_part + hold_part

        return emissions

    def norm(self, emissions: torch.Tensor) -> torch.Tensor:
        # Scale to 1
        logits = emissions.exp()
        logits = logits * (1 / torch.sum(logits, dim=-1).unsqueeze(-1))
        emissions = logits.log()

        return emissions

    def __call__(self, emissions: torch.Tensor):
        # Boost and smooth all tokens
        # smooth_tokens = self.vocab
        # emissions = self.smooth(emissions, smooth_tokens, 1.2)
        # emissions = self.norm(emissions)

        # # Boost OOV tokens
        # boost_tokens = [*self.character]
        # emissions = self.boost(emissions, boost_tokens, 1.5)
        # emissions = self.norm(emissions)

        # Diminish blank score
        emissions = self.diminish(emissions, self.blank, 1.0)
        emissions = self.norm(emissions)

        # # Diminish tag score
        # emissions = self.diminish(emissions, self.tag, 0.2)
        # emissions = self.norm(emissions)
        return emissions
