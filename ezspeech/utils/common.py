import json
from typing import List, Tuple, Union, OrderedDict, Optional
import wave
import contextlib


from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm
import torch
import orjson
import torch.nn.functional as F
from torch.special import gammaln


def save_dataset(x: List[dict], filepath: str):
    with open(filepath, "w", encoding="utf8") as outfile:
        for entry in x:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")
def load_dataset(filepaths: Union[str, List[str]]) -> List[dict]:
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    dataset = []
    for filepath in tqdm(filepaths):
        with open(filepath, encoding="utf-8") as datas:
            dataset += [orjson.loads(d) for d in tqdm(datas)]

    return dataset

def make_padding_mask(seq_lens: torch.Tensor, max_time: int) -> torch.Tensor:
    bs = seq_lens.size(0)
    device = seq_lens.device

    seq_range = torch.arange(0, max_time, dtype=torch.long, device=device)
    seq_range = seq_range.unsqueeze(0).expand(bs, max_time)

    seq_length = seq_lens.unsqueeze(-1)
    mask = seq_range < seq_length

    return mask

def time_reduction(
    xs: torch.Tensor, x_lens: torch.Tensor, stride: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    b, t, d = xs.shape
    n = t + (stride - t % stride) % stride
    p = n - t

    xs = F.pad(xs, (0, 0, 0, p))
    xs = xs.reshape(b, n // stride, d * stride).contiguous()

    x_lens = torch.div(x_lens - 1, stride, rounding_mode="trunc")
    x_lens = (x_lens + 1).type(torch.long)

    return xs, x_lens
def load_module(
    hparams: DictConfig, weights: OrderedDict, device: Optional[str] = "cpu"
) -> torch.nn.Module:

    net = instantiate(hparams)
    net.load_state_dict(weights)
    net.to(device)

    net.eval()
    for param in net.parameters():
        param.requires_grad = False

    return net

def csv2json(csv_path,jsonl_path,sep=","):
    import pandas as pd 
    df=pd.read_csv(csv_path,sep=sep)
    df.to_json(jsonl_path,orient="records",lines=True,force_ascii=False)    

if __name__ == "__main__":
    import fire
    fire.Fire()