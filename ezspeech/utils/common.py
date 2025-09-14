import contextlib
import json
import wave
from contextlib import nullcontext
from typing import List, Optional, OrderedDict, Tuple, Union

import orjson
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.special import gammaln
from tqdm import tqdm


def save_dataset(x: List[dict], filepath: str):
    with open(filepath, "w", encoding="utf8") as outfile:
        for entry in tqdm(x):
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


def compute_statistic(
    xs: torch.Tensor, x_lens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    masks = make_padding_mask(x_lens, xs.size(1))
    masks = masks[:, :, None]

    T = masks.sum(1)
    mean = (xs * masks).sum(1) / T
    std = (((xs - mean.unsqueeze(1)) ** 2 * masks).sum(1) / T).sqrt()

    return mean, std


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
    net.eval()
    net.load_state_dict(weights)
    net.to(device)

    for param in net.parameters():
        param.requires_grad = False

    return net


def csv2json(csv_path, jsonl_path, sep=",", replace_columns=None):
    import pandas as pd

    df = pd.read_csv(csv_path, sep=sep)
    if replace_columns != None:
        df.rename(columns=replace_columns, inplace=True)

    df = list(df.T.to_dict().values())
    save_dataset(df, jsonl_path)


def avoid_float16_autocast_context():
    """
    If the current autocast context is float16, cast it to bfloat16
    if available (unless we're in jit) or float32
    """

    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return torch.amp.autocast("cuda", dtype=torch.float32)

        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        else:
            return torch.amp.autocast("cuda", dtype=torch.float32)
    else:
        return nullcontext()


if __name__ == "__main__":
    import pandas as pd
    import librosa
    a=load_dataset("train_vietbud.jsonl")
    for i in a:
        text=i["text"]
        i["text"]=i["text"].lower().replace(",","").replace(".","").replace("?","").replace("!","").replace(";","").replace(":","").replace("\"","").replace("`","").replace("(","").replace(")","").replace("[","").replace("]","").replace("{","").replace("}","")
        if text!=i["text"]:
            print(f"{text} -> {i['text']}")
    save_dataset(a,"train_vietbud.jsonl")