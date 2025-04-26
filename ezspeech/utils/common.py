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


def to_jsonl(x: List[dict], filepath: str):
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

def get_wave_duration(audio_filepath: str):
    with contextlib.closing(wave.open(audio_filepath, "r")) as f:
        num_frames = f.getnframes()
        sample_rate = f.getframerate()
        duration = num_frames / sample_rate
    return duration





def compute_statistic(
    xs: torch.Tensor, x_lens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    masks = make_padding_mask(x_lens, xs.size(1))
    masks = masks[:, :, None]

    T = masks.sum(1)
    mean = (xs * masks).sum(1) / T
    std = (((xs - mean.unsqueeze(1)) ** 2 * masks).sum(1) / T).sqrt()

    return mean, std


def make_padding_mask(seq_lens: torch.Tensor, max_time: int) -> torch.Tensor:
    bs = seq_lens.size(0)
    device = seq_lens.device

    seq_range = torch.arange(0, max_time, dtype=torch.long, device=device)
    seq_range = seq_range.unsqueeze(0).expand(bs, max_time)

    seq_length = seq_lens.unsqueeze(-1)
    mask = seq_range < seq_length

    return mask


def length_regulator(
    xs: torch.Tensor,
    x_masks: torch.Tensor,
    durs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    y_lens = durs.sum(dim=1)
    y_masks = make_padding_mask(y_lens, y_lens.amax())

    b_x, t_x = x_masks.shape
    b_y, t_y = y_masks.shape
    assert b_x == b_y, "Batch size dimension isn't match"

    cum_durs = torch.cumsum(durs, 1).contiguous().view(b_x * t_x)
    alignment = make_padding_mask(cum_durs, t_y)
    alignment = alignment.contiguous().view(b_x, t_x, t_y).int()
    alignment = alignment - F.pad(alignment, [0, 0, 1, 0, 0, 0])[:, :-1]
    alignment = alignment * x_masks.unsqueeze(2) * y_masks.unsqueeze(1)
    alignment = alignment.type(xs.dtype)

    ys = torch.matmul(alignment.transpose(1, 2), xs)

    return ys, y_lens


def word_level_pooling(
    xs: torch.Tensor,
    wbs: torch.Tensor,
    reduction: Optional[str] = "sum",
) -> torch.Tensor:

    B, Tp, D = xs.size()

    Tw = wbs.amax() + 1
    wbs = wbs.masked_fill(wbs < 0, Tw)

    ys = xs.new_zeros([B, Tw + 1, D])
    ys = ys.scatter_add_(1, wbs[:, :, None].repeat([1, 1, D]), xs)
    ys = ys[:, :-1].contiguous()

    if reduction == "mean":
        ones = xs.new_ones(xs.shape[:2])
        N = xs.new_zeros([B, Tw + 1])
        N = N.scatter_add_(1, wbs, ones)
        N = N[:, :-1].contiguous()
        ys = ys / torch.clamp(N[:, :, None], min=1)

    return ys


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


def logbeta(x, y):
    return gammaln(x) + gammaln(y) - gammaln(x + y)


def logcombinations(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def logbetabinom(n, a, b, x):
    return logcombinations(n, x) + logbeta(x + a, n - x + b) - logbeta(a, b)


def beta_binomial_prior_distribution(
    num_tokens: int, num_frames: int, scaling_factor: float = 1.0
):
    x = torch.arange(0, num_tokens).unsqueeze(0)
    y = torch.arange(1, num_frames + 1).unsqueeze(1)
    a = scaling_factor * y
    b = scaling_factor * (num_frames + 1 - y)
    n = torch.FloatTensor([num_tokens - 1])
    return logbetabinom(n, a, b, x)


def is_flash_attn_supported() -> bool:

    if not torch.cuda.is_available():
        return False

    # Approximate list of supported GPUs, might be missing some
    AMPERE_GPUs = ["RTX 20", "RTX 30", "A10", "A20", "A30", "A40", "A50", "A60"]
    ADA_GPUs = ["RTX 40", "RTX 45", "RTX 50", "RTX 60", "L4"]
    HOPPER_GPUs = ["H100", "H200"]

    SUPPORTED_GPUs = AMPERE_GPUs + ADA_GPUs + HOPPER_GPUs

    gpu_name = torch.cuda.get_device_name()
    for supported in SUPPORTED_GPUs:
        if supported in gpu_name:
            return True
