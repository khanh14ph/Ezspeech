from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.nn.utils import weight_norm


LRELU_SLOPE = 0.2
SEGMENT_SIZE = 8192


class PWD(nn.Module):
    def __init__(self, period: int):
        super(PWD, self).__init__()
        self.period = period
        self.layers = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 64, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(64, 128, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(128, 256, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(256, 512, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(512, 1024, (5, 1), (1, 1), (2, 0))),
            ]
        )
        self.proj = weight_norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(
        self,
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        b, c, t = xs.shape
        if t % self.period != 0:
            padding = self.period - (t % self.period)
            xs = F.pad(xs, (0, padding), "reflect")
            t = t + padding
        xs = xs.contiguous().view(b, c, t // self.period, self.period)

        fmap = []
        for layer in self.layers:
            xs = F.leaky_relu(layer(xs), LRELU_SLOPE)
            fmap.append(xs)

        xs = self.proj(xs)
        fmap.append(xs)

        xs = torch.flatten(xs, 1, -1)

        return xs, fmap


class MPWD(nn.Module):
    def __init__(self, periods: List[int]):
        super(MPWD, self).__init__()
        self.discriminators = nn.ModuleList([PWD(prd) for prd in periods])

    def forward(
        self, xs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:

        outputs = [disc(xs) for disc in self.discriminators]
        disc_outs, fmap_outs = map(list, zip(*outputs))

        return disc_outs, fmap_outs


class RSD(nn.Module):
    def __init__(self, n_fft: int, win_length: int, hop_length: int):
        super(RSD, self).__init__()
        self.spectrogram = T.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=1
        )
        self.layers = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (3, 9), (1, 1), (1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))),
            ]
        )
        self.proj = weight_norm(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(
        self,
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        xs = self.spectrogram(xs)
        for layer in self.layers:
            xs = F.leaky_relu(layer(xs), LRELU_SLOPE)

        xs = self.proj(xs)
        xs = torch.flatten(xs, 1, -1)

        return xs


class MRSD(nn.Module):
    def __init__(self, resolutions: List[List[int]]):
        super(MRSD, self).__init__()
        self.discriminators = nn.ModuleList([RSD(*rst) for rst in resolutions])

    def forward(
        self, xs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outputs = [disc(xs) for disc in self.discriminators]
        return outputs


class PQMF(nn.Module):
    def __init__(self, N: int, taps: int, cutoff: float, beta: float):
        super(PQMF, self).__init__()

        self.N = N
        self.taps = taps
        self.cutoff = cutoff
        self.beta = beta

        QMF = self._firwin(taps + 1, cutoff, beta)
        H = torch.zeros((N, len(QMF)))
        G = torch.zeros((N, len(QMF)))
        for k in range(N):
            constant_factor = (
                (2 * k + 1)
                * (torch.pi / (2 * N))
                * (torch.arange(taps + 1) - ((taps - 1) / 2))
            )
            phase = (-1) ** k * torch.pi / 4

            H[k] = 2 * QMF * torch.cos(constant_factor + phase)
            G[k] = 2 * QMF * torch.cos(constant_factor - phase)

        H = H[:, None, :].float()
        G = G[None, :, :].float()

        self.register_buffer("H", H)
        self.register_buffer("G", G)

        updown_filter = torch.zeros((N, N, N), dtype=torch.float)
        for k in range(N):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        outs = F.conv1d(xs, self.H, padding=self.taps // 2, stride=self.N)
        return outs

    def _firwin(
        self,
        numtaps: int,
        cutoff: float,
        beta: float,
    ) -> torch.Tensor:

        pi = torch.pi
        alpha = 0.5 * (numtaps - 1)

        coeff = torch.arange(0, numtaps) - alpha
        coeff = torch.sin(pi * cutoff * coeff) / (pi * cutoff * coeff)
        coeff = coeff.nan_to_num(1.0)

        window = torch.kaiser_window(numtaps, periodic=False, beta=beta)

        filters = cutoff * coeff * window
        filters /= filters.sum()

        return filters


class MDC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilations: List[int],
    ):
        super(MDC, self).__init__()

        self.layers = nn.ModuleList()
        for dilation in dilations:
            self.layers.append(
                weight_norm(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=(kernel_size * dilation - dilation) // 2,
                        dilation=dilation,
                    )
                )
            )

        self.proj = weight_norm(
            nn.Conv1d(out_channels, out_channels, 3, stride=stride, padding=1)
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:

        _xs = 0.0
        for layer in self.layers:
            _xs += layer(xs)

        xs = _xs / len(self.layers)

        xs = self.proj(xs)
        xs = F.leaky_relu(xs, LRELU_SLOPE)

        return xs


class SBD(nn.Module):
    def __init__(
        self,
        init_channel: int,
        channels: List[int],
        kernel: int,
        strides: List[int],
        dilations: List[List[int]],
    ):
        super(SBD, self).__init__()

        self.layers = nn.ModuleList()
        for c, s, d in zip(channels, strides, dilations):
            k = kernel
            self.layers.append(MDC(init_channel, c, k, s, d))
            init_channel = c

        self.proj = weight_norm(nn.Conv1d(init_channel, 1, 3, 1, 1))

    def forward(
        self,
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        fmap = []
        for layer in self.layers:
            xs = layer(xs)
            fmap.append(xs)

        xs = self.proj(xs)
        fmap.append(xs)

        xs = torch.flatten(xs, 1, -1)

        return xs, fmap


class MSBD(torch.nn.Module):
    def __init__(
        self,
        time_kernels: List[int],
        freq_kernel: int,
        time_channels: List[int],
        freq_channels: List[int],
        time_strides: List[List[int]],
        freq_stride: List[int],
        time_dilations: List[List[List[int]]],
        freq_dilations: List[List[int]],
        time_subband: List[int],
    ):

        super(MSBD, self).__init__()

        self.N = 16
        self.M = 64

        self.time_subband_1 = time_subband[0]
        self.time_subband_2 = time_subband[1]
        self.time_subband_3 = time_subband[2]

        self.fsbd = SBD(
            init_channel=SEGMENT_SIZE // self.M,
            channels=freq_channels,
            kernel=freq_kernel,
            strides=freq_stride,
            dilations=freq_dilations,
        )

        self.tsbd1 = SBD(
            init_channel=time_subband[0],
            channels=time_channels,
            kernel=time_kernels[0],
            strides=time_strides[0],
            dilations=time_dilations[0],
        )

        self.tsbd2 = SBD(
            init_channel=time_subband[1],
            channels=time_channels,
            kernel=time_kernels[1],
            strides=time_strides[1],
            dilations=time_dilations[1],
        )

        self.tsbd3 = SBD(
            init_channel=time_subband[2],
            channels=time_channels,
            kernel=time_kernels[2],
            strides=time_strides[2],
            dilations=time_dilations[2],
        )

        self.pqmf_n = PQMF(N=self.N, taps=256, cutoff=0.03, beta=10.0)
        self.pqmf_m = PQMF(N=self.M, taps=256, cutoff=0.1, beta=9.0)

    def forward(
        self, xs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:

        disc_outs, fmap_outs = [], []

        xm = self.pqmf_m(xs)
        xm = xm.transpose(2, 1)

        q4, fmap = self.fsbd(xm)
        disc_outs.append(q4)
        fmap_outs.append(fmap)

        xn = self.pqmf_n(xs)

        q3, fmap = self.tsbd3(xn[:, : self.time_subband_3, :])
        disc_outs.append(q3)
        fmap_outs.append(fmap)

        q2, fmap = self.tsbd2(xn[:, : self.time_subband_2, :])
        disc_outs.append(q2)
        fmap_outs.append(fmap)

        q1, fmap = self.tsbd1(xn[:, : self.time_subband_1, :])
        disc_outs.append(q1)
        fmap_outs.append(fmap)

        return disc_outs, fmap_outs


class MBD(nn.Module):
    def __init__(
        self,
        channels: List[int],
        kernels: List[int],
        strides: List[int],
        groups: List[int],
    ):
        super(MBD, self).__init__()

        init_channel = 1
        self.layers = nn.ModuleList()
        for c, k, s, g in zip(channels, kernels, strides, groups):
            self.layers.append(
                weight_norm(
                    nn.Conv1d(
                        in_channels=init_channel,
                        out_channels=c,
                        kernel_size=k,
                        stride=s,
                        padding=(k - 1) // 2,
                        groups=g,
                    )
                )
            )
            init_channel = c

        self.proj = weight_norm(nn.Conv1d(channels[-1], 1, 3, 1, 1))

    def forward(
        self,
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        fmap = []
        for layer in self.layers:
            xs = F.leaky_relu(layer(xs), LRELU_SLOPE)
            fmap.append(xs)

        xs = self.proj(xs)
        fmap.append(xs)

        xs = torch.flatten(xs, 1, -1)

        return xs, fmap


class MMBD(nn.Module):
    def __init__(
        self,
        channels: List[int],
        kernels: List[List[int]],
        strides: List[int],
        groups: List[int],
    ):
        super(MMBD, self).__init__()

        self.combd_1 = MBD(channels, kernels[0], strides, groups)
        self.combd_2 = MBD(channels, kernels[1], strides, groups)
        self.combd_3 = MBD(channels, kernels[2], strides, groups)

        self.pqmf_2 = PQMF(N=2, taps=256, cutoff=0.25, beta=10.0)
        self.pqmf_4 = PQMF(N=4, taps=192, cutoff=0.13, beta=10.0)

    def forward(
        self, xs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:

        disc_outs, fmap_outs = [], []

        p3, fmap = self.combd_3(xs)
        disc_outs.append(p3)
        fmap_outs.append(fmap)

        x2_prime = self.pqmf_2(xs)[:, :1, :]
        p2, fmap = self.combd_2(x2_prime)
        disc_outs.append(p2)
        fmap_outs.append(fmap)

        x1_prime = self.pqmf_4(xs)[:, :1, :]
        p1, fmap = self.combd_1(x1_prime)
        disc_outs.append(p1)
        fmap_outs.append(fmap)

        return disc_outs, fmap_outs


class TFD(nn.Module):
    def __init__(self, n_fft: int, win_length: int, hop_length: int):
        super().__init__()

        self.transform = T.Spectrogram(n_fft, win_length, hop_length, power=1)
        self.activation = nn.LeakyReLU(LRELU_SLOPE)

        self.coarse_grained_conv = weight_norm(nn.Conv2d(256, 1, 3, 1, 1))
        self.fine_grained_conv = weight_norm(nn.Conv2d(4, 1, 3, 1, 1))

        self.encoder_1 = weight_norm(nn.Conv2d(1, 8, 3, 1, 1))
        self.encoder_2 = weight_norm(nn.Conv2d(8, 16, 3, 2, 1))
        self.encoder_3 = weight_norm(nn.Conv2d(16, 32, 3, 1, 1))
        self.encoder_4 = weight_norm(nn.Conv2d(32, 64, 3, 2, 1))
        self.encoder_5 = weight_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.encoder_6 = weight_norm(nn.Conv2d(128, 256, 3, 2, 1))

        self.decoder_6 = weight_norm(nn.ConvTranspose2d(256, 128, 4, 2, 1))
        self.decoder_5 = weight_norm(nn.Conv2d(256, 64, 3, 1, 1))
        self.decoder_4 = weight_norm(nn.ConvTranspose2d(128, 32, 4, 2, 1))
        self.decoder_3 = weight_norm(nn.Conv2d(64, 16, 3, 1, 1))
        self.decoder_2 = weight_norm(nn.ConvTranspose2d(32, 8, 4, 2, 1))
        self.decoder_1 = weight_norm(nn.Conv2d(16, 4, 3, 1, 1))

    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self.transform(xs)
        xs = xs[:, :, :-1, :-1]

        enc_out_1 = self.encoder_1(xs)
        enc_out_2 = self.encoder_2(self.activation(enc_out_1))
        enc_out_3 = self.encoder_3(self.activation(enc_out_2))
        enc_out_4 = self.encoder_4(self.activation(enc_out_3))
        enc_out_5 = self.encoder_5(self.activation(enc_out_4))
        enc_out_6 = self.encoder_6(self.activation(enc_out_5))

        output_1 = self.coarse_grained_conv(self.activation(enc_out_6))

        dec_out_6 = self.decoder_6(self.activation(enc_out_6))
        dec_out_5 = self.decoder_5(
            self.activation(torch.cat((enc_out_5, dec_out_6), dim=1))
        )
        dec_out_4 = self.decoder_4(
            self.activation(torch.cat((enc_out_4, dec_out_5), dim=1))
        )
        dec_out_3 = self.decoder_3(
            self.activation(torch.cat((enc_out_3, dec_out_4), dim=1))
        )
        dec_out_2 = self.decoder_2(
            self.activation(torch.cat((enc_out_2, dec_out_3), dim=1))
        )
        dec_out_1 = self.decoder_1(
            self.activation(torch.cat((enc_out_1, dec_out_2), dim=1))
        )

        output_2 = self.fine_grained_conv(self.activation(dec_out_1))

        return output_1, output_2


class MTFD(nn.Module):
    def __init__(self, resolutions: List[List[int]]):
        super(MTFD, self).__init__()
        self.discriminators = nn.ModuleList([TFD(*res) for res in resolutions])

    def forward(self, xs: torch.Tensor) -> List[Tuple[torch.Tensor, ...]]:
        outputs = (disc(xs) for disc in self.discriminators)
        outputs = [out for output in outputs for out in output]
        return outputs


class Downsampling(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()

        self.main_conv = nn.Conv1d(
            input_channels, output_channels, kernel_size=1, stride=1
        )
        self.main_pool = nn.AvgPool1d(kernel_size=7, stride=4, padding=3)

        self.skip_pool = nn.AvgPool1d(kernel_size=7, stride=4, padding=3)
        self.skip_conv = nn.Conv1d(
            input_channels, input_channels, kernel_size=7, stride=4, padding=3
        )
        self.skip_proj = nn.Conv1d(
            input_channels, output_channels, kernel_size=3, padding=1
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        # Pre-processing
        main_xs = xs.clone()
        skip_xs = xs.clone()

        # Main branch
        main_xs = self.main_conv(main_xs)
        main_xs = self.main_pool(main_xs)

        # Skip branch
        skip_xs = F.leaky_relu(skip_xs, LRELU_SLOPE)
        skip_xs = self.skip_conv(skip_xs) + self.skip_pool(skip_xs)
        skip_xs = F.leaky_relu(skip_xs, LRELU_SLOPE)
        skip_xs = self.skip_proj(skip_xs)

        # Post-processing
        xs = main_xs + 0.5 * skip_xs
        xs = F.normalize(xs, p=2.0, dim=1)

        return xs


class Upsampling(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()

        self.main_conv = nn.Conv1d(
            input_channels, output_channels, kernel_size=1, stride=1
        )
        self.main_upsm = nn.Upsample(scale_factor=4)

        self.skip_upsm = nn.Upsample(scale_factor=4)
        self.skip_conv = nn.ConvTranspose1d(
            input_channels, input_channels, kernel_size=8, stride=4, padding=2
        )
        self.skip_proj = nn.Conv1d(
            input_channels, output_channels, kernel_size=3, padding=1
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        # Pre-processing
        main_xs = xs.clone()
        skip_xs = xs.clone()

        # Main branch
        main_xs = self.main_conv(main_xs)
        main_xs = self.main_upsm(main_xs)

        # Skip branch
        skip_xs = F.leaky_relu(skip_xs, LRELU_SLOPE)
        skip_xs = self.skip_conv(skip_xs) + self.skip_upsm(skip_xs)
        skip_xs = F.leaky_relu(skip_xs, LRELU_SLOPE)
        skip_xs = self.skip_proj(skip_xs)

        # Post-processing
        xs = main_xs + 0.5 * skip_xs
        xs = F.normalize(xs, p=2.0, dim=1)

        return xs


class TimeDisc(nn.Module):
    def __init__(self):
        super().__init__()

        self.inp_conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.out_conv = nn.Conv1d(32, 1, kernel_size=3, padding=1)

        self.down1 = Downsampling(32, 64)
        self.down2 = Downsampling(64, 128)
        self.down3 = Downsampling(128, 256)
        self.down4 = Downsampling(256, 512)

        self.up4 = Upsampling(512, 256)
        self.up3 = Upsampling(512, 128)
        self.up2 = Upsampling(256, 64)
        self.up1 = Upsampling(128, 32)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        # Pre-processing
        xs = self.inp_conv(xs)

        # Downsampling
        down1 = self.down1(xs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        # Upsampling
        up4 = self.up4(down4)
        up3 = self.up3(torch.cat((up4, down3), dim=1))
        up2 = self.up2(torch.cat((up3, down2), dim=1))
        up1 = self.up1(torch.cat((up2, down1), dim=1))

        # Post-processing
        xs = self.out_conv(up1)

        return xs


class FreqDisc(nn.Module):
    def __init__(self):
        super().__init__()

        self.transform = T.Spectrogram(1024, 1024, 256, power=1)
        self.activation = nn.LeakyReLU(LRELU_SLOPE)

        self.inp_conv = nn.Conv2d(1, 4, 3, 1, 1)
        self.out_conv = nn.Conv2d(4, 1, 3, 1, 1)

        self.encoder_1 = weight_norm(nn.Conv2d(4, 8, 3, 1, 1))
        self.encoder_2 = weight_norm(nn.Conv2d(8, 16, 3, 2, 1))
        self.encoder_3 = weight_norm(nn.Conv2d(16, 32, 3, 1, 1))
        self.encoder_4 = weight_norm(nn.Conv2d(32, 64, 3, 2, 1))
        self.encoder_5 = weight_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.encoder_6 = weight_norm(nn.Conv2d(128, 256, 3, 2, 1))

        self.decoder_6 = weight_norm(nn.ConvTranspose2d(256, 128, 4, 2, 1))
        self.decoder_5 = weight_norm(nn.Conv2d(256, 64, 3, 1, 1))
        self.decoder_4 = weight_norm(nn.ConvTranspose2d(128, 32, 4, 2, 1))
        self.decoder_3 = weight_norm(nn.Conv2d(64, 16, 3, 1, 1))
        self.decoder_2 = weight_norm(nn.ConvTranspose2d(32, 8, 4, 2, 1))
        self.decoder_1 = weight_norm(nn.Conv2d(16, 4, 3, 1, 1))

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = self.transform(xs)
        xs = xs[:, :, :-1, :-1]

        xs = self.inp_conv(xs)

        enc_out_1 = self.encoder_1(self.activation(xs))
        enc_out_2 = self.encoder_2(self.activation(enc_out_1))
        enc_out_3 = self.encoder_3(self.activation(enc_out_2))
        enc_out_4 = self.encoder_4(self.activation(enc_out_3))
        enc_out_5 = self.encoder_5(self.activation(enc_out_4))
        enc_out_6 = self.encoder_6(self.activation(enc_out_5))

        dec_out_6 = self.decoder_6(self.activation(enc_out_6))
        dec_out_5 = self.decoder_5(
            self.activation(torch.cat((enc_out_5, dec_out_6), dim=1))
        )
        dec_out_4 = self.decoder_4(
            self.activation(torch.cat((enc_out_4, dec_out_5), dim=1))
        )
        dec_out_3 = self.decoder_3(
            self.activation(torch.cat((enc_out_3, dec_out_4), dim=1))
        )
        dec_out_2 = self.decoder_2(
            self.activation(torch.cat((enc_out_2, dec_out_3), dim=1))
        )
        dec_out_1 = self.decoder_1(
            self.activation(torch.cat((enc_out_1, dec_out_2), dim=1))
        )

        xs = self.out_conv(dec_out_1)

        return xs


class TimeFreqDiscriminator(nn.Module):
    def __init__(self, time_disc: nn.Module, freq_disc: nn.Module):
        super().__init__()
        self.time_disc = time_disc
        self.freq_disc = freq_disc

    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        time_out = self.time_disc(xs)
        freq_out = self.freq_disc(xs)
        return time_out, freq_out
