
import torch
from torch import nn
from ezspeech.modules.convolution import ConvSubsampling
from ezspeech.modules.conformer import ConformerLayer, _lengths_to_padding_mask
from typing import List, Optional, Tuple, Union

class Conformer(torch.nn.Module):
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        num_heads: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        vocab_size: int = 98,
    ):
        super().__init__()
        # self.conv_subsample = ConvSubsampling(input_dim=d_input,
        #                             feat_out=d_hidden,conv_channels=d_hidden,subsampling_factor=2)
        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    d_hidden,
                    4 * d_hidden,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.lm_head = torch.nn.Linear(d_hidden, vocab_size)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # x, lengths = self.conv_subsample(x, lengths)

        encoder_padding_mask = _lengths_to_padding_mask(lengths)
        x = x.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        last_hidden=x.transpose(0, 1)
        x = self.lm_head(last_hidden)
        x = x.log_softmax(2)

        return x,last_hidden, lengths


class Conformer_self_condition_phoneme_share(torch.nn.Module):

    def __init__(
        self,
        d_hidden: int,
        num_heads: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        inter_layer: List[int]=[3,6,9],
        phoneme_inter_layer: List[int]=[2,5,7],
        vocab_size: int = 98,
        phoneme_vocab_size: int= 1,

    ):
        super().__init__()
        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    d_hidden,
                    4 * d_hidden,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.inter_layer=inter_layer
        self.lm_head = torch.nn.Linear(d_hidden, vocab_size)
        self.back_to_hidden=torch.nn.Linear(vocab_size, d_hidden)

        self.phoneme_inter_layer=phoneme_inter_layer
        self.phoneme_lm_head = torch.nn.Linear(d_hidden, phoneme_vocab_size)
        self.phoneme_back_to_hidden=torch.nn.Linear(phoneme_vocab_size, d_hidden)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        encoder_padding_mask = _lengths_to_padding_mask(lengths)
        hidden = x.transpose(0, 1)
        inter_layer_softmax_lst=[]
        phoneme_inter_layer_softmax_lst=[]


        for idx,layer in enumerate(self.conformer_layers):
            hidden=layer(hidden,encoder_padding_mask)
            hidden=hidden.transpose(0, 1)
            new_hidden=hidden
            if idx+1 in self.phoneme_inter_layer:
                inter_layer_out=self.phoneme_lm_head(hidden)
                phoneme_inter_prediction= inter_layer_out.log_softmax(2)
                phoneme_inter_layer_in=self.phoneme_back_to_hidden(phoneme_inter_prediction)
                phoneme_inter_layer_softmax_lst.append(phoneme_inter_prediction)
                new_hidden=hidden+phoneme_inter_layer_in
                
            if idx+1 in self.inter_layer:
                
                inter_layer_out=self.lm_head(hidden)
                inter_prediction= inter_layer_out.log_softmax(2)
                inter_layer_in=self.back_to_hidden(inter_prediction)
                inter_layer_softmax_lst.append(inter_prediction)
                new_hidden=new_hidden+inter_layer_in
                    
            
            hidden=new_hidden.transpose(1,0)
        last_hidden=hidden.transpose(0, 1)
        pred = self.lm_head(last_hidden)
        pred = pred.log_softmax(2)

        return pred, lengths,inter_layer_softmax_lst,phoneme_inter_layer_softmax_lst
    
