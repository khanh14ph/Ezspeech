import math
from typing import Callable, Optional, Tuple, Union
from torch import nn
import torch


class WhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        layer_idx: Optional[int] = None,
        config: Optional[WhisperConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        # TODO: we need a refactor so that the different attention modules can get their specific kwargs
        # ATM, we have mixed things encoder, decoder, and encoder-decoder attn
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # determine input shapes
        bsz, tgt_len = hidden_states.shape[:-1]
        q_input_shape = (bsz, tgt_len, -1, self.head_dim)

        # Scaling is susceptible to floating point arithmetics' inprecisions
        # which can lead to different results (this is dependent from model
        # to model, e.g. whisper is one such case). We therefore keep the
        # original order of scaling to follow the original implementation
        # and enforce no scaling (1.0) in the attention call below.
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(*q_input_shape)
        query_states = query_states.transpose(1, 2).contiguous()

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = (
            key_value_states if key_value_states is not None else hidden_states
        )
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(current_states).view(
                bsz, -1, self.num_heads, self.head_dim
            )
            value_states = self.v_proj(current_states).view(
                bsz, -1, self.num_heads, self.head_dim
            )
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"cache_position": cache_position},
                )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            output_attentions=output_attentions,
            head_mask=layer_head_mask,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class WhisperDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            layer_idx=layer_idx,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            layer_idx=layer_idx,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states, cross_attn_weights, cross_attn_present_key_value = (
                self.encoder_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states

            # add cross-attn to positions 1 of present_key_value tuple
            present_key_value = (present_key_value, cross_attn_present_key_value)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
