# coding=utf-8
""" PyTorch LLaMA model."""
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.cache_utils import Cache
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from xmixers.modules import get_channel_mixer, get_norm_fn, get_token_mixer
from xmixers.utils import (
    XmixersCache,
    XmixersGenerationMixin,
    _init_weights,
    pad_embed_dim,
)
from xmixers.utils.loss_utils import Loss

from .configuration_llama import LLaMAConfig
from xmixers.pisa_config import PisaConfig
from xmixers.modules.normalizations.rms_norm import RMSNorm
import math

logger = logging.get_logger(__name__)


class LLaMALayer(nn.Module):
    def __init__(self, config: LLaMAConfig, layer_idx=0):
        super().__init__()

        self.token_mixer = get_token_mixer(config, layer_idx)
        self.token_norm = get_norm_fn(config.norm_type)(
            config.embed_dim, eps=config.rms_norm_eps, bias=config.bias,
            num_warps=config.rms_norm_num_warps
        )
        self.channel_mixer = get_channel_mixer(config)
        self.channel_norm = get_norm_fn(config.norm_type)(
            config.embed_dim, eps=config.rms_norm_eps, bias=config.bias,
            num_warps=config.rms_norm_num_warps
        )

        self.use_postnorm = config.use_postnorm
        if self.use_postnorm:
            self.forward = self.forward_postnorm
        self.fuse_norm_add = config.fuse_norm_add

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if not self.fuse_norm_add:
            # token mixer
            residual = x
            x, past_key_values = self.token_mixer(
                x=self.token_norm(x),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            x = x + residual

            # channel mixer
            x = self.channel_mixer(self.channel_norm(x)) + x
        else:
            # token mixer
            residual_token = x
            x_attn, past_key_values = self.token_mixer(
                x=self.token_norm(x),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

            # channel mixer
            x_channel, residual_channel = self.channel_norm(
                x_attn, residual=residual_token
            )
            x = self.channel_mixer(x_channel) + residual_channel

        outputs = (x, past_key_values, None)

        return outputs

    def forward_postnorm(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # token mixer
        residual = x
        x, past_key_values = self.token_mixer(
            x=x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        x = self.token_norm(x + residual)

        # channel mixer
        x = self.channel_norm(self.channel_mixer(x) + x)

        outputs = (x, past_key_values, None)

        return outputs


class LLaMAPreTrainedModel(PreTrainedModel):
    config_class = LLaMAConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["LLaMALayer"]

    def __init__(self, config):
        if isinstance(config, PisaConfig):
            config = LLaMAConfig.from_pisa_config(config)
        super().__init__(config)

    def _init_weights(self, module):
        return _init_weights(self, module)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LLaMAModel):
            module.gradient_checkpointing = value


class LLaMAModel(LLaMAPreTrainedModel):
    def __init__(self, config: LLaMAConfig):
        super().__init__(config)
        config = self.config
        # hf origin
        self.padding_idx = config.pad_token_id
        self.embedding_vocab_size = config.padded_vocab_size
        self.config = config
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = config.gradient_checkpointing

        # params
        self.embed_scale = config.embed_dim**0.5 if config.use_embed_scale else 1
        self.embed_tokens = nn.Embedding(
            self.embedding_vocab_size, config.embed_dim, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LLaMALayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = get_norm_fn(config.norm_type)(
            config.embed_dim, eps=config.rms_norm_eps, bias=config.bias,
            num_warps=config.rms_norm_num_warps
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        use_cache = (
            use_cache
            if use_cache is not None
            else (self.config.use_cache if not self.training else False)
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if kwargs.get("cu_seqlens") is not None and (use_cache or past_key_values is not None):
            raise ValueError("Packed model caching is not supported")
        if past_key_values and len(past_key_values) != len(self.layers):
            raise ValueError("Expected one cache entry per decoder layer")
        if use_cache and not isinstance(past_key_values, XmixersCache):
            past_key_values = XmixersCache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = self.embed_scale * inputs_embeds

        checkpoint_layers = (self.gradient_checkpointing and self.training
                             and torch.is_grad_enabled() and not use_cache)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        residual = None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if checkpoint_layers:
                (
                    hidden_states,
                    past_key_values,
                    residual,
                ) = torch.utils.checkpoint.checkpoint(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    residual,
                    use_reentrant=False,
                    **kwargs,
                )
            else:
                hidden_states, past_key_values, residual = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    residual=residual,
                    **kwargs,
                )

        if residual is not None:
            hidden_states = self.final_norm(hidden_states, residual=residual)[0]
        else:
            hidden_states = self.final_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LLaMAForCausalLM(LLaMAPreTrainedModel, XmixersGenerationMixin):
    _supports_cache_class = False
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        config = self.config
        self.model = LLaMAModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.embed_dim, config.padded_vocab_size, bias=False)
        self.loss = Loss()
        # Initialize weights and apply final processing
        self.post_init()
        self.post_init_weights()

    @torch.no_grad()
    def post_init_weights(self):
        config = self.config
        if config.attention_method in {"pisa", "pisa1", "pisa2"}:
            self.initialize_pyramid_weights()
            return
        else:
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    nn.init.trunc_normal_(
                        module.weight,
                        std=config.init_std,
                        a=-3 * config.init_std,
                        b=3 * config.init_std,
                    )
                    if getattr(module, "bias", None) is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, RMSNorm):
                    nn.init.ones_(module.weight)
            for layer in self.model.layers:
                layer.token_mixer.rope._init_weights()
                attention = layer.token_mixer
                if hasattr(attention, "nsa"):
                    attention.nsa.reset_parameters()
                projections = [
                    attention.q_proj,
                    attention.k_proj,
                    attention.v_proj,
                    attention.o_proj,
                ]
                if hasattr(attention, "nsa"):
                    projections.extend((attention.nsa.g_cmp_proj, attention.nsa.g_slc_proj))
                for projection in projections:
                    nn.init.xavier_uniform_(
                        projection.weight, gain=config.attention_init_gain
                    )
                    if projection.bias is not None:
                        nn.init.zeros_(projection.bias)
                layer.channel_mixer.w3.weight.div_(math.sqrt(2 * config.num_hidden_layers))
                if config.zero_init_output:
                    nn.init.zeros_(attention.o_proj.weight)
                    nn.init.zeros_(layer.channel_mixer.w3.weight)
        if config.pad_token_id is not None:
            from torch.distributed.tensor import DTensor

            weight = self.model.embed_tokens.weight
            padding_row = config.pad_token_id
            if isinstance(weight, DTensor):
                from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

                # Initialization runs after FSDP sharding. Only the owning shard
                # has this row; indexing the DTensor itself is not supported.
                _, offset = compute_local_shape_and_global_offset(
                    weight.shape, weight.device_mesh, weight.placements
                )
                padding_row -= offset[0]
                weight = weight.to_local()
            if 0 <= padding_row < weight.shape[0]:
                weight[padding_row].zero_()

    @torch.no_grad()
    def initialize_pyramid_weights(self):
        self.model.embed_tokens.apply(self._init_weights)
        self.model.final_norm._init_weights()
        for layer in self.model.layers:
            attention = layer.token_mixer
            projections = (attention.q_proj, attention.k_proj,
                           attention.v_proj, attention.o_proj)
            # Preserve the random draws preceding the final normal initialization.
            for projection in projections:
                nn.init.xavier_uniform_(
                    projection.weight, gain=self.config.attention_init_gain
                )
            attention.rope._init_weights()
            for projection in projections:
                self._init_weights(projection)
            layer.channel_mixer.apply(self._init_weights)
            layer.token_norm._init_weights()
            layer.channel_norm._init_weights()
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.lm_head, self.model.embed_tokens)
        else:
            self.lm_head.apply(self._init_weights)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LLaMAForCausalLM

        >>> model = LLaMAForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is not None:
                if attention_mask.ndim != 2 or attention_mask.shape[0] != input_ids.shape[0] or attention_mask.shape[1] < input_ids.shape[1]:
                    raise ValueError("attention_mask must cover [batch, sequence]")
                mask = attention_mask[:, -input_ids.shape[1]:].bool()
            if isinstance(past_key_values, dict) or not bool(mask.all()):
                if kwargs.get("cu_seqlens") is not None or inputs_embeds is not None:
                    raise ValueError("Use either evaluation padding or packed document boundaries")
                if past_key_values is not None and not isinstance(past_key_values, dict):
                    raise ValueError("Cannot add padding to an existing batched cache")
                if output_attentions or output_hidden_states:
                    raise ValueError("Padded evaluation does not expose intermediate histories")
                row_cache = use_cache if use_cache is not None else self.config.use_cache and not self.training
                logits = self.lm_head.weight.new_zeros((*input_ids.shape, self.config.padded_vocab_size))
                next_rows, losses, counts = [], [], []
                for row in range(input_ids.shape[0]):
                    active = mask[row]
                    if not bool(active.any()):
                        raise ValueError("Each evaluation sequence must contain an unmasked token")
                    row_labels = None if labels is None else labels[row, active][None]
                    output = self(input_ids[row, active][None], labels=row_labels,
                                  past_key_values=None if past_key_values is None else past_key_values["rows"][row],
                                  use_cache=row_cache, return_logits=True)
                    logits[row, active] = output.logits[0]
                    next_rows.append(output.past_key_values)
                    if labels is not None:
                        count = (row_labels[:, 1:] != -100).sum()
                        losses.append(output.loss * count)
                        counts.append(count)
                cache = {"rows": next_rows} if row_cache else None
                loss = sum(losses) / torch.stack(counts).sum().clamp_min(1) if losses else None
                if num_logits_to_keep:
                    logits = logits[:, -num_logits_to_keep:]
                if not return_dict:
                    values = (logits, cache)
                    return ((loss,) + values) if loss is not None else values
                return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=cache)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        fuse_linear_and_cross_entropy = (
            self.config.ce_type not in ["fla_fce", "naive", "xopes_ce"]
            and self.training
        )

        # since we may use forward_pre_hook, we can't use key word arguments
        logits, loss = self.loss(
            self.lm_head.weight,
            hidden_states,
            labels,
            self.lm_head.bias,
            num_logits_to_keep,
            fuse_linear_and_cross_entropy,
            ce_type=self.config.loss_backend,
            cu_seqlens=kwargs.get("cu_seqlens"),
            mask_document_targets=self.config.mask_document_targets,
            chunk_size=self.config.loss_chunk_size,
            return_logits=kwargs.get("return_logits"),
        )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def num_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())

    @classmethod
    def _supports_default_dynamic_cache(cls):
        # Generation uses the native cache, which also stores the key pyramid.
        return False

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        use_cache=True, **kwargs,
    ):
        # Transformers manages length, stopping and sampling processors. Only
        # the native cache boundary needs this model-specific input slicing.
        return {
            "input_ids": input_ids[:, -1:] if past_key_values and use_cache else input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @torch.no_grad()
    def generate(self, inputs=None, generation_config=None, **kwargs):
        config = generation_config or self.generation_config
        if kwargs.get("do_sample", config.do_sample) or kwargs.get("num_beams", config.num_beams) != 1:
            raise ValueError("The PISA HF adapter supports greedy generation with num_beams=1")
        if kwargs.get("num_return_sequences", config.num_return_sequences) != 1:
            raise ValueError("Greedy generation supports one result per input sequence")
        input_ids = kwargs.pop("input_ids", inputs)
        if input_ids is None:
            raise ValueError("PISA generation requires input_ids")
        original = input_ids
        mask = kwargs.get("attention_mask")
        pad_id = kwargs.get("pad_token_id", config.pad_token_id)
        if pad_id is None:
            pad_id = self.config.pad_token_id
        eos_ids = kwargs.get("eos_token_id", config.eos_token_id)
        eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        if mask is None and pad_id is not None and pad_id not in eos_ids:
            mask = input_ids.ne(pad_id)
        if mask is not None:
            if mask.shape != input_ids.shape or not bool(mask.bool().any(-1).all()):
                raise ValueError("Generation attention_mask must match nonempty input sequences")
            # Decoder generation reads the final column. Normalize right padding
            # to left padding, then restore the caller's prompt prefix on return.
            input_ids = torch.full_like(input_ids, pad_id if pad_id is not None else 0)
            left_mask = torch.zeros_like(mask)
            for row in range(len(input_ids)):
                tokens = original[row, mask[row].bool()]
                input_ids[row, -len(tokens):] = tokens
                left_mask[row, -len(tokens):] = 1
            kwargs["attention_mask"] = left_mask
        processors = LogitsProcessorList(kwargs.pop("logits_processor", None) or [])
        processors.append(TokenizerVocabulary(self.config.vocab_size))
        result = super().generate(input_ids, generation_config=generation_config,
                                  logits_processor=processors, **kwargs)
        sequences = result if isinstance(result, torch.Tensor) else result.sequences
        sequences[:, :original.shape[1]] = original
        return result

class TokenizerVocabulary(LogitsProcessor):
    """Keep padded embedding rows out of generation, as native PISA does."""
    def __init__(self, size):
        self.size = size

    def __call__(self, input_ids, scores):
        scores[:, self.size:] = -torch.inf
        return scores

