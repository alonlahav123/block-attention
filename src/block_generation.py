from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Sequence

import torch
from transformers import AutoConfig, GenerationConfig, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaRotaryEmbedding,
)


@dataclass(frozen=True)
class EncodedBlockInputs:
    block_token_ids: list[list[int]]
    instruction_token_ids: list[int]


def encode_block_inputs(
    *,
    blocks: Sequence[str],
    instruction: str,
    tokenizer: PreTrainedTokenizer,
) -> EncodedBlockInputs:
    return EncodedBlockInputs(
        block_token_ids=[
            tokenizer.encode(block, add_special_tokens=False)
            for block in blocks
        ],
        instruction_token_ids=tokenizer.encode(instruction, add_special_tokens=False),
    )


def count_block_prompt_tokens(encoded_inputs: EncodedBlockInputs) -> int:
    return sum(len(token_ids) for token_ids in encoded_inputs.block_token_ids) + len(
        encoded_inputs.instruction_token_ids
    )


def build_rotary_embedding(
    *,
    model_name_or_path: str,
    device: torch.device | str,
) -> LlamaRotaryEmbedding:
    config: LlamaConfig = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path
    )
    emb = LlamaRotaryEmbedding(config=config).to(device=device, dtype=torch.float32)
    emb.eval()
    return emb


def build_default_generation_config(
    *,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int,
) -> GenerationConfig:
    return GenerationConfig(
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.0,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        stop_strings=[
            "<|im_end|>",
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|endoftext|>",
            "</s>",
            "Question:",
        ],
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat(tensors=(-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed.to(dtype=torch.bfloat16)


def _get_first_layer_keys(pkv: DynamicCache) -> torch.Tensor:
    for layer in pkv.layers:
        if layer.keys is not None:
            return layer.keys
    raise ValueError("DynamicCache has no key tensors.")


def apply_pkv_rotary_position_embeddings(
    *,
    pkv: DynamicCache,
    emb: LlamaRotaryEmbedding,
) -> DynamicCache:
    first_layer_keys = _get_first_layer_keys(pkv=pkv)
    device = first_layer_keys.device
    emb.to(device=device)
    position_ids = torch.arange(
        start=0,
        end=first_layer_keys.size(-2),
        dtype=torch.int64,
        device=device,
    )
    position_ids = position_ids.unsqueeze(dim=0).repeat(
        repeats=[first_layer_keys.size(0), 1]
    )
    cos, sin = emb(x=first_layer_keys.to(dtype=torch.float32), position_ids=position_ids)
    for layer in pkv.layers:
        if layer.keys is None:
            continue
        layer.keys = apply_rotary_pos_emb(
            k=layer.keys.to(dtype=torch.float32),
            cos=cos,
            sin=sin,
            position_ids=position_ids,
        )
    return pkv


def apply_pkv_rerotary_position_embeddings(
    *,
    pkv: DynamicCache,
    emb: LlamaRotaryEmbedding,
) -> DynamicCache:
    first_layer_keys = _get_first_layer_keys(pkv=pkv)
    device = first_layer_keys.device
    emb.to(device=device)
    position_ids = torch.arange(
        start=0,
        end=first_layer_keys.size(-2),
        dtype=torch.int64,
        device=device,
    )
    position_ids = position_ids.unsqueeze(dim=0).repeat(
        repeats=[first_layer_keys.size(0), 1]
    )
    cos, sin = emb(x=first_layer_keys.to(dtype=torch.float32), position_ids=position_ids)
    for layer in pkv.layers:
        if layer.keys is None:
            continue
        layer.keys = apply_rotary_pos_emb(
            k=layer.keys.to(dtype=torch.float32),
            cos=cos,
            sin=-sin,
            position_ids=position_ids,
        )
    return pkv


def merge_and_rotary_past_key_values(
    *,
    pkvs: list[DynamicCache],
    emb: LlamaRotaryEmbedding,
) -> DynamicCache:
    cache = pkvs[0]
    for layer_index, layer in enumerate(cache.layers):
        if layer.keys is None or layer.values is None:
            continue
        layer.keys = torch.cat(
            tensors=[layer.keys]
            + [pkvs[cache_index].layers[layer_index].keys for cache_index in range(1, len(pkvs))],
            dim=-2,
        )
        layer.values = torch.cat(
            tensors=[layer.values]
            + [pkvs[cache_index].layers[layer_index].values for cache_index in range(1, len(pkvs))],
            dim=-2,
        )
    return apply_pkv_rotary_position_embeddings(pkv=cache, emb=emb)


def _split_encoded_inputs(
    *,
    encoded_inputs: EncodedBlockInputs,
    num_local_attention_blocks: int,
) -> tuple[list[list[int]], list[int]]:
    if num_local_attention_blocks < 0:
        raise ValueError("num_local_attention_blocks must be non-negative")

    block_token_ids = [list(token_ids) for token_ids in encoded_inputs.block_token_ids]
    instruction_token_ids = list(encoded_inputs.instruction_token_ids)

    if len(block_token_ids) > num_local_attention_blocks:
        overflow_blocks = block_token_ids[num_local_attention_blocks:]
        instruction_token_ids = list(chain.from_iterable(overflow_blocks)) + instruction_token_ids
        block_token_ids = block_token_ids[:num_local_attention_blocks]

    if num_local_attention_blocks == 0:
        instruction_token_ids = list(chain.from_iterable(block_token_ids)) + instruction_token_ids
        block_token_ids = []

    return block_token_ids, instruction_token_ids


@torch.no_grad()
def build_block_past_key_values(
    *,
    encoded_inputs: EncodedBlockInputs,
    model: LlamaForCausalLM,
    emb: LlamaRotaryEmbedding,
    num_local_attention_blocks: int,
) -> tuple[list[DynamicCache] | None, torch.Tensor]:
    block_token_ids, instruction_token_ids = _split_encoded_inputs(
        encoded_inputs=encoded_inputs,
        num_local_attention_blocks=num_local_attention_blocks,
    )

    caches: list[DynamicCache] = []
    input_ids: torch.Tensor | None = None
    for token_ids in block_token_ids:
        block_input_ids = torch.tensor(
            data=[token_ids],
            dtype=torch.int64,
            device=model.device,
        )
        if input_ids is None:
            input_ids = block_input_ids
        else:
            input_ids = torch.cat(tensors=[input_ids, block_input_ids], dim=-1)

        output: CausalLMOutputWithPast = model(
            input_ids=block_input_ids,
            use_cache=True,
            past_key_values=DynamicCache(config=model.config),
            return_dict=True,
        )
        caches.append(
            apply_pkv_rerotary_position_embeddings(
                pkv=output.past_key_values,
                emb=emb,
            )
        )

    response_input_ids = torch.tensor(
        data=[instruction_token_ids],
        dtype=torch.int64,
        device=model.device,
    )
    if input_ids is None:
        return None, response_input_ids

    input_ids = torch.cat(tensors=[input_ids, response_input_ids], dim=-1)
    return caches, input_ids


@torch.no_grad()
def generate_block_tokens(
    *,
    encoded_inputs: EncodedBlockInputs,
    generation_config: GenerationConfig,
    model: LlamaForCausalLM,
    emb: LlamaRotaryEmbedding,
    tokenizer: PreTrainedTokenizer,
    num_local_attention_blocks: int,
) -> tuple[torch.Tensor, int]:
    past_key_values, input_ids = build_block_past_key_values(
        encoded_inputs=encoded_inputs,
        model=model,
        emb=emb,
        num_local_attention_blocks=num_local_attention_blocks,
    )
    return generate_from_precomputed_block_state(
        past_key_values=past_key_values,
        input_ids=input_ids,
        generation_config=generation_config,
        model=model,
        emb=emb,
        tokenizer=tokenizer,
    )


@torch.no_grad()
def generate_from_precomputed_block_state(
    *,
    past_key_values: list[DynamicCache] | None,
    input_ids: torch.Tensor,
    generation_config: GenerationConfig,
    model: LlamaForCausalLM,
    emb: LlamaRotaryEmbedding,
    tokenizer: PreTrainedTokenizer,
) -> tuple[torch.Tensor, int]:
    if past_key_values is not None:
        past_key_values = merge_and_rotary_past_key_values(pkvs=past_key_values, emb=emb)

    return generate_from_merged_past(
        past_key_values=past_key_values,
        input_ids=input_ids,
        generation_config=generation_config,
        model=model,
        tokenizer=tokenizer,
    )


@torch.no_grad()
def generate_from_merged_past(
    *,
    past_key_values: DynamicCache | None,
    input_ids: torch.Tensor,
    generation_config: GenerationConfig,
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
) -> tuple[torch.Tensor, int]:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be rank-2 [batch, seq]")

    input_length = input_ids.size(-1)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.int64),
        generation_config=generation_config,
        past_key_values=past_key_values,
        use_cache=True,
        eos_token_id=[tokenizer.eos_token_id],
        tokenizer=tokenizer,
    )
    return outputs[0][input_length:].detach().cpu(), input_length


def decode_generated_tokens(
    *,
    tokenizer: PreTrainedTokenizer,
    token_ids: Sequence[int] | torch.Tensor,
) -> str:
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids=token_ids)


def block_generate(
    *,
    blocks: Sequence[str],
    instruction: str,
    generation_config: GenerationConfig,
    model: LlamaForCausalLM,
    emb: LlamaRotaryEmbedding,
    tokenizer: PreTrainedTokenizer,
    num_local_attention_blocks: int,
) -> str:
    encoded_inputs = encode_block_inputs(
        blocks=blocks,
        instruction=instruction,
        tokenizer=tokenizer,
    )
    generated_token_ids, _ = generate_block_tokens(
        encoded_inputs=encoded_inputs,
        generation_config=generation_config,
        model=model,
        emb=emb,
        tokenizer=tokenizer,
        num_local_attention_blocks=num_local_attention_blocks,
    )
    return decode_generated_tokens(tokenizer=tokenizer, token_ids=generated_token_ids)
