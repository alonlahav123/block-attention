import gc
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import fire
import torch
from flask import Flask, request
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.block_generation import (
    build_default_generation_config,
    build_rotary_embedding,
    decode_generated_tokens,
    encode_block_inputs,
    generate_block_tokens,
)
from src.runtime import get_cuda_device

app = Flask(__name__)
CORS(app, supports_credentials=True)
VERBOSE_PROMPTS = False


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


@app.route("/generate", methods=["POST"])
def _block_generate():
    try:
        form = request.get_json()
        blocks = form["blocks"]
        if not blocks:
            raise ValueError("Expected at least one block in the request payload")

        prompt_blocks = blocks[:-1]
        instruction = blocks[-1]
        if VERBOSE_PROMPTS:
            print(
                json.dumps(
                    {
                        "blocks": prompt_blocks,
                        "instruction": instruction,
                        "num_local_attention_blocks": form.get(
                            "num_local_attention_blocks",
                            10000,
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )

        encoded_inputs = encode_block_inputs(
            blocks=prompt_blocks,
            instruction=instruction,
            tokenizer=tokenizer,
        )
        generated_token_ids, _ = generate_block_tokens(
            encoded_inputs=encoded_inputs,
            generation_config=generation_config,
            model=model,
            emb=emb,
            tokenizer=tokenizer,
            num_local_attention_blocks=form.get("num_local_attention_blocks", 10000),
        )
        generated = decode_generated_tokens(
            tokenizer=tokenizer,
            token_ids=generated_token_ids,
        )
        print("generated: ", generated)
        return {"ret": 0, "generated": generated, "message": ""}
    except Exception as exc:
        traceback.print_exc()
        return {"ret": 1, "generated": "", "message": str(exc)}, 500
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass
class Args:
    model: str
    port: int
    dtype: str
    device: str = get_cuda_device()
    attn_implementation: str = "auto"
    max_new_tokens: int = 256


def load_model(args: Args):
    attn_implementations = [args.attn_implementation]
    if args.attn_implementation == "auto":
        attn_implementations = ["flash_attention_2", "sdpa"]

    dtype = resolve_dtype(args.dtype)
    last_error = None
    for attn_implementation in attn_implementations:
        try:
            print(
                f"Loading model with attention implementation: {attn_implementation}",
                flush=True,
            )
            return AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model,
                dtype=dtype,
                device_map=args.device,
                attn_implementation=attn_implementation,
            )
        except Exception as exc:
            last_error = exc
            if args.attn_implementation != "auto":
                raise
            print(
                f"Failed to load with {attn_implementation}: {exc}. "
                "Falling back to the next attention implementation.",
                flush=True,
            )

    raise last_error


if __name__ == "__main__":
    args: Args = fire.Fire(component=Args)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model,
        use_fast=False,
    )
    model = load_model(args=args)
    model.eval()
    emb = build_rotary_embedding(model_name_or_path=args.model, device=model.device)
    generation_config = build_default_generation_config(
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )
    app.run(host="0.0.0.0", port=args.port)
