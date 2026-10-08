#!/usr/bin/env python3
"""Smoke-test local Qwen3.8-27B Transformers loading on the GPU."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen3.8-27B"

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Check `nvidia-smi` and your PyTorch CUDA install.")

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM: %.2f GB" % (
    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
))

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).eval()

messages = [
    {
        "role": "user",
        "content": (
            "Return JSON only. What is the product category in: "
            "'We need 100 cotton terry towels, 145 x 72.5 cm, sky blue'?"
        ),
    }
]

try:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        enable_thinking=False,
    )
except TypeError:
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(text, return_tensors="pt").input_ids

inputs = inputs.to(next(model.parameters()).device)

with torch.inference_mode():
    outputs = model.generate(
        inputs,
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

answer = tokenizer.decode(
    outputs[0, inputs.shape[-1]:],
    skip_special_tokens=True,
)

print("\nModel output:")
print(answer)
print(
    "\nGPU allocated: %.2f GB"
    % (torch.cuda.memory_allocated() / (1024 ** 3))
)
