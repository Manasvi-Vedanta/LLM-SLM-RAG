"""
merge_lora_cpu.py  –  Merge LoRA adapter into base model on CPU
---------------------------------------------------------------
Produces a clean FP16 HuggingFace model (no bitsandbytes metadata)
that llama.cpp can convert to GGUF.

Usage:
    python merge_lora_cpu.py                                          # v1 defaults
    python merge_lora_cpu.py --lora finetuned_model_v2/lora_adapter --out finetuned_model_v2/merged_fp16
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model on CPU")
parser.add_argument("--base", type=str, default="google/gemma-3-4b-it", help="Base model name or path")
parser.add_argument("--lora", type=str, default="finetuned_model/lora_adapter", help="LoRA adapter directory")
parser.add_argument("--out", type=str, default="finetuned_model/merged_fp16", help="Output directory for merged FP16 model")
args = parser.parse_args()

BASE_MODEL = args.base
LORA_DIR   = Path(args.lora)
OUTPUT_DIR = Path(args.out)

print(f"[1/4] Loading base model on CPU in float16 (≈8 GB RAM) ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",          # CPU-only → no bitsandbytes
    trust_remote_code=True,
)

print(f"[2/4] Loading LoRA adapter from {LORA_DIR} ...")
model = PeftModel.from_pretrained(model, str(LORA_DIR))

print("[3/4] Merging LoRA weights into base model ...")
model = model.merge_and_unload()

print(f"[4/4] Saving merged model to {OUTPUT_DIR} ...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("Done!  Merged FP16 model saved to:", OUTPUT_DIR)
