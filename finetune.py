"""
finetune.py  -  QLoRA Fine-Tuning for Gemma 3 4B
--------------------------------------------------
Takes the JSONL training data produced by ``generate_training_data.py``
and fine-tunes Gemma 3 4B-IT using QLoRA (4-bit quantisation + low-rank
adapters) so only ~1-2 % of parameters are trained.

Requires HuggingFace authentication (accept license at
https://huggingface.co/google/gemma-3-4b-it then run
``huggingface-cli login``).

Uses HuggingFace PEFT + BitsAndBytes directly (no Unsloth dependency).

Prerequisites
~~~~~~~~~~~~~
- NVIDIA GPU with >= 6 GB VRAM  (RTX 3060 / 4060 or better)
- CUDA toolkit installed
- ``pip install transformers peft bitsandbytes datasets trl accelerate``

Usage
~~~~~
    python finetune.py                              # defaults
    python finetune.py --data training_data.jsonl   # custom data path
    python finetune.py --epochs 5 --lr 1e-4         # override hyper-params
    python finetune.py --export-only                # skip training, just convert existing checkpoint

After training the script saves a merged model + LoRA adapter,
then converts to GGUF (Q4_K_M) for Ollama import.

Re-training
~~~~~~~~~~~
You can re-run this script at any time with a larger / updated dataset.
Each run starts from the base Gemma 3 4B weights (not the previous
fine-tune) to avoid compounding LoRA errors.  Use ``--resume`` to
continue from the last checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =====================================================================
# Prompt formatting  (must match Gemma 3 instruct chat template)
# =====================================================================

def format_chat(prompt: str, completion: str) -> str:
    """
    Wrap a (prompt, completion) pair in the Gemma 3 instruct chat template.

    Gemma 3 uses ``<start_of_turn>user`` / ``<start_of_turn>model`` markers.
    """
    return (
        f"<start_of_turn>user\n"
        f"{prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{completion}<end_of_turn>"
    )


def load_dataset_from_jsonl(path: Path) -> List[Dict]:
    """Load the JSONL file and convert to HuggingFace Dataset-compatible dicts."""
    records: List[Dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = format_chat(obj["prompt"], obj["completion"])
            records.append({"text": text})
    logger.info("Loaded %d training examples from %s", len(records), path)
    return records


# =====================================================================
# Training
# =====================================================================

def run_training(args: argparse.Namespace) -> Path:
    """
    Load base model in 4-bit, attach LoRA adapters, train, merge, and
    export to GGUF via llama.cpp.

    Returns the Path to the merged model directory (or GGUF dir if
    conversion succeeds).
    """
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logger.error("No CUDA GPU detected.  QLoRA requires a GPU.")
        sys.exit(1)

    logger.info("GPU: %s  |  VRAM: %.1f GB",
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ── 4-bit quantisation config ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load base model ──
    logger.info("Loading base model: %s  (4-bit QLoRA) ...", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    # Prepare for k-bit training (freeze + cast norms to fp32)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # silence warning during training

    # ── Attach LoRA adapters ──
    logger.info(
        "Attaching LoRA adapters  (r=%d, alpha=%d, dropout=%.2f) ...",
        args.lora_r, args.lora_alpha, args.lora_dropout,
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable params: %s / %s  (%.2f%%)",
                f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # ── Load training data ──
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Training data not found: %s", data_path)
        logger.error("Run  python generate_training_data.py  first.")
        sys.exit(1)

    records = load_dataset_from_jsonl(data_path)
    if len(records) < 10:
        logger.warning(
            "Only %d examples - results may be poor.  "
            "Aim for 200+ examples for meaningful fine-tuning.",
            len(records),
        )
    dataset = Dataset.from_list(records)

    # ── Configure trainer ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy="epoch",
        output_dir=str(output_dir / "checkpoints"),
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        gradient_checkpointing=True,       # saves VRAM
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        max_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    logger.info("Starting training ...  (%d examples, %d epochs)",
                len(records), args.epochs)
    trainer.train(resume_from_checkpoint=args.resume if args.resume else None)
    logger.info("Training complete.")

    # ── Save LoRA adapter ──
    lora_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    logger.info("LoRA adapter saved to %s", lora_dir)

    # ── Merge LoRA into base model and save full weights ──
    merged_dir = output_dir / "merged"
    logger.info("Merging LoRA adapter into base model ...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    logger.info("Merged model saved to %s", merged_dir)

    # ── Convert to GGUF using llama.cpp ──
    gguf_dir = output_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = convert_to_gguf(merged_dir, gguf_dir)

    return gguf_dir


# =====================================================================
# GGUF conversion
# =====================================================================

def convert_to_gguf(merged_dir: Path, gguf_dir: Path) -> Path | None:
    """
    Convert the merged HuggingFace model to GGUF format using
    llama-cpp-python or the llama.cpp convert script.

    Returns the path to the .gguf file, or None if conversion fails.
    """
    gguf_path = gguf_dir / "gemma3-critic-q4_k_m.gguf"

    # --- Method 1: Try llama-cpp-python's built-in converter ---
    try:
        from llama_cpp import Llama
        logger.info("llama-cpp-python found - attempting GGUF conversion ...")
    except ImportError:
        pass

    # --- Method 2: Try using the HF-to-GGUF script from llama.cpp ---
    # Usually users need to clone llama.cpp and use convert_hf_to_gguf.py
    # We'll try to find it or suggest manual steps.

    logger.info(
        "Automatic GGUF conversion requires llama.cpp tools.\n"
        "To convert manually:\n"
        "  1. Clone llama.cpp:  git clone https://github.com/ggml-org/llama.cpp\n"
        "  2. Install:  pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt\n"
        "  3. Convert:  python llama.cpp/convert_hf_to_gguf.py %s --outfile %s --outtype q4_k_m\n"
        "\n"
        "OR use Ollama directly with the merged safetensors model:\n"
        "  Create a Modelfile with:  FROM %s\n"
        "  Then:  ollama create gemma3-critic -f Modelfile",
        merged_dir, gguf_path, merged_dir,
    )

    # --- Method 3: Try calling convert_hf_to_gguf.py if available in PATH ---
    convert_scripts = [
        "convert_hf_to_gguf.py",
        "llama.cpp/convert_hf_to_gguf.py",
        str(Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"),
    ]

    for script in convert_scripts:
        if Path(script).exists():
            logger.info("Found conversion script: %s", script)
            try:
                subprocess.run(
                    [sys.executable, script,
                     str(merged_dir),
                     "--outfile", str(gguf_path),
                     "--outtype", "q4_k_m"],
                    check=True,
                )
                logger.info("GGUF exported to %s", gguf_path)
                return gguf_path
            except subprocess.CalledProcessError as e:
                logger.warning("GGUF conversion failed: %s", e)

    # If no conversion tool found, create a Modelfile that points to safetensors
    logger.info("GGUF conversion tool not found — will create Modelfile for safetensors path.")
    return None


# =====================================================================
# Ollama Modelfile generation
# =====================================================================

def write_modelfile(gguf_dir: Path, output_dir: Path, merged_dir: Path | None = None) -> Path:
    """
    Auto-generate an Ollama Modelfile.

    If a .gguf file exists in gguf_dir, point to it.
    Otherwise, point to the merged safetensors directory.
    """
    modelfile_path = output_dir / "Modelfile"

    # Check for .gguf file
    gguf_files = list(gguf_dir.glob("*.gguf")) if gguf_dir.exists() else []

    if gguf_files:
        from_path = gguf_files[0].resolve()
        logger.info("Found GGUF: %s", from_path)
    elif merged_dir and merged_dir.exists():
        from_path = merged_dir.resolve()
        logger.info("Using merged safetensors dir: %s", from_path)
    else:
        logger.error(
            "No GGUF file or merged model found.  "
            "Run training first or provide --export-only with valid paths."
        )
        sys.exit(1)

    content = f"""\
# Modelfile for fine-tuned Gemma 3 4B RAG Critic
# Generated by finetune.py
#
# Usage:
#   ollama create gemma3-critic -f {modelfile_path}
#   Then set OLLAMA_MODEL_NAME = "gemma3-critic" in config.py

FROM {from_path}

PARAMETER temperature 0.3
PARAMETER num_ctx 2048
PARAMETER stop "<end_of_turn>"

SYSTEM "You are a strict validation judge for a Retrieval-Augmented Generation system. Always respond with valid JSON containing exactly two keys: confidence (int 0-100) and explanation (one sentence)."
"""
    modelfile_path.write_text(content, encoding="utf-8")
    logger.info("Modelfile written to %s", modelfile_path)
    return modelfile_path


# =====================================================================
# CLI
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma 3 4B for RAG critic validation using QLoRA."
    )

    # -- data --
    parser.add_argument(
        "--data", type=str, default="training_data.jsonl",
        help="Path to JSONL training data (default: training_data.jsonl).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="finetuned_model",
        help="Directory for checkpoints, LoRA adapter, and GGUF export.",
    )

    # -- model --
    parser.add_argument(
        "--base-model", type=str, default="google/gemma-3-4b-it",
        help="HuggingFace model ID for the base model.",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=2048,
        help="Maximum sequence length for training.",
    )

    # -- LoRA --
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")

    # -- training --
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume from last checkpoint (for incremental training).",
    )

    # -- export --
    parser.add_argument(
        "--export-only", action="store_true",
        help="Skip training, just generate Modelfile from existing export.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    merged_dir = output_dir / "merged"
    gguf_dir = output_dir / "gguf"

    if args.export_only:
        if not gguf_dir.exists() and not merged_dir.exists():
            logger.error(
                "No export found at %s or %s - run training first.",
                gguf_dir, merged_dir,
            )
            sys.exit(1)
    else:
        gguf_dir_result = run_training(args)
        gguf_dir = gguf_dir_result

    # Generate the Ollama Modelfile
    modelfile_path = write_modelfile(gguf_dir, output_dir, merged_dir)

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"  Merged model:  {merged_dir}")
    print(f"  GGUF dir:      {gguf_dir}")
    print(f"  Modelfile:     {modelfile_path}")
    print(f"  LoRA adapter:  {output_dir / 'lora_adapter'}")
    print()
    print("Next steps:")
    print(f"  1. ollama create gemma3-critic -f {modelfile_path}")
    print(f"  2. Set OLLAMA_MODEL_NAME = \"gemma3-critic\" in config.py")
    print(f"  3. python evaluate.py   (compare against Gemini)")
    print()
    print("To re-train later with more data:")
    print(f"  1. python generate_training_data.py --expand  (grow the dataset)")
    print(f"  2. python finetune.py --data training_data.jsonl")
    print()


if __name__ == "__main__":
    main()
