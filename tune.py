import os
import sys
import json
import warnings
from typing import Optional, Dict, Any

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("TORCH_LOGS", "-all")
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import load_config
from codon_utils import get_output_dir


def format_prompt(protein: str, organism: str, dna: Optional[str] = None, eos_token: str = "") -> str:
    prompt = (
        "Below is an instruction that describes a biological task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"Optimize the following protein sequence into a high-expression DNA coding sequence for {organism}.\n\n"
        "### Input:\n"
        f"{protein}\n\n"
        "### Response:\n"
    )
    if dna:
        prompt += dna + eos_token
    return prompt


def load_local_dataset(dataset_path: str, eos_token: str = "", default_organism: str = "Escherichia coli"):
    from datasets import Dataset

    entries = []
    if dataset_path.endswith(".jsonl"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    entries.append({
                        "text": format_prompt(
                            item["protein"],
                            item.get("organism") or default_organism,
                            item["dna"],
                            eos_token=eos_token
                        )
                    })
    elif dataset_path.endswith(".csv"):
        import csv
        with open(dataset_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prot = row.get("protein") or row.get("amino_acid_sequence", "")
                dna = row.get("dna") or row.get("coding_sequence", "")
                org = row.get("organism") or default_organism
                if prot and dna:
                    entries.append({
                        "text": format_prompt(prot, org, dna, eos_token=eos_token)
                    })
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path} (expected .jsonl or .csv)")

    if not entries:
        raise ValueError(f"No valid records found in {dataset_path}")

    return Dataset.from_list(entries)


def run_tuning(
    model_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    learning_rate: Optional[float] = None,
    max_seq_length: Optional[int] = None,
    organism: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
):
    cfg = config or load_config()
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    lora_cfg = train_cfg.get("lora", {})

    target_organism = organism or model_cfg.get("organism")
    model_name = model_name or model_cfg.get("name", "unsloth/Qwen2.5-Coder-1.5B-Instruct")
    dataset_path = dataset_path or cfg.get("data", {}).get("output_path", "data/dataset.jsonl")
    output_dir = output_dir or (model_cfg.get("output_dir") if not target_organism else None) or get_output_dir(model_name, target_organism)
    epochs = epochs if epochs is not None else train_cfg.get("epochs", 1)
    max_steps = max_steps if max_steps is not None else train_cfg.get("max_steps", -1)
    batch_size = batch_size if batch_size is not None else train_cfg.get("batch_size", 1)
    gradient_accumulation_steps = gradient_accumulation_steps if gradient_accumulation_steps is not None else train_cfg.get("gradient_accumulation_steps", 4)
    learning_rate = learning_rate if learning_rate is not None else train_cfg.get("learning_rate", 2e-4)
    max_seq_length = max_seq_length if max_seq_length is not None else model_cfg.get("max_seq_length", 2048)
    load_in_4bit = model_cfg.get("load_in_4bit", True)

    try:
        from unsloth import FastLanguageModel
        import torch
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError as e:
        print(f"Error: Missing training dependencies: {e}")
        print("Please make sure unsloth, torch, trl, and transformers are installed.")
        sys.exit(1)

    print(f"Fine-tuning {model_name}")
    print(f"Dataset: {dataset_path} | Output: {output_dir}")
    print(f"Epochs: {epochs}, max_steps: {max_steps}, batch_size: {batch_size}, grad_accum: {gradient_accumulation_steps}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading base model {model_name} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    print("Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.get("r", 16),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0),
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=train_cfg.get("seed", 42),
    )

    eos_token = tokenizer.eos_token or "<|im_end|>"
    print(f"Loading dataset from {dataset_path} (eos_token: {repr(eos_token)})...")
    dataset = load_local_dataset(dataset_path, eos_token=eos_token, default_organism=target_organism or "Escherichia coli")
    print(f"Loaded {len(dataset)} examples.")

    is_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=train_cfg.get("dataset_num_proc", 2),
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=train_cfg.get("warmup_steps", 5),
            max_steps=max_steps if max_steps > 0 else -1,
            num_train_epochs=epochs if max_steps <= 0 else 1,
            learning_rate=learning_rate,
            fp16=not is_bf16,
            bf16=is_bf16,
            logging_steps=train_cfg.get("logging_steps", 1),
            optim=train_cfg.get("optim", "adamw_8bit"),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
            seed=train_cfg.get("seed", 42),
            output_dir=output_dir,
            save_strategy=train_cfg.get("save_strategy", "no"),
        ),
    )

    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="### Instruction:\n",
            response_part="### Response:\n",
        )
        print("Successfully applied train_on_responses_only to mask instruction prompt tokens.")
    except Exception as e:
        print(f"Notice: train_on_responses_only could not be applied ({e}); continuing with standard SFT.")

    print("Starting training...")
    trainer.train()

    print(f"Saving fine-tuned adapters to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune an LLM for codon optimization with Unsloth")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--organism", type=str, default=None, help="Target host organism")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_tuning(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        max_steps=args.max_steps,
        organism=args.organism,
        config=cfg
    )
