#!/usr/bin/env python3
import os
import sys
import json
import warnings
import argparse
from typing import Optional

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("TORCH_LOGS", "-all")
warnings.filterwarnings("ignore")

_venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
if os.path.exists(_venv_py) and os.path.abspath(sys.executable) != os.path.abspath(_venv_py):
    os.execv(_venv_py, [_venv_py] + sys.argv)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import load_config
from codon_utils import get_short_name, get_model_name, get_output_dir, resolve_organism_key


def detect_dataset_organism(dataset_path: Optional[str], max_check: int = 500) -> Optional[str]:
    """Inspect dataset to detect if it contains samples exclusively from one organism."""
    if not dataset_path or not os.path.exists(dataset_path):
        return None
    try:
        canonical_keys = set()
        representative = None
        if dataset_path.endswith(".jsonl"):
            with open(dataset_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_check:
                        break
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        org = item.get("organism")
                        if org:
                            canonical_keys.add(resolve_organism_key(org.strip()))
                            if representative is None:
                                representative = org.strip()
                            if len(canonical_keys) > 1:
                                return None
        elif dataset_path.endswith(".csv"):
            import csv
            with open(dataset_path, "r", encoding="utf-8") as f:
                for i, row in enumerate(csv.DictReader(f)):
                    if i >= max_check:
                        break
                    org = row.get("organism")
                    if org:
                        canonical_keys.add(resolve_organism_key(org.strip()))
                        if representative is None:
                            representative = org.strip()
                        if len(canonical_keys) > 1:
                            return None
        if len(canonical_keys) == 1:
            return representative
    except Exception:
        pass
    return None


def resolve_or_collect_dataset(
    dataset_arg: str | None,
    max_samples: int | None = None,
    target_organism: str | None = None,
    config: dict | None = None
) -> str:
    cfg = config or load_config()
    data_cfg = cfg.get("data", {})
    resolved_max_samples = max_samples if max_samples is not None else data_cfg.get("max_samples", 5000)

    if dataset_arg:
        if os.path.exists(dataset_arg):
            if target_organism:
                detected = detect_dataset_organism(dataset_arg)
                if detected and (detected.lower() != target_organism.lower() and resolve_organism_key(detected) != resolve_organism_key(target_organism)):
                    print(f"Warning: Specified dataset '{dataset_arg}' appears to contain data for '{detected}', but target organism is '{target_organism}'.")
            print(f"Using dataset: {dataset_arg}")
            return dataset_arg
        raise FileNotFoundError(f"Specified dataset '{dataset_arg}' not found on disk.")

    candidates = data_cfg.get("candidate_paths", [
        "data/dataset.jsonl",
        "dataset.jsonl",
        "data/dataset.csv",
        "dataset.csv"
    ])

    if target_organism:
        from codon_utils import get_bacteria_slug
        slug = get_bacteria_slug(target_organism) or target_organism.lower().replace(" ", "_")
        for c in [f"data/dataset_{slug}.jsonl", f"dataset_{slug}.jsonl"]:
            if os.path.exists(c):
                c_org = detect_dataset_organism(c)
                if c_org and (c_org.lower() != target_organism.lower() and resolve_organism_key(c_org) != resolve_organism_key(target_organism)):
                    print(f"Notice: Local file '{c}' contains '{c_org}', not '{target_organism}'. Skipping.")
                    continue
                print(f"Found host-matched dataset: {c}")
                return c
        for c in candidates:
            if os.path.exists(c):
                c_org = detect_dataset_organism(c)
                if c_org and (c_org.lower() == target_organism.lower() or resolve_organism_key(c_org) == resolve_organism_key(target_organism)):
                    print(f"Found compatible local dataset: {c}")
                    return c
        print(f"No local dataset found for {target_organism}. Collecting/generating...")
        from data_collection import collect_and_validate
        target_path = f"data/dataset_{slug}.jsonl"
        return collect_and_validate(max_samples=resolved_max_samples, output_path=target_path, organism=target_organism, config=cfg)

    for c in candidates:
        if os.path.exists(c):
            print(f"Found local dataset: {c}")
            return c

    print("No dataset found locally. Running data collection...")
    from data_collection import collect_and_validate
    target_path = data_cfg.get("output_path", "data/dataset.jsonl")
    return collect_and_validate(max_samples=resolved_max_samples, output_path=target_path, config=cfg)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Codon Optimization Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )

    mode_group = parser.add_argument_group("Execution Modes")
    mode_group.add_argument("--tune", action="store_true", help="Fine-tune model on codon dataset")
    mode_group.add_argument("--test", action="store_true", help="Run inference and benchmark against traditional tools")
    mode_group.add_argument("--collect", action="store_true", help="Standalone data collection and validation")
    mode_group.add_argument(
        "--merge",
        nargs="?",
        const="__DEFAULT__",
        default=None,
        metavar="MODEL",
        help="Merge fine-tuned LoRA adapter into standalone base model (default model if no argument given)"
    )

    opt_group = parser.add_argument_group("Configuration")
    opt_group.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config (default: config.yaml)")
    opt_group.add_argument("--model", type=str, default=None, help="Base model identifier")
    opt_group.add_argument("--organism", type=str, default=None, help="Target host organism (e.g. 'Escherichia coli')")
    opt_group.add_argument("--dataset", type=str, default=None, help="Path to dataset (.jsonl or .csv)")
    opt_group.add_argument("--protein", type=str, default=None, help="Target protein sequence for inference")
    opt_group.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    opt_group.add_argument("--batch-size", type=int, default=None, help="Training batch size per device")
    opt_group.add_argument("--learning-rate", "--lr", type=float, default=None, help="Training learning rate")
    opt_group.add_argument("--max-steps", type=int, default=None, help="Maximum training steps (-1 for full epoch)")
    opt_group.add_argument("--max-samples", type=int, default=None, help="Maximum samples to collect")
    opt_group.add_argument("--output", type=str, default=None, help="Override output directory")

    args = parser.parse_args()

    if not (args.tune or args.test or args.collect or args.merge is not None):
        parser.print_help()
        print("\nPlease specify one of: --tune, --test, --collect, or --merge.")
        sys.exit(1)

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    infer_cfg = cfg.get("inference", {})

    model_name = args.model or model_cfg.get("name", "unsloth/Qwen2.5-Coder-1.5B-Instruct")
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 1)
    max_steps = args.max_steps if args.max_steps is not None else train_cfg.get("max_steps", -1)
    max_samples = args.max_samples if args.max_samples is not None else data_cfg.get("max_samples", 5000)

    if args.collect:
        from data_collection import collect_and_validate
        from codon_utils import get_bacteria_slug
        target_organism = args.organism or model_cfg.get("organism")
        slug = (get_bacteria_slug(target_organism) or target_organism.lower().replace(" ", "_")) if target_organism else None
        output_file = args.output or (f"data/dataset_{slug}.jsonl" if slug else data_cfg.get("output_path", "data/dataset.jsonl"))
        collect_and_validate(max_samples=max_samples, output_path=output_file, organism=target_organism, config=cfg)
        print("Data collection complete.")
        return

    if args.tune:
        target_organism = args.organism or model_cfg.get("organism")
        dataset_path = resolve_or_collect_dataset(args.dataset, max_samples=max_samples, target_organism=target_organism, config=cfg)
        target_organism = target_organism or detect_dataset_organism(dataset_path)
        model_out = args.output or (model_cfg.get("output_dir") if args.model is None and not target_organism else None) or get_output_dir(model_name, target_organism)

        from tune import run_tuning
        run_tuning(
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=model_out,
            epochs=epochs,
            max_steps=max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            organism=target_organism,
            config=cfg
        )
        print(f"\nFine-tuning complete. Checkpoint saved to: {model_out}")
        return

    if args.test:
        target_organism = args.organism or model_cfg.get("organism")
        candidate_dir = get_output_dir(model_name, target_organism)
        if args.model:
            model_dir = args.output or args.model
        else:
            default_out = model_cfg.get("output_dir")
            if default_out and not os.path.exists(default_out):
                default_out = None
            if not os.path.exists(candidate_dir):
                # Search for any existing trained model on disk if candidate_dir doesn't exist
                if os.path.exists("models"):
                    for entry in os.listdir("models"):
                        full_p = os.path.join("models", entry)
                        if os.path.isdir(full_p) and any(os.path.exists(os.path.join(full_p, f)) for f in ("adapter_config.json", "config.json")):
                            candidate_dir = full_p
                            break
            model_dir = args.output or default_out or candidate_dir
        protein_seq = args.protein or infer_cfg.get("benchmark_protein")
        from inference import run_test
        run_test(
            model_dir=model_dir,
            protein_seq=protein_seq,
            organism=target_organism or infer_cfg.get("default_organism"),
            config=cfg
        )
        return

    if args.merge is not None:
        target_organism = args.organism or model_cfg.get("organism")
        if args.merge != "__DEFAULT__":
            chosen_model = args.merge
        elif args.model:
            chosen_model = args.model
        else:
            candidate_dir = get_output_dir(model_name, target_organism)
            default_out = model_cfg.get("output_dir")
            if default_out and not os.path.exists(default_out):
                default_out = None
            if not os.path.exists(candidate_dir) and os.path.exists("models"):
                for entry in os.listdir("models"):
                    full_p = os.path.join("models", entry)
                    if os.path.isdir(full_p) and any(os.path.exists(os.path.join(full_p, f)) for f in ("adapter_config.json", "config.json")):
                        candidate_dir = full_p
                        break
            chosen_model = default_out or candidate_dir

        from tune import run_merge
        run_merge(
            model_dir=chosen_model,
            output_dir=args.output,
            config=cfg
        )
        return


if __name__ == "__main__":
    main()
