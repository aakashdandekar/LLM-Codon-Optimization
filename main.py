#!/usr/bin/env python3
import os
import sys

_venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
if os.path.exists(_venv_py) and os.path.abspath(sys.executable) != os.path.abspath(_venv_py):
    try:
        import torch  # noqa: F401
        import unsloth  # noqa: F401
    except ImportError:
        os.execv(_venv_py, [_venv_py] + sys.argv)

import json
import warnings
import argparse
from typing import Optional

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("TORCH_LOGS", "-all")
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import load_config
from codon_utils import get_short_name, get_model_name, get_output_dir


def detect_dataset_organism(dataset_path: Optional[str], max_check: int = 500) -> Optional[str]:
    """Inspect dataset to detect if it contains samples exclusively from one organism."""
    if not dataset_path or not os.path.exists(dataset_path):
        return None
    try:
        organisms = set()
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
                            organisms.add(org.strip())
                            if len(organisms) > 1:
                                return None
        elif dataset_path.endswith(".csv"):
            import csv
            with open(dataset_path, "r", encoding="utf-8") as f:
                for i, row in enumerate(csv.DictReader(f)):
                    if i >= max_check:
                        break
                    org = row.get("organism")
                    if org:
                        organisms.add(org.strip())
                        if len(organisms) > 1:
                            return None
        if len(organisms) == 1:
            return next(iter(organisms))
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
                if detected and detected.lower() != target_organism.lower():
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
                print(f"Found host-matched dataset: {c}")
                return c
        for c in candidates:
            if os.path.exists(c):
                c_org = detect_dataset_organism(c)
                if c_org is None or c_org.lower() == target_organism.lower():
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

    opt_group = parser.add_argument_group("Configuration")
    opt_group.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config (default: config.yaml)")
    opt_group.add_argument("--model", type=str, default=None, help="Base model identifier")
    opt_group.add_argument("--organism", type=str, default=None, help="Target host organism (e.g. 'Escherichia coli')")
    opt_group.add_argument("--dataset", type=str, default=None, help="Path to dataset (.jsonl or .csv)")
    opt_group.add_argument("--protein", type=str, default=None, help="Target protein sequence for inference")
    opt_group.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    opt_group.add_argument("--max-steps", type=int, default=None, help="Maximum training steps (-1 for full epoch)")
    opt_group.add_argument("--max-samples", type=int, default=None, help="Maximum samples to collect")
    opt_group.add_argument("--output", type=str, default=None, help="Override output directory")

    args = parser.parse_args()

    if not (args.tune or args.test or args.collect):
        parser.print_help()
        print("\nPlease specify one of: --tune, --test, or --collect.")
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
            organism=target_organism,
            config=cfg
        )
        print(f"\nFine-tuning complete. Checkpoint saved to: {model_out}")
        print("To evaluate this model, run:")
        test_cmd = f"  python main.py --test --model {model_out}"
        if target_organism:
            test_cmd += f" --organism \"{target_organism}\""
        print(f"{test_cmd}\n")
        return

    if args.test:
        target_organism = args.organism or model_cfg.get("organism")
        candidate_dir = get_output_dir(model_name, target_organism)
        if args.model:
            model_dir = args.output or args.model
        else:
            if not os.path.exists(candidate_dir) and not target_organism:
                fallback = get_output_dir(model_name, None)
                if os.path.exists(fallback):
                    candidate_dir = fallback
            model_dir = args.output or (model_cfg.get("output_dir") if args.model is None and not target_organism else None) or candidate_dir
        protein_seq = args.protein or infer_cfg.get("benchmark_protein")
        from inference import run_test
        run_test(
            model_dir=model_dir,
            protein_seq=protein_seq,
            organism=target_organism or infer_cfg.get("default_organism"),
            config=cfg
        )
        return


if __name__ == "__main__":
    main()
