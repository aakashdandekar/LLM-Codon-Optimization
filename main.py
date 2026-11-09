import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import load_config


def get_short_name(model_name: str) -> str:
    name = model_name.lower()
    for arch in ("qwen", "gpt2", "llama", "mistral", "codon"):
        if arch in name:
            return arch
    return name.split("/")[-1].split("-")[0].replace(".", "") or "model"


def get_output_dir(model_name: str) -> str:
    short = get_short_name(model_name)
    return f"models/{short}-codon-optimization-model"


def resolve_or_collect_dataset(
    dataset_arg: str | None,
    max_samples: int | None = None,
    config: dict | None = None
) -> str:
    cfg = config or load_config()
    data_cfg = cfg.get("data", {})
    resolved_max_samples = max_samples if max_samples is not None else data_cfg.get("max_samples", 5000)

    if dataset_arg:
        if os.path.exists(dataset_arg):
            print(f"Using dataset: {dataset_arg}")
            return dataset_arg
        print(f"Warning: specified dataset '{dataset_arg}' not found on disk.")

    candidates = data_cfg.get("candidate_paths", [
        "data/dataset.jsonl",
        "dataset.jsonl",
        "data/dataset.csv",
        "dataset.csv"
    ])
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
        output_file = args.output or data_cfg.get("output_path", "data/dataset.jsonl")
        collect_and_validate(max_samples=max_samples, output_path=output_file, config=cfg)
        print("Data collection complete.")
        return

    if args.tune:
        dataset_path = resolve_or_collect_dataset(args.dataset, max_samples=max_samples, config=cfg)
        model_out = args.output or (model_cfg.get("output_dir") if args.model is None else None) or get_output_dir(model_name)

        from tune import run_tuning
        run_tuning(
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=model_out,
            epochs=epochs,
            max_steps=max_steps,
            config=cfg
        )
        print(f"\nFine-tuning complete. Checkpoint saved to: {model_out}")
        print("To evaluate this model, run:")
        print(f"  python main.py --test --model {model_name}\n")
        return

    if args.test:
        model_dir = args.output or (model_cfg.get("output_dir") if args.model is None else None) or get_output_dir(model_name)
        protein_seq = args.protein or infer_cfg.get("benchmark_protein")
        from inference import run_test
        run_test(
            model_dir=model_dir,
            protein_seq=protein_seq,
            config=cfg
        )
        return


if __name__ == "__main__":
    main()
