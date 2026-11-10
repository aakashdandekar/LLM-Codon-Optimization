#!/usr/bin/env python3
import os
import sys
import re
import warnings
from typing import Optional, Dict, Any

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
from codon_utils import (
    validate_candidate,
    calculate_cai,
    calculate_gc,
    detect_restriction_sites,
    naive_frequency_optimizer,
    get_output_dir,
    format_prompt
)

DEFAULT_BENCHMARK_PROTEIN = load_config().get("inference", {}).get(
    "benchmark_protein",
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFF"
    "KDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHY"
    "LSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

format_inference_prompt = format_prompt


def extract_dna_from_output(raw_text: str, expected_length_aa: Optional[int] = None) -> str:
    if "### Response:" in raw_text:
        raw_text = raw_text.split("### Response:")[-1].strip()

    # Extract nucleotide characters from valid sequence lines
    candidate = ""
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = re.sub(r"[^ACGTacgt]", "", line).upper()
        if cleaned and (len(cleaned) / len(stripped)) >= 0.8:
            candidate += cleaned

    if not candidate:
        return ""

    # Locate first in-frame stop codon
    first_stop = None
    for i in range(0, len(candidate) - 2, 3):
        if candidate[i:i+3] in ("TAA", "TAG", "TGA"):
            first_stop = i
            break

    # If expected amino acid length is provided, handle premature stop or extract coding region
    if expected_length_aa is not None:
        expected_nt = expected_length_aa * 3
        if first_stop is not None and first_stop < expected_nt:
            return candidate[:first_stop]
        if len(candidate) >= expected_nt:
            return candidate[:expected_nt]

    if first_stop is not None:
        return candidate[:first_stop]

    # Fallback to nearest multiple of 3
    valid_len = (len(candidate) // 3) * 3
    return candidate[:valid_len] if valid_len > 0 else candidate


def run_test(
    model_dir: Optional[str] = None,
    protein_seq: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    organism: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
):
    cfg = config or load_config()
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("inference", {})

    target_organism = organism or infer_cfg.get("default_organism", "Escherichia coli")
    model_dir = model_dir or (model_cfg.get("output_dir") if not organism else None) or get_output_dir(model_cfg.get("name", "qwen"), target_organism)
    target_protein = (protein_seq or infer_cfg.get("benchmark_protein") or DEFAULT_BENCHMARK_PROTEIN).strip().upper().rstrip("*")
    estimated_tokens = int(len(target_protein) * 1.8) + 64
    config_limit = infer_cfg.get("max_new_tokens", 1024)
    resolved_max_new_tokens = max_new_tokens if max_new_tokens is not None else min(config_limit, max(128, estimated_tokens))
    temperature = temperature if temperature is not None else infer_cfg.get("temperature", 0.2)
    top_p = top_p if top_p is not None else infer_cfg.get("top_p", 0.9)
    max_seq_length = model_cfg.get("max_seq_length", 2048)
    organism = target_organism

    is_remote_hf = not os.path.exists(model_dir) and "/" in model_dir and not model_dir.startswith(("models/", "./", "../", "/"))
    if not os.path.exists(model_dir) and not is_remote_hf:
        print(f"Model directory not found: {model_dir}")
        print("Run fine-tuning first: python main.py --tune")
        return

    if os.path.isdir(model_dir) and not any(os.path.exists(os.path.join(model_dir, f)) for f in ("adapter_config.json", "config.json", "adapter_model.safetensors")):
        print(f"Model directory '{model_dir}' does not contain trained model weights.")
        print("Run fine-tuning first: python main.py --tune")
        return

    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError as e:
        print(f"Error: Missing inference dependencies: {e}")
        print("Please activate your virtual environment ('source .venv/bin/activate') or install dependencies ('uv sync' / 'pip install -r requirements.txt').")
        sys.exit(1)

    has_cuda = torch.cuda.is_available()
    load_in_4bit = model_cfg.get("load_in_4bit", True) and has_cuda

    print(f"Loading model from {model_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    eos_token_ids = []
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id not in eos_token_ids and im_end_id != tokenizer.unk_token_id:
        eos_token_ids.append(im_end_id)

    print(f"Generating DNA for {len(target_protein)} aa protein ({organism})...")
    prompt = format_inference_prompt(target_protein, organism=organism)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    gen_kwargs = {
        "max_new_tokens": resolved_max_new_tokens,
        "use_cache": True,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "top_p": top_p,
    }
    if eos_token_ids:
        gen_kwargs["eos_token_id"] = eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_kwargs
        )

    prompt_len = inputs.input_ids.shape[1]
    generated_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    candidate_dna = extract_dna_from_output(generated_text, expected_length_aa=len(target_protein))

    # Compare against traditional naive frequency selection
    naive_dna = naive_frequency_optimizer(target_protein, organism=organism)

    cand_valid, cand_reason = validate_candidate(target_protein, candidate_dna)
    naive_valid, naive_reason = validate_candidate(target_protein, naive_dna)

    cand_cai = calculate_cai(candidate_dna, organism=organism) if len(candidate_dna) >= 3 else 0.0
    naive_cai = calculate_cai(naive_dna, organism=organism)

    cand_gc = calculate_gc(candidate_dna)
    naive_gc = calculate_gc(naive_dna)

    cand_sites = detect_restriction_sites(candidate_dna)
    naive_sites = detect_restriction_sites(naive_dna)

    print("\n" + "=" * 65)
    print("                    BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Target length: {len(target_protein)} aa ({len(target_protein) * 3} nt expected)")
    print("-" * 65)
    print(f"{'Metric':<25} | {'LLM Model':<17} | {'Traditional Greedy':<17}")
    print("-" * 65)
    print(f"{'Translation Check':<25} | {('PASS' if cand_valid else 'FAIL'):<17} | {('PASS' if naive_valid else 'FAIL'):<17}")
    print(f"{'Generated Length (nt)':<25} | {len(candidate_dna):<17} | {len(naive_dna):<17}")
    print(f"{'Codon Adaptation (CAI)':<25} | {cand_cai:<17.3f} | {naive_cai:<17.3f}")
    print(f"{'GC Content (%)':<25} | {cand_gc:<17.1f}% | {naive_gc:<17.1f}%")
    print(f"{'Restriction Sites':<25} | {len(cand_sites):<17} | {len(naive_sites):<17}")
    print("-" * 65)

    if not cand_valid:
        print(f"Validation failure: {cand_reason}")
    else:
        print("Sequence translated successfully with zero amino acid mismatches.")

    if cand_sites:
        print(f"Found restriction sites in sequence: {', '.join(cand_sites)}")

    print("\nFirst 60 nucleotides:")
    print(f"  Model: {candidate_dna[:60]}...")
    print(f"  Naive: {naive_dna[:60]}...")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned codon optimizer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model-dir", "--model", dest="model_dir", type=str, default=None, help="Path or name of model")
    parser.add_argument("--organism", type=str, default=None, help="Target host organism")
    parser.add_argument("--protein", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_test(
        model_dir=args.model_dir,
        protein_seq=args.protein,
        organism=args.organism,
        config=cfg
    )
