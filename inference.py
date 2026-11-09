import os
import sys
import re
from typing import Optional, Dict, Any

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import load_config
from codon_utils import (
    validate_candidate,
    calculate_cai,
    calculate_gc,
    detect_restriction_sites,
    naive_frequency_optimizer
)

DEFAULT_BENCHMARK_PROTEIN = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFF"
    "KDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHY"
    "LSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)


def format_inference_prompt(protein: str, organism: Optional[str] = None) -> str:
    target_organism = organism or "Escherichia coli"
    return (
        "Below is an instruction that describes a biological task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"Optimize the following protein sequence into a high-expression DNA coding sequence for {target_organism}.\n\n"
        "### Input:\n"
        f"{protein}\n\n"
        "### Response:\n"
    )


def extract_dna_from_output(raw_text: str, expected_length_aa: Optional[int] = None) -> str:
    if "### Response:" in raw_text:
        raw_text = raw_text.split("### Response:")[-1].strip()

    # Extract nucleotide characters from the output lines
    candidate = ""
    for line in raw_text.splitlines():
        cleaned = re.sub(r"[^ACGTacgt]", "", line).upper()
        if cleaned:
            candidate += cleaned

    if not candidate:
        return ""

    # If expected amino acid length is provided, extract exact coding region
    if expected_length_aa is not None:
        expected_nt = expected_length_aa * 3
        if len(candidate) >= expected_nt:
            return candidate[:expected_nt]

    # Otherwise scan for the first in-frame stop codon
    for i in range(0, len(candidate) - 2, 3):
        codon = candidate[i:i+3]
        if codon in ("TAA", "TAG", "TGA"):
            return candidate[:i]

    # Fallback to nearest multiple of 3
    valid_len = (len(candidate) // 3) * 3
    return candidate[:valid_len] if valid_len > 0 else candidate


def run_test(
    model_dir: Optional[str] = None,
    protein_seq: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
):
    cfg = config or load_config()
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("inference", {})

    model_dir = model_dir or model_cfg.get("output_dir", "models/qwen-codon-optimization-model")
    target_protein = (protein_seq or infer_cfg.get("benchmark_protein") or DEFAULT_BENCHMARK_PROTEIN).strip().upper().rstrip("*")
    estimated_tokens = int(len(target_protein) * 1.8) + 64
    config_limit = infer_cfg.get("max_new_tokens", 1024)
    resolved_max_new_tokens = max_new_tokens if max_new_tokens is not None else min(config_limit, max(128, estimated_tokens))
    temperature = temperature if temperature is not None else infer_cfg.get("temperature", 0.2)
    top_p = top_p if top_p is not None else infer_cfg.get("top_p", 0.9)
    max_seq_length = model_cfg.get("max_seq_length", 2048)
    load_in_4bit = model_cfg.get("load_in_4bit", True)
    organism = infer_cfg.get("default_organism", "Escherichia coli")

    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        print("Run fine-tuning first: python main.py --tune")
        return

    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError as e:
        print(f"Error: Missing inference dependencies: {e}")
        print("Please ensure unsloth and torch are installed.")
        sys.exit(1)

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
    naive_dna = naive_frequency_optimizer(target_protein)

    cand_valid, cand_reason = validate_candidate(target_protein, candidate_dna)
    naive_valid, naive_reason = validate_candidate(target_protein, naive_dna)

    cand_cai = calculate_cai(candidate_dna, organism=organism) if cand_valid else 0.0
    naive_cai = calculate_cai(naive_dna, organism=organism)

    cand_gc = calculate_gc(candidate_dna) if cand_valid else 0.0
    naive_gc = calculate_gc(naive_dna)

    cand_sites = detect_restriction_sites(candidate_dna) if cand_valid else []
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
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--protein", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_test(
        model_dir=args.model_dir,
        protein_seq=args.protein,
        config=cfg
    )
