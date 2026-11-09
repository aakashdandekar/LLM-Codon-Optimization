#!/usr/bin/env python3
import os
import sys

_venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
if os.path.exists(_venv_py) and os.path.abspath(sys.executable) != os.path.abspath(_venv_py):
    try:
        import datasets  # noqa: F401
    except ImportError:
        os.execv(_venv_py, [_venv_py] + sys.argv)

import json
import re
import random
from typing import Dict, List, Tuple, Any, Optional

from config import load_config
from codon_utils import (
    TRANSLATION_TABLE,
    SYNONYMOUS_CODONS,
    ECOLI_CODON_USAGE,
    YEAST_CODON_USAGE,
    HUMAN_CODON_USAGE,
    translate_dna,
    calculate_gc,
    is_bacteria
)

IUPAC_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

PREDEFINED_DATASETS = [
    "adibvafa/CodonTransformer",
    "alegendaryfish/CodonTranslator-data",
    "jheuschkel/cds-dataset"
]

CURATED_GENES = [
    ("recA", "Escherichia coli", "bacteria", "MAENQKKIQNALEKLKASRLEKATLLEEKLRQSQEELQRQRKQEERKRQEEKL", "ATGGCAGAGAATCAAAAGAAAATTCAGAATGCGCTGGAAAAGTTAAAAGCCAGCCGTTTAGAGAAGGCAACGTTATTAGAGGAGAAGTTACGCCAATCCCAGGAAGAATTACAGCGCCAGCGTAAGCAGGAGGAGCGGAAGCGGCAGGAGGAGAAATTG"),
    ("tufA", "Escherichia coli", "bacteria", "MSKEKFERTKPHVNVGTIGHVDHGKTTLTAAITTVLAKTYGGAARAFDQIDNAPEEKARGITINTSHVEYDTPTRHYAHVDCPGHADYVKNMITGAAQMDGAILVVAATDGPMPQTREHILLGRQVGVPYIIVFLNKCDMVDDEELLELVEMEVRELLSQYDFPGDDTPIVRGSALKALEGDAEWEAKILELAGFLDSYIPEPERAIDKPFLLPIEDVFSISGRGTVVTGRVERGIIKVGEEVEIVGIKETQKSTCTGVEMFRKLLDEGRAGENVGALLRGIKREEIERGQVLAKPGTIKPHTKFESEVYILSKDEGGRHTPFFKGYRPQFYFRTTDVTGTIELPEGVEMVMPGDNIKMVVTLIHPIAMDDGLRFAIREGGRTVGAGVVAKVLS", 
     "ATGTCTAAAGAAAAGTTTGAACGTACAAAACCGCACGTTAACGTTGGTACTATCGGCCACGTTGACCACGGTAAAACTACTCTGACCGCTGCAATCACTACCGTACTGGCTAAAACCTACGGCGGTGCTGCTCGCGCATTCGACCAGATCGATAACGCACCGGAAGAAAAAGCTCGTGGTATCACCATCAACACTTCTCACGTTGAATACGACACTCCGACCCGTCACTACGCACACGTAGACTGCCCGGGTCACGCCGACTATGTTAAAAACATGATCACCGGTGCTGCACAGATGGACGGCGCTATCCTGGTAGTTGCTGCGACTGACGGCCCGATGCCGCAGACTCGTGAGCACATCCTGCTGGGTCGTCAGGTAGGCGTTCCGTACATCATCGTGTTTCTGAACAAATGCGACATGGTTGATGACGAAGAGCTGCTGGAACTGGTTGAAATGGAAGTTCGTGAACTGCTGTCTCAGTACGACTTCCCGGGCGATGACACTCCGATCGTTCGTGGTTCTGCTCTGAAAGCGCTGGAAGGCGACGCAGAGTGGGAAGCGAAAATCCTGGAACTGGCTGGCTTCCTGGATTCTTACATTCCGGAACCAGAGCGTGCGATTGACAAGCCGTTCCTGCTGCCGATCGAAGACGTATTCTCCATCTCCGGTCGTGGTACCGTTGTTACCGGTCGTGTAGAACGCGGTATCATCAAAGTTGGTGAAGAAGTTGAAATCGTTGGTATCAAAGAGACTCAGAAATCTACCTGTACCGGCGTTGAAATGTTCCGCAAACTGCTGGACGAAGGCCGTGCAGGTGAGAACGTAGGTGCGCTGCTGCGTGGTATCAAACGTGAAGAAATCGAACGTGGTCAGGTACTGGCTAAGCCGGGCACCATCAAGCCGCACACCAAGTTCGAATCTGAAGTGTACATTCTGTCCAAAGATGAAGGCGGCCGTCACACTCCGTTCTTCAAAGGCTACCGTCCGCAGTTCTACTTCCGTACTACTGACGTGACTGGTACCATCGAACTGCCGGAAGGCGTAGAGATGGTAATGCCGGGCGATAACATCAAAATGGTTGTTACCCTGATCCACCCGATCGCGATGGACGACGGTCTGCGTTTCGCAATCCGTGAAGGTGGCCGTACCGTTGGCGCGGGCGTTGTAGCAAAAGTTCTGTCCTAA"),
    ("act1", "Saccharomyces cerevisiae", "eukaryota", "MDSEVAALVIDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILTLKYPIEHGIITNWDDMEKIWHHTFYNELRVAPEEHPTLLTEAPLNPKANREKMTQIMFETFNVPAMYVAIQAVLSLYASGRTTGIVLDSGDGVTHVVPIYAGFSLPHAILRIDLAGRDLTDYLMKILTERGYSFTTTAEREIVRDIKEKLCYVALDFEQEMQTAAQSSSLEKSYELPDGQVITIGNERFRCPEALFQPSFLGMESAGIHETTYNSIMKCDVDVRKELYGNIVMSGGTTMFPGIADRMQKEITALAPSSMKVKIIAPPERKYSVWIGGSILASLSTFQQMWISKQEYDESGPSIVHRKCF", "ATGGATTCTGAGGTTGCTGCTTTGGTTATTGATAACGGTTCTGGTATGTGTAAAGCCGGTTTTGCCGGTGACGATGCTCCCAGGGCTGTGTTCCCATCTATCGTGGGTCGTCCCCGACATCAGGGTGTGATGGTCGGTATGGGTCAGAAAGACTCCTACGTTGGTGATGAAGCCCAATCCAAGAGAGGTATCTTGACCCTGAAATACCCCATTGAGCACGGTATTATCACCAACTGGGATGATATGGAGAAGATCTGGCATCATACTTTCTACAACGAGCTTCGTGTTGCCCCTGAAGAACACCCTACTTTGTTGACTGAAGCTCCATTGAACCCAAAAGCCAACAGAGAGAAGATGACCCAGATCATGTTTGAGACCTTCAACGTTCCAGCCATGTACGTTGCCATCCAGGCTGTGTTATCCTTGTACGCCTCTGGTCGTACCACTGGTATTGTTTTGGACTCCGGTGATGGTGTTACTCATGTCGTCCCAATTTACGCTGGTTTCTCTCTACCTCACGCCATTTTGAGAATCGATTTGGCTGGTCGTGATCTTACTGATTACCTCATGAAGATCCTGACTGAACGTGGTTACAGCTTTACTACCACTGCTGAACGTGAAATTGTCCGTGACATCAAGGAGAAACTGTGTTACGTTGCCCTTGACTTTGAGCAGGAGATGCAGACTGCCGCTCAGTCTTCTTCTCTGGAGAAATCTTACGAGTTGCCTGATGGTCAGGTCATCACCATTGGTAACGAGCGTTTCCGTTGCCCTGAGGCTTTGTTCCAGCCTTCTTTCTTGGGTATGGAGTCCGCTGGTATTCATGAAACTACTTACAATTCCATCATGAAGTGTGATGTTGACGTCCGTAAGGAATTATACGGTAACATTGTCATGTCAGGTGGTACCACTATGTTCCCAGGTATTGCCGACCGTATGCAAAAGGAAATCACCGCTTTGGCTCCTTCTTCCATGAAGGTCAAGATCATTGCTCCTCCAGAACGTAAATACTCCGTCTGGATTGGTGGTTCTATCTTGGCTTCTTTGTCTACCTTCCAGCAGATGTGGATTTCCAAGCAGGAGTACGATGAATCTGGTCCATCCATTGTCCACCGTAAATGTTTCTAA"),
    ("ins", "Homo sapiens", "eukaryota", "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN", "ATGGCCCTGTGGATGCGCCTCCTGCCCCTGCTGGCGCTGCTGGCCCTCTGGGGACCTGACCCAGCCGCAGCCTTTGTGAACCAACACCTGTGCGGCTCACACCTGGTGGAAGCTCTCTACCTAGTGTGCGGGGAACGAGGCTTCTTCTACACACCCAAGACCCGCCGGGAGGCAGAGGACCTGCAGGTGGGGCAGGTGGAGCTGGGCGGTGGCCCTGGTGCAGGCAGCCTGCAGCCCTTGGCCCTGGAGGGGTCCCTGCAGAAGCGTGGCATTGTGGAACAATGCTGTACCAGCATCTGCTCCCTCTACCAGCTGGAGAACTACTGCAACTAG")
]


def infer_taxonomy(organism: str) -> str:
    if is_bacteria(organism):
        return "bacteria"
    org = organism.lower()
    if any(k in org for k in ["saccharomyces", "homo", "sapiens", "cricetulus", "mus", "arabidopsis", "drosophila", "yeast"]):
        return "eukaryota"
    elif any(k in org for k in ["methano", "archaea", "sulfolobus"]):
        return "archaea"
    return "unclassified"


def validate_biological_schema(record: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate sequence record: letters, length divisibility, in-frame translation, and stop codons."""
    organism = str(record.get("organism", "")).strip() or "Standard Host"
    protein = record.get("protein", "").upper().strip().rstrip("*_")
    dna = record.get("dna", "").upper().strip()

    if not protein:
        return False, "Empty protein sequence", {}
    if not dna:
        return False, "Empty DNA sequence", {}

    invalid_aa = set(protein) - IUPAC_AMINO_ACIDS
    if invalid_aa:
        return False, f"Invalid amino acids: {invalid_aa}", {}

    dna_core = dna[:-3] if dna.endswith(("TAA", "TAG", "TGA")) else dna
    if not re.fullmatch(r"[ACGT]+", dna):
        return False, "DNA contains non-canonical bases", {}

    expected_len = len(protein) * 3
    if len(dna_core) != expected_len:
        return False, f"Length mismatch: {len(dna_core)} nt vs expected {expected_len} nt", {}

    try:
        translated = translate_dna(dna, allow_stop=True).rstrip("*")
        if translated != protein:
            return False, "Translation does not match protein sequence", {}
    except ValueError as e:
        return False, f"Translation error: {e}", {}

    validated_record = {
        "organism": organism,
        "taxonomy": record.get("taxonomy") or infer_taxonomy(organism),
        "protein": protein,
        "dna": dna,
        "length_aa": len(protein),
        "length_nt": len(dna),
        "gc_content": round(calculate_gc(dna), 2),
        "genetic_code_id": 1,
    }
    return True, "PASSED", validated_record


def generate_synthetic_control(
    n_samples: int = 500,
    seed: int = 42,
    min_length: int = 40,
    max_length: int = 160,
    organism: Optional[str] = None
) -> List[Dict[str, str]]:
    """Generate synthetic protein-coding sequence pairs weighted by organism codon frequencies."""
    records = []

    for name, org, tax, prot, dna in CURATED_GENES:
        if organism and organism.lower() not in org.lower() and org.lower() not in organism.lower():
            continue
        records.append({
            "gene_name": name,
            "organism": org,
            "taxonomy": tax,
            "protein": prot.rstrip("*"),
            "dna": dna
        })

    random.seed(seed)
    amino_acids = list(IUPAC_AMINO_ACIDS)

    host_profiles = [
        ("Escherichia coli", "bacteria", ECOLI_CODON_USAGE),
        ("Saccharomyces cerevisiae", "eukaryota", YEAST_CODON_USAGE),
        ("Homo sapiens", "eukaryota", HUMAN_CODON_USAGE),
    ]

    weights_by_host = {}
    for org, tax, table in host_profiles:
        weights_dict = {}
        for aa, codons in SYNONYMOUS_CODONS.items():
            w = [table.get(c, 1.0) for c in codons]
            weights_dict[aa] = (codons, w)
        weights_by_host[org] = (tax, weights_dict)

    if organism:
        matched_host = None
        for h in weights_by_host:
            if organism.lower() in h.lower() or h.lower() in organism.lower():
                matched_host = h
                break
        if not matched_host:
            matched_host = "Escherichia coli" if is_bacteria(organism) else "Homo sapiens"
        host_names = [matched_host]
    else:
        host_names = list(weights_by_host.keys())

    while len(records) < n_samples:
        org = random.choice(host_names)
        tax, codon_weights = weights_by_host[org]
        length = random.randint(min_length, max_length)
        prot_chars = [random.choice(amino_acids) for _ in range(length)]
        prot = "".join(prot_chars)

        codons = [
            random.choices(codon_weights[aa][0], weights=codon_weights[aa][1], k=1)[0]
            for aa in prot_chars
        ]
        codons.append("TAA")
        dna = "".join(codons)

        records.append({
            "gene_name": f"syn_{len(records) + 1}",
            "organism": organism or org,
            "taxonomy": infer_taxonomy(organism or org),
            "protein": prot,
            "dna": dna
        })

    return records


def fetch_huggingface_dataset(
    max_samples: int = 10000,
    dataset_repos: Optional[List[str]] = None,
    organism: Optional[str] = None
) -> List[Dict[str, str]]:
    repos = dataset_repos if dataset_repos is not None else PREDEFINED_DATASETS
    if not repos:
        return []

    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets library not found; falling back to local generator.")
        return []

    for repo_id in repos:
        print(f"Trying Hugging Face dataset '{repo_id}'...")
        try:
            ds = load_dataset(repo_id, split="train", streaming=True)
            collected = []
            for row in ds:
                org = row.get("organism") or row.get("species") or "Standard Host"
                if organism and organism.lower() not in str(org).lower():
                    continue
                prot = row.get("protein") or row.get("amino_acid_sequence") or ""
                dna = row.get("dna") or row.get("coding_sequence") or ""
                if prot and dna:
                    collected.append({
                        "organism": str(org).strip(),
                        "protein": str(prot).strip(),
                        "dna": str(dna).strip()
                    })
                if len(collected) >= max_samples:
                    break
            if collected:
                print(f"Loaded {len(collected)} samples from {repo_id}")
                return collected
        except Exception as e:
            print(f"Could not load {repo_id} ({e}), trying next repository...")

    return []


def collect_and_validate(
    max_samples: Optional[int] = None,
    output_path: Optional[str] = None,
    organism: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> str:
    cfg = config or load_config()
    data_cfg = cfg.get("data", {})
    target_samples = max_samples if max_samples is not None else data_cfg.get("max_samples", 5000)
    target_output = output_path or data_cfg.get("output_path", "data/dataset.jsonl")
    hf_repos = data_cfg.get("huggingface_datasets", PREDEFINED_DATASETS)
    synth_cfg = data_cfg.get("synthetic", {})

    os.makedirs(os.path.dirname(os.path.abspath(target_output)), exist_ok=True)
    msg = f"Collecting {target_samples} sequence pairs"
    if organism:
        msg += f" for host '{organism}'"
    print(f"{msg} -> {target_output}")

    raw_records = fetch_huggingface_dataset(max_samples=target_samples, dataset_repos=hf_repos, organism=organism)
    if not raw_records:
        source_desc = f"for '{organism}'" if organism else "with host codon weights"
        print(f"No external dataset fetched. Generating synthetic sequence pairs {source_desc}...")
        raw_records = generate_synthetic_control(
            n_samples=target_samples,
            seed=synth_cfg.get("seed", 42),
            min_length=synth_cfg.get("min_length", 40),
            max_length=synth_cfg.get("max_length", 160),
            organism=organism
        )

    print(f"Validating {len(raw_records)} sequences...")
    validated = []
    rejections = {}

    for r in raw_records:
        ok, reason, rec = validate_biological_schema(r)
        if ok:
            validated.append(rec)
        else:
            rejections[reason] = rejections.get(reason, 0) + 1

    with open(target_output, "w", encoding="utf-8") as f:
        for item in validated:
            f.write(json.dumps(item) + "\n")

    print(f"Done: {len(validated)}/{len(raw_records)} sequences passed validation.")
    if rejections:
        for reason, count in rejections.items():
            print(f"  Rejected ({reason}): {count}")
    print(f"Saved dataset to {target_output}")

    return target_output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect and validate codon optimization sequences")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--organism", type=str, default=None, help="Target host organism")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    collect_and_validate(max_samples=args.max_samples, output_path=args.output, organism=args.organism, config=cfg)
