import math
import re
from typing import Dict, List, Tuple, Optional

TRANSLATION_TABLE: Dict[str, str] = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

SYNONYMOUS_CODONS: Dict[str, List[str]] = {}
for codon, aa in TRANSLATION_TABLE.items():
    if aa != '*':
        SYNONYMOUS_CODONS.setdefault(aa, []).append(codon)

ECOLI_CODON_USAGE: Dict[str, float] = {
    'TTT': 22.4, 'TTC': 16.6, 'TTA': 13.8, 'TTG': 13.7,
    'TCT': 8.6,  'TCC': 8.9,  'TCA': 7.1,  'TCG': 8.9,
    'TAT': 16.1, 'TAC': 12.2, 'TAA': 2.0,  'TAG': 0.3,
    'TGT': 5.2,  'TGC': 6.5,  'TGA': 1.0,  'TGG': 15.3,
    'CTT': 11.0, 'CTC': 11.1, 'CTA': 3.9,  'CTG': 52.8,
    'CCT': 7.0,  'CCC': 5.5,  'CCA': 8.5,  'CCG': 23.3,
    'CAT': 12.9, 'CAC': 9.7,  'CAA': 15.3, 'CAG': 28.8,
    'CGT': 20.9, 'CGC': 22.0, 'CGA': 3.6,  'CGG': 5.4,
    'ATT': 30.5, 'ATC': 25.1, 'ATA': 4.4,  'ATG': 27.9,
    'ACT': 8.9,  'ACC': 23.4, 'ACA': 7.0,  'ACG': 14.4,
    'AAT': 17.6, 'AAC': 21.6, 'AAA': 33.6, 'AAG': 10.3,
    'AGT': 8.8,  'AGC': 16.0, 'AGA': 2.1,  'AGG': 1.2,
    'GTT': 18.2, 'GTC': 15.3, 'GTA': 10.9, 'GTG': 26.3,
    'GCT': 15.3, 'GCC': 25.5, 'GCA': 20.2, 'GCG': 33.6,
    'GAT': 32.2, 'GAC': 19.3, 'GAA': 39.6, 'GAG': 17.8,
    'GGT': 24.7, 'GGC': 29.6, 'GGA': 8.0,  'GGG': 11.0
}

YEAST_CODON_USAGE: Dict[str, float] = {
    'TTT': 26.1, 'TTC': 18.4, 'TTA': 26.2, 'TTG': 27.2,
    'TCT': 23.5, 'TCC': 14.2, 'TCA': 18.7, 'TCG': 8.6,
    'TAT': 18.8, 'TAC': 14.8, 'TAA': 1.1,  'TAG': 0.5,
    'TGT': 8.1,  'TGC': 4.8,  'TGA': 0.7,  'TGG': 10.4,
    'CTT': 12.3, 'CTC': 5.4,  'CTA': 13.4, 'CTG': 10.5,
    'CCT': 13.5, 'CCC': 6.8,  'CCA': 18.3, 'CCG': 5.3,
    'CAT': 13.6, 'CAC': 7.8,  'CAA': 27.3, 'CAG': 12.1,
    'CGT': 6.4,  'CGC': 2.6,  'CGA': 3.0,  'CGG': 1.7,
    'ATT': 30.1, 'ATC': 17.2, 'ATA': 17.8, 'ATG': 20.9,
    'ACT': 20.3, 'ACC': 12.7, 'ACA': 17.8, 'ACG': 8.0,
    'AAT': 35.7, 'AAC': 24.8, 'AAA': 42.1, 'AAG': 30.8,
    'AGT': 14.2, 'AGC': 9.8,  'AGA': 21.3, 'AGG': 9.2,
    'GTT': 22.1, 'GTC': 11.8, 'GTA': 13.7, 'GTG': 10.8,
    'GCT': 21.2, 'GCC': 12.6, 'GCA': 16.2, 'GCG': 6.2,
    'GAT': 37.6, 'GAC': 20.2, 'GAA': 45.6, 'GAG': 19.2,
    'GGT': 23.9, 'GGC': 9.8,  'GGA': 10.9, 'GGG': 6.0
}

HUMAN_CODON_USAGE: Dict[str, float] = {
    'TTT': 17.6, 'TTC': 20.3, 'TTA': 7.7,  'TTG': 12.9,
    'TCT': 15.2, 'TCC': 17.7, 'TCA': 12.2, 'TCG': 4.4,
    'TAT': 12.2, 'TAC': 15.3, 'TAA': 1.0,  'TAG': 0.8,
    'TGT': 10.6, 'TGC': 12.6, 'TGA': 1.6,  'TGG': 13.2,
    'CTT': 13.2, 'CTC': 19.6, 'CTA': 7.2,  'CTG': 39.6,
    'CCT': 17.5, 'CCC': 19.8, 'CCA': 16.9, 'CCG': 6.9,
    'CAT': 10.9, 'CAC': 15.1, 'CAA': 12.3, 'CAG': 34.2,
    'CGT': 4.5,  'CGC': 10.4, 'CGA': 6.2,  'CGG': 11.4,
    'ATT': 16.0, 'ATC': 20.8, 'ATA': 7.5,  'ATG': 22.0,
    'ACT': 13.1, 'ACC': 18.9, 'ACA': 15.1, 'ACG': 6.1,
    'AAT': 17.0, 'AAC': 19.1, 'AAA': 24.4, 'AAG': 31.9,
    'AGT': 12.1, 'AGC': 19.5, 'AGA': 12.2, 'AGG': 12.0,
    'GTT': 11.0, 'GTC': 14.5, 'GTA': 7.1,  'GTG': 28.1,
    'GCT': 18.4, 'GCC': 27.7, 'GCA': 15.8, 'GCG': 7.4,
    'GAT': 21.8, 'GAC': 25.1, 'GAA': 29.0, 'GAG': 39.6,
    'GGT': 10.8, 'GGC': 22.2, 'GGA': 16.5, 'GGG': 16.5
}

def _build_relative_adaptiveness(usage_table: Dict[str, float]) -> Dict[str, float]:
    w: Dict[str, float] = {}
    for aa, codons in SYNONYMOUS_CODONS.items():
        max_freq = max(usage_table.get(c, 0.001) for c in codons)
        for c in codons:
            w[c] = usage_table.get(c, 0.001) / max(max_freq, 0.001)
    return w

_ECOLI_W = _build_relative_adaptiveness(ECOLI_CODON_USAGE)
_YEAST_W = _build_relative_adaptiveness(YEAST_CODON_USAGE)
_HUMAN_W = _build_relative_adaptiveness(HUMAN_CODON_USAGE)

def _build_log_w(w_table: Dict[str, float]) -> Dict[str, float]:
    return {c: math.log(max(w, 0.001)) for c, w in w_table.items()}

_ECOLI_LOG_W = _build_log_w(_ECOLI_W)
_YEAST_LOG_W = _build_log_w(_YEAST_W)
_HUMAN_LOG_W = _build_log_w(_HUMAN_W)

ORGANISM_TABLES = {
    'ecoli': _ECOLI_W,
    'escherichia': _ECOLI_W,
    'yeast': _YEAST_W,
    'saccharomyces': _YEAST_W,
    'human': _HUMAN_W,
    'homo': _HUMAN_W,
}

ORGANISM_LOG_TABLES = {
    'ecoli': _ECOLI_LOG_W,
    'escherichia': _ECOLI_LOG_W,
    'yeast': _YEAST_LOG_W,
    'saccharomyces': _YEAST_LOG_W,
    'human': _HUMAN_LOG_W,
    'homo': _HUMAN_LOG_W,
}

_COMPLEMENT = str.maketrans("ATGC", "TACG")

def reverse_complement(dna: str) -> str:
    """Return the reverse complement of a nucleotide sequence."""
    return dna.upper().translate(_COMPLEMENT)[::-1]

BACTERIA_KEYWORDS = (
    "escherichia", "coli", "bacillus", "pseudomonas", "streptomyces",
    "bacterium", "bacteria", "salmonella", "staphylococcus", "streptococcus",
    "clostridium", "vibrio", "mycobacterium", "helicobacter", "lactobacillus",
    "corynebacterium", "rhizobium", "agrobacterium", "mycoplasma"
)

def is_bacteria(organism: str) -> bool:
    """Check if organism belongs to bacteria domain."""
    if not organism:
        return False
    org = organism.lower().strip()
    return any(k in org for k in BACTERIA_KEYWORDS)

def get_bacteria_slug(organism: str) -> Optional[str]:
    """Return a clean slug for a bacteria (e.g. 'Escherichia coli' -> 'ecoli')."""
    if not is_bacteria(organism):
        return None
    org = organism.lower().strip()
    if any(k in org for k in ("escherichia", "coli", "ecoli")):
        return "ecoli"
    slug = re.sub(r"[^a-z0-9]+", "-", org).strip("-")
    return slug or None

def get_short_name(model_name: str) -> str:
    if not model_name:
        return "model"
    name = model_name.lower()
    for arch in ("qwen", "gpt2", "llama", "mistral", "codon"):
        if arch in name:
            return arch
    return name.split("/")[-1].split("-")[0].replace(".", "") or "model"

def get_model_name(model_name: str, organism: Optional[str] = None) -> str:
    """Return model name, appending bacteria slug if model is specific to a bacterium."""
    short = get_short_name(model_name)
    slug = get_bacteria_slug(organism) if organism else None
    if not slug:
        for kw in BACTERIA_KEYWORDS:
            if kw in model_name.lower():
                slug = get_bacteria_slug(kw)
                break
    if slug and slug not in short:
        return f"{short}-{slug}-codon-optimization-model"
    return f"{short}-codon-optimization-model"

def get_output_dir(model_name: str, organism: Optional[str] = None) -> str:
    """Resolve output directory incorporating bacteria slug if specific to a bacterium."""
    return f"models/{get_model_name(model_name, organism)}"

DEFAULT_RESTRICTION_SITES: Dict[str, str] = {
    'EcoRI': 'GAATTC',
    'BamHI': 'GGATCC',
    'HindIII': 'AAGCTT',
    'XhoI': 'CTCGAG',
    'NdeI': 'CATATG',
    'NcoI': 'CCATGG'
}

def translate_dna(dna: str, allow_stop: bool = True) -> str:
    dna = dna.upper().strip()
    if len(dna) % 3 != 0:
        raise ValueError(f"DNA length must be divisible by 3, got length {len(dna)}")
    
    protein = []
    num_codons = len(dna) // 3
    for i in range(num_codons):
        codon = dna[i*3:(i+1)*3]
        if codon not in TRANSLATION_TABLE:
            raise ValueError(f"Invalid codon '{codon}' at position {i*3}")
        aa = TRANSLATION_TABLE[codon]
        if aa == '*':
            if i != num_codons - 1:
                raise ValueError(f"Premature internal stop codon '{codon}' at codon position {i}")
            if allow_stop:
                protein.append('*')
        else:
            protein.append(aa)
            
    return "".join(protein)

def validate_candidate(target_protein: str, candidate_dna: str) -> Tuple[bool, str]:
    target = target_protein.upper().rstrip('*')
    cand_dna = candidate_dna.upper().strip()
    
    if len(cand_dna) < len(target) * 3:
        return False, f"DNA too short ({len(cand_dna)} nt for {len(target)} aa)"
    
    try:
        translated = translate_dna(cand_dna, allow_stop=True).rstrip('*')
        if translated != target:
            return False, f"Translation mismatch: expected {len(target)} aa, got {len(translated)} aa"
        return True, "PASSED"
    except ValueError as e:
        return False, str(e)

def format_prompt(protein: str, organism: Optional[str] = None, dna: Optional[str] = None, eos_token: str = "") -> str:
    """Format standard biological instruction prompt for tuning and inference."""
    target_organism = organism or "Escherichia coli"
    prompt = (
        "Below is an instruction that describes a biological task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"Optimize the following protein sequence into a high-expression DNA coding sequence for {target_organism}.\n\n"
        "### Input:\n"
        f"{protein}\n\n"
        "### Response:\n"
    )
    if dna:
        prompt += dna + eos_token
    return prompt

def resolve_organism_key(organism: Optional[str]) -> str:
    """Map organism string or abbreviation to canonical key ('ecoli', 'yeast', 'human')."""
    if not organism:
        return "ecoli"
    org = organism.lower().strip()
    if any(k in org for k in ("yeast", "saccharomyces", "cerevisiae")):
        return "yeast"
    if any(k in org for k in ("human", "homo", "sapiens")):
        return "human"
    return "ecoli"

_CACHED_ENZYMES = None

def calculate_cai(dna: str, usage_table: Dict[str, float] | None = None, organism: str | None = None) -> float:
    dna = dna.upper().strip()
    if len(dna) < 3 or len(dna) % 3 != 0:
        return 0.0
    
    if usage_table is not None:
        w_table = _build_relative_adaptiveness(usage_table)
        log_w = _build_log_w(w_table)
    elif organism:
        org_key = resolve_organism_key(organism)
        log_w = ORGANISM_LOG_TABLES.get(org_key, _ECOLI_LOG_W)
    else:
        log_w = _ECOLI_LOG_W
    
    valid_codons = 0
    log_sum = 0.0
    count = 0
    num_codons = len(dna) // 3
    for i in range(num_codons):
        codon = dna[i*3:(i+1)*3]
        if codon in TRANSLATION_TABLE:
            valid_codons += 1
        if codon in log_w:
            if codon not in ('ATG', 'TGG'):
                log_sum += log_w[codon]
                count += 1
                
    if valid_codons == 0:
        return 0.0
    if count == 0:
        return 1.0
    return math.exp(log_sum / count)

def calculate_gc(dna: str) -> float:
    dna = dna.upper().strip()
    if not dna:
        return 0.0
    gc_count = dna.count('G') + dna.count('C')
    return (gc_count / len(dna)) * 100.0

def detect_restriction_sites(dna: str, enzymes: Dict[str, str] | None = None) -> List[str]:
    global _CACHED_ENZYMES
    dna = dna.upper().strip()
    if enzymes is None:
        if _CACHED_ENZYMES is None:
            try:
                from config import load_config
                _CACHED_ENZYMES = load_config().get("restriction_enzymes") or DEFAULT_RESTRICTION_SITES
            except Exception:
                _CACHED_ENZYMES = DEFAULT_RESTRICTION_SITES
        enzymes = _CACHED_ENZYMES
    found = []
    for name, site in enzymes.items():
        s = site.upper()
        rev_s = reverse_complement(s)
        if s in dna:
            found.append(f"{name} ({s})")
        elif rev_s != s and rev_s in dna:
            found.append(f"{name} ({s} rev)")
    return found

def naive_frequency_optimizer(
    protein: str,
    usage_table: Dict[str, float] | None = None,
    organism: str | None = None
) -> str:
    protein = protein.upper().rstrip('*')
    if usage_table is not None:
        table = usage_table
    elif organism:
        org_key = resolve_organism_key(organism)
        table = YEAST_CODON_USAGE if org_key == "yeast" else (HUMAN_CODON_USAGE if org_key == "human" else ECOLI_CODON_USAGE)
    else:
        table = ECOLI_CODON_USAGE
    
    best_codons: Dict[str, str] = {}
    for aa, codons in SYNONYMOUS_CODONS.items():
        best_codons[aa] = max(codons, key=lambda c: table.get(c, 0.0))
        
    dna = []
    for aa in protein:
        if aa in best_codons:
            dna.append(best_codons[aa])
        else:
            raise ValueError(f"Unknown amino acid '{aa}'")
    return "".join(dna)
