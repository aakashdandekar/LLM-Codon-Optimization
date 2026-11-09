# LLM-Based Codon Optimization Pipeline

An end-to-end framework for fine-tuning and evaluating autoregressive Large Language Models (LLMs) on synonymous codon optimization tasks. The pipeline implements memory-efficient 4-bit Quantized Low-Rank Adaptation (QLoRA) using Unsloth, specifically configured to operate within consumer GPU environments (such as the NVIDIA GeForce RTX 4050 with 6 GB VRAM).

---

## 1. Overview and Biological Motivation

### 1.1 The Genetic Code and Synonymous Redundancy
The canonical genetic code maps 64 triplet nucleotide codons to 20 standard amino acids and 3 termination signals. Because multiple codons specify the same amino acid, protein coding sequences contain synonymous redundancy. While synonymous mutations do not alter primary amino acid sequences, codon choice strongly influences translational efficiency, mRNA stability, and protein folding kinetics within a target host organism.

### 1.2 Limitations of Classical Heuristic Tools
Traditional codon optimization software relies on position-independent lookup tables or greedy frequency substitutions (e.g., selecting the most frequent host codon across all positions to maximize the Codon Adaptation Index). In practical heterologous expression, this strategy frequently fails due to:
- **Disrupted Co-Translational Folding:** Nascent polypeptide chains fold hierarchically as they emerge from the ribosome exit tunnel. Uniformly maximal elongation rates eliminate natural translational pauses, resulting in protein misfolding and insoluble inclusion body aggregation.
- **5' Translational Ramp Destruction:** Endogenous high-expression transcripts feature slower, rarer codons within the initial 30 to 50 codons. This translational ramp meters ribosomal entry and prevents ribosomal collisions near the initiation site.
- **Dicodon and Context Invariance:** Table-based methods overlook codon-pair (dicodon) bias and contextual interactions that influence tRNA translocation kinetics.
- **Repetitive Motifs and Secondary Structure:** Naive maximization induces homopolymeric runs and stable mRNA secondary structures that hinder translation initiation and chemical DNA synthesis.

### 1.3 The Learned Sequence Model Approach
This pipeline trains an autoregressive language model on validated endogenous coding sequences to learn contextual codon harmonization. By conditioning codon selection on preceding sequence context, the model balances translational ramp preservation, dicodon cadence, and global GC content while strictly preserving the target amino acid sequence.

---

## 2. System Architecture

The pipeline consists of modular, decoupled components:

1. **`config.yaml` / `config.py`**: Central declarative configuration and lightweight loader providing unified hyperparameters, model checkpoints, dataset settings, inference defaults, and restriction enzyme targets with CLI override support.
2. **`main.py`**: Central command-line interface. Coordinates data collection, fine-tuning, and inference benchmarking. Manages dynamic model naming conventions, config resolution, and automated dataset path resolution.
3. **`tune.py`**: Dedicated fine-tuning engine using `unsloth.FastLanguageModel`. Applies 4-bit QLoRA ($r=16, \alpha=32$) with gradient checkpointing and 8-bit AdamW optimization.
4. **`inference.py`**: Generation and benchmarking module. Enforces translation integrity, computes biophysical proxy metrics, and produces side-by-side comparative evaluations against greedy baseline algorithms.
5. **`data_collection.py`**: Data acquisition and biological schema validation module. Extracts pairs from sequence repositories and verifies IUPAC compliance, exact length divisibility, in-frame translation fidelity, and absence of internal stop codons.
6. **`codon_utils.py`**: Biological utility library containing translation matrices, the Kazusa *E. coli* K-12 codon frequency distribution, Codon Adaptation Index (CAI) calculators, GC-content analyzers, and restriction endonuclease site scanners.

---

## 3. Installation and Dependencies

The pipeline requires Python 3.10–3.12 and a CUDA-capable GPU.

```bash
# Create and activate a Python 3.12 virtual environment
uv venv .venv --python /usr/bin/python3.12
source .venv/bin/activate

# Install dependencies
uv pip install unsloth trl datasets transformers accelerate bitsandbytes pyyaml
```

---

## 4. Execution Workflow

### 4.1 Data Collection and Biological Schema Validation
To acquire and validate coding sequence pairs across all available organisms:
```bash
python main.py --collect --max-samples 5000 --output data/dataset.jsonl
```

Every candidate pair undergoes validation against five schema rules:
- Amino acid sequence contains only standard IUPAC single-letter codes (`ACDEFGHIKLMNPQRSTVWY`).
- Nucleotide sequence contains only canonical bases (`A`, `C`, `G`, `T`).
- Exact length correspondence: $\text{length}(\text{DNA}) = 3 \times \text{length}(\text{Protein})$.
- In-frame translation matches the target protein with 100% sequence identity.
- No internal termination codons prior to the final position.

### 4.2 Fine-Tuning
Fine-tuning is decoupled from evaluation. If no dataset path is specified via `--dataset`, the pipeline automatically resolves local candidate files or triggers `data_collection.py`.

```bash
python main.py --tune --model unsloth/Qwen2.5-Coder-1.5B-Instruct --epochs 1
```

Parameters:
- `--config`: Path to YAML configuration file (default: `config.yaml`).
- `--model`: Base model identifier (defaults to `model.name` in `config.yaml`).
- `--dataset`: Path to training dataset (.jsonl or .csv). Auto-detected or auto-collected if omitted.
- `--epochs`: Number of complete training epochs (defaults to `training.epochs` in `config.yaml`).
- `--max-steps`: Explicit step ceiling for dry-run verification (defaults to `training.max_steps` in `config.yaml`).
- `--output`: Override output directory for checkpoints.

Trained weights and adapters are saved directly to:
```
models/<normalized-model-name>-codon-optimization-model/
```
Upon completion, the training process exits without initiating inference.

### 4.3 Inference and Benchmarking
To evaluate the fine-tuned model against traditional greedy substitution:
```bash
python main.py --test --model unsloth/Qwen2.5-Coder-1.5B-Instruct
```

To evaluate a specific target protein sequence:
```bash
python main.py --test --model unsloth/Qwen2.5-Coder-1.5B-Instruct --protein "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEG..."
```

---

## 5. Evaluation Metrics

Model performance is evaluated across both hard constraints and quantitative biophysical proxies:

| Metric | Target / Standard | Description |
|---|---|---|
| **Hard Constraint Pass** | 100% Identity | Binary verification that generated DNA translates identically to input protein with zero missense or nonsense mutations. |
| **Codon Adaptation Index (CAI)** | $0.65 - 0.85$ (Balanced) | Geometric mean of relative synonymous codon adaptiveness ($w_i$) relative to reference *E. coli* genes. Avoids artificial maximization to 1.0. |
| **GC Content (%)** | $45.0\% - 55.0\%$ | Quantifies overall and windowed GC distribution relative to host physiological equilibrium. |
| **Restriction Endonuclease Sites** | 0 occurrences | Scans for forbidden cloning motifs (EcoRI, BamHI, HindIII, XhoI, NdeI, NcoI). |
| **Sequence Complexity** | Low Repetition | Assesses local sequence repetition to prevent homopolymeric DNA synthesis failures and secondary structure hairpins. |

---

## 6. Directory Structure

```
├── main.py                  # CLI driver and execution coordinator
├── tune.py                  # Unsloth 4-bit QLoRA fine-tuning module
├── inference.py             # Inference generation and comparative benchmarking
├── data_collection.py       # Data collection and biological schema validation
├── codon_utils.py           # Genetic code dictionaries, CAI, GC, and motif scanning
├── data/
│   └── dataset.jsonl        # Validated training and evaluation sequences
└── models/
    └── <shortname>-codon-optimization-model/  # Model checkpoints and LoRA adapters
```
