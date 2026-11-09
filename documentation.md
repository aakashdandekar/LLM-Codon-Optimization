# System Documentation: LLM Codon Optimization Pipeline

This document provides a comprehensive technical reference for the biological specifications, mathematical models, machine learning architectures, and system operations implemented across the codebase.

---

## 1. Biological Specifications and Reference Data

### 1.1 Canonical Translation Matrix
The standard genetic code maps 64 triplet nucleotide combinations to 20 standard amino acids and 3 termination signals (`*`).

| Codon | Amino Acid | Codon | Amino Acid | Codon | Amino Acid | Codon | Amino Acid |
|---|---|---|---|---|---|---|---|
| `TTT` | Phenylalanine (F) | `TCT` | Serine (S) | `TAT` | Tyrosine (Y) | `TGT` | Cysteine (C) |
| `TTC` | Phenylalanine (F) | `TCC` | Serine (S) | `TAC` | Tyrosine (Y) | `TGC` | Cysteine (C) |
| `TTA` | Leucine (L) | `TCA` | Serine (S) | `TAA` | Stop (`*`) | `TGA` | Stop (`*`) |
| `TTG` | Leucine (L) | `TCG` | Serine (S) | `TAG` | Stop (`*`) | `TGG` | Tryptophan (W) |
| `CTT` | Leucine (L) | `CCT` | Proline (P) | `CAT` | Histidine (H) | `CGT` | Arginine (R) |
| `CTC` | Leucine (L) | `CCC` | Proline (P) | `CAC` | Histidine (H) | `CGC` | Arginine (R) |
| `CTA` | Leucine (L) | `CCA` | Proline (P) | `CAA` | Glutamine (Q) | `CGA` | Arginine (R) |
| `CTG` | Leucine (L) | `CCG` | Proline (P) | `CAG` | Glutamine (Q) | `CGG` | Arginine (R) |
| `ATT` | Isoleucine (I) | `ACT` | Threonine (T) | `AAT` | Asparagine (N) | `AGT` | Serine (S) |
| `ATC` | Isoleucine (I) | `ACC` | Threonine (T) | `AAC` | Asparagine (N) | `AGC` | Serine (S) |
| `ATA` | Isoleucine (I) | `ACA` | Threonine (T) | `AAA` | Lysine (K) | `AGA` | Arginine (R) |
| `ATG` | Methionine (M) | `ACG` | Threonine (T) | `AAG` | Lysine (K) | `AGG` | Arginine (R) |
| `GTT` | Valine (V) | `GCT` | Alanine (A) | `GAT` | Aspartate (D) | `GGT` | Glycine (G) |
| `GTC` | Valine (V) | `GCC` | Alanine (A) | `GAC` | Aspartate (D) | `GGC` | Glycine (G) |
| `GTA` | Valine (V) | `GCA` | Alanine (A) | `GAA` | Glutamate (E) | `GGA` | Glycine (G) |
| `GTG` | Valine (V) | `GCG` | Alanine (A) | `GAG` | Glutamate (E) | `GGG` | Glycine (G) |

### 1.2 Host Codon Usage Frequencies
Reference frequencies derived from the Kazusa Codon Usage Database (measured per thousand codons across representative genomes):

#### *Escherichia coli* K-12
```
TTT: 22.4, TTC: 16.6, TTA: 13.8, TTG: 13.7, TCT:  8.6, TCC:  8.9, TCA:  7.1, TCG:  8.9
TAT: 16.1, TAC: 12.2, TAA:  2.0, TAG:  0.3, TGT:  5.2, TGC:  6.5, TGA:  1.0, TGG: 15.3
CTT: 11.0, CTC: 11.1, CTA:  3.9, CTG: 52.8, CCT:  7.0, CCC:  5.5, CCA:  8.5, CCG: 23.3
CAT: 12.9, CAC:  9.7, CAA: 15.3, CAG: 28.8, CGT: 20.9, CGC: 22.0, CGA:  3.6, CGG:  5.4
ATT: 30.5, ATC: 25.1, ATA:  4.4, ATG: 27.9, ACT:  8.9, ACC: 23.4, ACA:  7.0, ACG: 14.4
AAT: 17.6, AAC: 21.6, AAA: 33.6, AAG: 10.3, AGT:  8.8, AGC: 16.0, AGA:  2.1, AGG:  1.2
GTT: 18.2, GTC: 15.3, GTA: 10.9, GTG: 26.3, GCT: 15.3, GCC: 25.5, GCA: 20.2, GCG: 33.6
GAT: 32.2, GAC: 19.3, GAA: 39.6, GAG: 17.8, GGT: 24.7, GGC: 29.6, GGA:  8.0, GGG: 11.0
```

#### *Saccharomyces cerevisiae* (Baker's Yeast)
```
TTT: 26.1, TTC: 18.4, TTA: 26.2, TTG: 27.2, TCT: 23.5, TCC: 14.2, TCA: 18.7, TCG:  8.6
TAT: 18.8, TAC: 14.8, TAA:  1.1, TAG:  0.5, TGT:  8.1, TGC:  4.8, TGA:  0.7, TGG: 10.4
CTT: 12.3, CTC:  5.4, CTA: 13.4, CTG: 10.5, CCT: 13.5, CCC:  6.8, CCA: 18.3, CCG:  5.3
CAT: 13.6, CAC:  7.8, CAA: 27.3, CAG: 12.1, CGT:  6.4, CGC:  2.6, CGA:  3.0, CGG:  1.7
ATT: 30.1, ATC: 17.2, ATA: 17.8, ATG: 20.9, ACT: 20.3, ACC: 12.7, ACA: 17.8, ACG:  8.0
AAT: 35.7, AAC: 24.8, AAA: 42.1, AAG: 30.8, AGT: 14.2, AGC:  9.8, AGA: 21.3, AGG:  9.2
GTT: 22.1, GTC: 11.8, GTA: 13.7, GTG: 10.8, GCT: 21.2, GCC: 12.6, GCA: 16.2, GCG:  6.2
GAT: 37.6, GAC: 20.2, GAA: 45.6, GAG: 19.2, GGT: 23.9, GGC:  9.8, GGA: 10.9, GGG:  6.0
```

#### *Homo sapiens* (Human)
```
TTT: 17.6, TTC: 20.3, TTA:  7.7, TTG: 12.9, TCT: 15.2, TCC: 17.7, TCA: 12.2, TCG:  4.4
TAT: 12.2, TAC: 15.3, TAA:  1.0, TAG:  0.8, TGT: 10.6, TGC: 12.6, TGA:  1.6, TGG: 13.2
CTT: 13.2, CTC: 19.6, CTA:  7.2, CTG: 39.6, CCT: 17.5, CCC: 19.8, CCA: 16.9, CCG:  6.9
CAT: 10.9, CAC: 15.1, CAA: 12.3, CAG: 34.2, CGT:  4.5, CGC: 10.4, CGA:  6.2, CGG: 11.4
ATT: 16.0, ATC: 20.8, ATA:  7.5, ATG: 22.0, ACT: 13.1, ACC: 18.9, ACA: 15.1, ACG:  6.1
AAT: 17.0, AAC: 19.1, AAA: 24.4, AAG: 31.9, AGT: 12.1, AGC: 19.5, AGA: 12.2, AGG: 12.0
GTT: 11.0, GTC: 14.5, GTA:  7.1, GTG: 28.1, GCT: 18.4, GCC: 27.7, GCA: 15.8, GCG:  7.4
GAT: 21.8, GAC: 25.1, GAA: 29.0, GAG: 39.6, GGT: 10.8, GGC: 22.2, GGA: 16.5, GGG: 16.5
```

### 1.3 Codon Adaptation Index (CAI) Mathematical Definition
The Codon Adaptation Index (Sharp & Li, 1987) calculates the geometric mean of relative adaptiveness values ($w_i$) for each codon in a gene:

$$w_i = \frac{f_i}{\max_{j \in \text{syn}(aa)} f_j}$$

$$\text{CAI} = \exp\left( \frac{1}{L} \sum_{k=1}^{L} \ln(w_k) \right)$$

Where:
- $f_i$ represents the frequency of codon $i$ in the reference genome.
- $\max_{j \in \text{syn}(aa)} f_j$ is the frequency of the most abundant synonymous codon for that amino acid.
- Codons with no synonymous alternatives (Methionine `ATG` and Tryptophan `TGG`) have $w_i = 1.0$ and are excluded from the summation to prevent skewing the geometric mean.
- A score of $1.0$ represents uniform utilization of the highest-frequency codon across every position. Realistic, well-folded endogenous transcripts typically exhibit balanced CAI values between $0.65$ and $0.85$.

### 1.4 Restriction Endonuclease Recognition Motifs
The pipeline screens sequences against palindromic recognition sequences of common molecular cloning restriction enzymes:
- **EcoRI:** `GAATTC`
- **BamHI:** `GGATCC`
- **HindIII:** `AAGCTT`
- **XhoI:** `CTCGAG`
- **NdeI:** `CATATG`
- **NcoI:** `CCATGG`

---

## 2. Biological Schema Validation Rules

The `data_collection.py` module enforces strict biological validation checks before any sequence pair is committed to the training corpus:

1. **Organism Identifier:** Retained per record as specified by source or generator (e.g. `"Escherichia coli"`, `"Homo sapiens"`, `"Saccharomyces cerevisiae"`).
2. **Taxonomic Classification (`taxonomy`):** High-level domain assignment (`"bacteria"`, `"eukaryota"`, `"archaea"`).
3. **Amino Acid Alphabet:** Must strictly consist of the 20 standard IUPAC characters (`A`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `K`, `L`, `M`, `N`, `P`, `Q`, `R`, `S`, `T`, `V`, `W`, `Y`). Ambiguity codes (`X`, `B`, `Z`, `J`) or non-standard characters cause immediate rejection.
4. **Nucleotide Alphabet:** Must consist exclusively of canonical bases (`A`, `C`, `G`, `T`).
5. **Length Divisibility:** The nucleotide coding sequence length must equal exactly three times the amino acid sequence length:
   $$\text{len}(\text{DNA}) = 3 \times \text{len}(\text{Protein})$$
6. **In-Frame Translation Fidelity:** Translating the DNA sequence using the universal genetic code must reproduce the target amino acid sequence with 100% identity.
7. **Premature Termination Prohibition:** No stop codons (`TAA`, `TAG`, `TGA`) are permitted within the open reading frame prior to the final termination position.
8. **Pre-computed GC Content (`gc_content`):** Pre-calculated percentage of G and C nucleotides for downstream dataset stratification.
9. **Genetic Code ID (`genetic_code_id`):** Standard Genetic Code (NCBI Table 1).

---

## 3. Machine Learning and Fine-Tuning Specifications

### 3.1 Prompt Formatting Specification
The instruction-following prompt structure isolates task instructions from input and output tokens:

```
Below is an instruction that describes a biological task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Optimize the following protein sequence into a high-expression DNA coding sequence for {organism}.

### Input:
{protein_sequence}

### Response:
{dna_sequence}
```

### 3.2 4-Bit QLoRA Configuration (6GB VRAM Optimization)
To train on consumer hardware with 6GB VRAM (e.g., NVIDIA GeForce RTX 4050), the pipeline utilizes Unsloth's optimized kernels and quantization:
- **Quantization:** 4-bit NormalFloat (`load_in_4bit=True`).
- **LoRA Rank ($r$):** 16.
- **LoRA Alpha ($\alpha$):** 32.
- **Target Projection Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Batch Size:** `per_device_train_batch_size = 1`.
- **Gradient Accumulation Steps:** 4 (effective batch size = 4).
- **Optimizer:** `adamw_8bit`.
- **Precision:** `bfloat16` if hardware-supported; otherwise `float16`.
- **Gradient Checkpointing:** Enabled (`use_gradient_checkpointing="unsloth"`).
- **Maximum Sequence Length:** 2048 tokens.

### 3.3 Dynamic Output Directory Resolution
The pipeline normalizes model names and maps them to clean directory paths:
- Models containing `"qwen"` (e.g., `unsloth/Qwen2.5-Coder-1.5B-Instruct`, `qwen3.5-coder`) $\rightarrow$ `models/qwen-codon-optimization-model`
- Models containing `"gpt2"` (e.g., `gpt2-large`) $\rightarrow$ `models/gpt2-codon-optimization-model`
- Models containing `"llama"` (e.g., `meta-llama/Llama-3.2-1B-Instruct`) $\rightarrow$ `models/llama-codon-optimization-model`
- Models containing `"mistral"` $\rightarrow$ `models/mistral-codon-optimization-model`

### 3.4 Decoupled Lifecycle Operations
- `--collect`: Fetches and validates sequence pairs across all available organisms, saving to `data/dataset.jsonl`.
- `--tune`: Resolves dataset path (auto-collecting across organisms if missing), executes training, saves model adapters, prints testing instructions, and exits cleanly.
- `--test`: Loads fine-tuned LoRA checkpoint for accelerated inference and produces comparative biophysical benchmarks.

---

## 4. Configuration Architecture (`config.yaml`)

All pipeline operations draw defaults from `config.yaml`, which is parsed by `config.py:load_config()`. The configuration hierarchy is structured into five functional sections:

- **`model`**: Base HuggingFace identifier (`unsloth/Qwen2.5-Coder-1.5B-Instruct`), output checkpoint directory, maximum token sequence length, and 4-bit quantization toggle.
- **`training`**: Epoch count, step limits, batch size, gradient accumulation steps, 8-bit AdamW optimizer parameters, learning rate, warmup steps, and LoRA hyperparameters ($r=16, \alpha=32$, target projections).
- **`data`**: Output path (`data/dataset.jsonl`), candidate dataset paths for auto-detection, default sample target, remote Hugging Face streaming sources, and synthetic data generator bounds.
- **`inference`**: Target host organism (`Escherichia coli`), sampling temperature ($0.2$), top-p ($0.9$), token generation budget, and canonical benchmark target protein (GFP).
- **`restriction_enzymes`**: Dictionary mapping restriction enzyme identifiers to their recognition sequences (e.g. `EcoRI: GAATTC`, `BamHI: GGATCC`).

Precedence order: **CLI Arguments > `config.yaml` > Hardcoded Fallbacks**.
Environment variable `CODON_CONFIG` can be used to set an alternative config path.
