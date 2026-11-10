# Codon Optimization Prompt Template

This document defines the standard prompt template used for fine-tuning and inference in the **CodonLLM** pipeline.

---

## 1. Prompt Format

The model was instruction fine-tuned using an **Alpaca-style template**. It conditions sequence generation on both the target amino acid sequence and the destination host organism.

### Inference Prompt Template
Send this exact prompt to the model when generating an optimized sequence:

```text
Below is an instruction that describes a biological task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Optimize the following protein sequence into a high-expression DNA coding sequence for {organism}.

### Input:
{protein}

### Response:

```

> **Note:** The trailing `### Response:\n` acts as the trigger sequence. The model generates the optimized nucleotide sequence immediately following this line.

---

## 2. Example Prompt

### Input
- **Organism:** `Escherichia coli`
- **Protein Sequence:** `MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGE`

### Formatted Prompt
```text
Below is an instruction that describes a biological task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Optimize the following protein sequence into a high-expression DNA coding sequence for Escherichia coli.

### Input:
MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGE

### Response:

```

### Expected Model Output
```text
ATGTCTAAAGGCGAAGAGCTGTTCACTGGCGTTGTCCCGATCCTGGTTGAACTGGACGGCGACGTTAACGGTCACAAATTCTCCGTCTCCGGCGAAGGCGAA
```

---

## 3. Supported Host Organisms

The model dynamically adjusts synonymous codon weights based on the specified organism:
- **`Escherichia coli`** (or `ecoli`) — Bacterial host using bacterial codon adaptation tables.
- **`Saccharomyces cerevisiae`** (or `yeast`) — Yeast eukaryote host using fungal codon frequencies.
- **`Homo sapiens`** (or `human`) — Human/mammalian expression system.

---

## 4. Programmatic Usage (Python)

You do not need to construct this prompt string manually. Use the built-in helper in `codon_utils.py`:

```python
from codon_utils import format_prompt

# Generate prompt for inference
prompt = format_prompt(
    protein="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGE",
    organism="Escherichia coli"
)

# Pass to tokenizer
# inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
```

Or run inference directly from the CLI:

```bash
python main.py --test \
  --protein "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGE" \
  --organism "Escherichia coli"
```
