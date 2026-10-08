Yes. The practical version is:

- **Keep the current v21 semantic pipeline intact.**
- **Use local Qwen3.8-27B with 4-bit NF4 + BF16 compute.**
- **Load the model once and run inference serially.**
- **Do not redesign the model interface unless the current loader actually fails.**
- **First objective: get 10–20 real bids through the complete pipeline and inspect the resulting constraints.**
- Only tune context size if there is an actual VRAM problem.

### Files

:chatgpt-content-reference{index="2"}[Download the modified local Transformers extractor](sandbox:/mnt/data/extract_constraints_qwen_local.py)

requirements.txt[Download `requirements.txt`](sandbox:/mnt/data/requirements.txt)

:chatgpt-content-reference{index="4"}[Download the Qwen GPU loading smoke test](sandbox:/mnt/data/test_qwen38_transformers.py)

### Setup

Create an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install a CUDA-enabled PyTorch build. For CUDA 12.6:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Then install the rest:

```bash
pip install -r requirements.txt
```

Check that PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory / 1024**3)"
```

### Don't run a giant test suite

You can optionally run the small loader test:

```bash
python test_qwen38_transformers.py
```

But the **real test is one actual bid**.

First run:

```bash
CUDA_VISIBLE_DEVICES=0 python extract_constraints_qwen_local.py \
    --input_dir ./bid_pdfs \
    --output_dir ./out \
    --limit 1 \
    --concurrency 1 \
    --workers 1 \
    --max_new_tokens 4096 \
    --llm_context_chars 120000
```

If that successfully produces the expected outputs, run the actual batch:

```bash
CUDA_VISIBLE_DEVICES=0 python extract_constraints_qwen_local.py \
    --input_dir ./bid_pdfs \
    --output_dir ./out \
    --concurrency 1 \
    --workers 1 \
    --max_new_tokens 4096 \
    --llm_context_chars 120000
```

### The setup we're sticking with

```text
GeM bid PDFs / linked documents
            │
            ▼
     Existing v21 ingestion
            │
            ▼
   Complete document bundle
            │
            ▼
   Qwen3.8-27B — local GPU
       4-bit NF4
       BF16 compute
       concurrency = 1
            │
            ▼
    Agent 1 extraction
            │
            ▼
       Anchoring
            │
            ▼
       Risk router
            │
            ▼
    Agent 2 validation
            │
            ▼
       Finalizer
            │
            ▼
   final_constraints.csv
```

The existing architecture is deliberately being left alone: the source describes the v21 design as a full-bundle Agent 1 extraction followed by targeted Agent 2 validation, rather than replacing the semantic pipeline with deterministic parsing. Pasted text

### VRAM tuning

Start at:

```text
--llm_context_chars 120000
```

Don't change it preemptively.

If you get a CUDA out-of-memory error:

```bash
--llm_context_chars 80000
```

If **120k works comfortably**, you can later try:

```bash
--llm_context_chars 160000
```

That's the only tuning I would do initially.

The original v21 configuration allowed a much larger combined document context, but the code explicitly requires the complete current document set to fit the configured context rather than semantically splitting it. Pasted text For a 36-GB GPU, starting at 120k characters is the sensible compromise.

The target is simply:

```text
1 bid
  ↓
Qwen loads
  ↓
Agent 1 returns JSON
  ↓
Agent 2 validates
  ↓
final_constraints.csv
  ↓
inspect quality
  ↓
10–20 bids
  ↓
inspect again
  ↓
full dataset
```

One important point: the modified script loads `Qwen/Qwen3.8-27B` locally rather than calling OpenRouter, so **you do not need an OpenRouter API key for this version**. The model is loaded once in the main process; `workers=1` and `concurrency=1` keep the 36-GB GPU from being unnecessarily stressed.

If the loader itself crashes, **paste the exact traceback rather than changing the architecture**. That will tell us whether the problem is the Transformers class, model config, bitsandbytes/CUDA compatibility, tokenizer, or VRAM.