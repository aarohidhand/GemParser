# GeM Bid Constraint Extraction — Local Qwen on a 36 GB NVIDIA GPU

This package runs the GeM bid constraint extraction pipeline locally using Hugging Face Transformers and a 4-bit Qwen3.8-27B checkpoint. It avoids OpenRouter/API quotas and uses the same extraction, document-ingestion, canonicalization, and validation pipeline as the API version.

> **Hardware target:** one NVIDIA GPU with 36 GB VRAM. The default model is a third-party 4-bit NF4 checkpoint, `greghavens/Qwen3.8-27B-bnb-4bit`. It is not an official Qwen-hosted quantization. Review its model card and license before using it. The pipeline has not been live-tested on your exact GPU; complete the smoke test below before processing a batch.

## 1. What you need

- Ubuntu 22.04/24.04 recommended. Windows users should preferably use WSL2 + Ubuntu.
- NVIDIA GPU with 36 GB VRAM and a compatible NVIDIA driver.
- At least 64 GB system RAM recommended; 32 GB may work for smaller jobs, but document processing and model loading can use substantial RAM.
- 50–80 GB free disk space for the environment, model cache, input documents, and outputs.
- Python 3.11 recommended.
- Tesseract OCR installed as a separate system application if OCR fallback is needed. `pytesseract` in `requirements.txt` is only the Python wrapper; it does not install the Tesseract executable.

## 2. Download the project files

Put these files in the same project folder:

- `extract_constraints_qwen_local.py`
- `requirements.txt`
- `README.md`

Create two folders next to them:

```bash
mkdir -p bid_pdfs output
```

Put the main GeM bid PDFs in `bid_pdfs/`. The script may retrieve linked documents when links are available and network access is permitted. For reproducible or offline runs, keep the relevant linked documents alongside the bid or in the expected downloaded linked-document structure.

## 3. Verify the NVIDIA driver and GPU

```bash
nvidia-smi
```

Confirm the GPU appears and has roughly 36 GB VRAM. Install or update the NVIDIA driver using the official instructions for the operating system and GPU.

Do not install a random CUDA toolkit just because PyTorch uses CUDA. The PyTorch wheel includes its required CUDA runtime libraries. A working, compatible NVIDIA driver is still required.

## 4. Install Tesseract OCR (if OCR fallback is needed)

Tesseract is not installed by `pip`. Download it from the official Tesseract project GitHub page:

- Main project: https://github.com/tesseract-ocr/tesseract
- Windows installer builds (UB Mannheim): https://github.com/UB-Mannheim/tesseract/wiki

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr

tesseract --version
which tesseract
```

If a language other than English is needed, install the corresponding language package, for example `tesseract-ocr-hin` for Hindi.

### Windows

Download and run the installer linked from the UB Mannheim Tesseract wiki. During installation, note the executable path, commonly:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Add that folder to `PATH`, reopen the terminal, then check:

```powershell
tesseract --version
```

If Tesseract is installed but not on `PATH`, set `TESSERACT_CMD` to the full path to `tesseract.exe` before running the script. For example, in PowerShell:

```powershell
$env:TESSERACT_CMD = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## 5. Create a Python virtual environment

From the project folder on Ubuntu/WSL2:

```bash
sudo apt-get install -y python3.11 python3.11-venv python3-pip git
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

On Windows, create and activate a Python 3.11 virtual environment using the equivalent commands:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

## 6. Install PyTorch with CUDA support

Install a CUDA-enabled PyTorch build **before** installing the rest of the requirements. Use the official selector to choose the current command for your operating system and supported CUDA runtime:

https://pytorch.org/get-started/locally/

For example, for a Linux x86-64 machine with a compatible NVIDIA driver and CUDA 12.8 wheel support, the command is typically:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

If the official selector offers a different CUDA wheel for your platform, use that command instead of copying the example blindly.

Verify that CUDA is visible to PyTorch:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM GiB:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    )
PY
```

`CUDA available` must print `True` before continuing.

## 7. Install the Python requirements

Use the provided requirements file rather than installing packages one by one:

```bash
python -m pip install -r requirements.txt
```

The important inference packages are:

- `transformers`: model loading, tokenizer/processor, chat template, and generation.
- `accelerate`: device placement via `device_map="auto"`.
- `bitsandbytes`: support for 4-bit model layers and quantization.
- `safetensors` and `huggingface_hub`: model weight download and loading.

The remaining dependencies cover PDF, DOCX, spreadsheet, OCR, web-download, JSON-repair, and progress-bar support used by the pipeline.

Check the installation:

```bash
python - <<'PY'
import torch
import transformers
import accelerate
import bitsandbytes
import fitz
import pdfplumber
import pytesseract

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
print("CUDA:", torch.cuda.is_available())

try:
    print("Tesseract:", pytesseract.get_tesseract_version())
except Exception as exc:
    print("Tesseract unavailable:", exc)
PY
```

If the model architecture is not recognized by the installed Transformers version, check the model card and the version requirements before upgrading:

```bash
python -m pip install --upgrade transformers accelerate bitsandbytes
```

Restart the terminal or Python process after changing packages. Do not add `trust_remote_code=True` as a blind workaround. First verify the model repository and its documented loading instructions.

## 8. Model download and quantization

The default model ID is:

```text
greghavens/Qwen3.8-27B-bnb-4bit
```

Model card: https://huggingface.co/greghavens/Qwen3.8-27B-bnb-4bit

This checkpoint is already quantized to 4-bit NF4. Do not quantize it again or pass a second `BitsAndBytesConfig` unless the repository's documented loading instructions specifically require it. The model's existing quantization configuration should be read when loading.

The first run downloads the weights to the Hugging Face cache. Ensure there is enough free disk space and a working connection to Hugging Face.

You can pre-download the model before the first extraction run:

```bash
hf download greghavens/Qwen3.8-27B-bnb-4bit
```

If the `hf` command is not found, confirm that `huggingface_hub` is installed and that the virtual environment is activated. Older versions may expose the legacy `huggingface-cli` command instead.

A Hugging Face account or token is not normally required for a public model, but may be needed if the repository's access settings change or the host imposes limits.

### Alternative: quantize the official base model yourself

This is optional and not required for the provided 4-bit checkpoint. Quantization of the base model can require additional RAM and disk space and should be done only if you need to control the exact quantization recipe. Follow the relevant Transformers/bitsandbytes quantization documentation and Qwen model documentation.

Do not attempt to run the full-precision 27B model on a 36 GB card and assume it will fit. Use a supported quantized checkpoint.

## 9. Smoke test before processing real bids

Start with one known PDF, not a whole folder. This tests model loading, document processing, and the pipeline's output path.

```bash
python extract_constraints_qwen_local.py \
  --backend local \
  --local_model greghavens/Qwen3.8-27B-bnb-4bit \
  --pdf ./bid_pdfs/ONE_BID.pdf \
  --output_dir ./output_smoke \
  --llm_context_chars 60000 \
  --chunk_overlap_chars 3000 \
  --no_think_mode none
```

Replace `ONE_BID.pdf` with an actual file name. Do not type the placeholder literally.

Watch GPU memory in another terminal:

```bash
watch -n 1 nvidia-smi
```

The first run can take longer because the model must download and load. If the smoke test completes, do not immediately assume extraction quality is reliable. Inspect the output files before processing the full dataset.

Check that:

1. The model loads and inference completes without CUDA errors.
2. The expected JSON and CSV files are written.
3. The bid's main PDF and relevant linked documents appear in the document records where expected.
4. The extracted constraints are supported by the original PDF.
5. Any failed extraction, missing evidence, or unresolved specification is visible in the output or logs.

A successful smoke test confirms that the pipeline can run on the current setup. It does not establish that every linked document was retrieved or that every extracted constraint is correct.

## 10. Run the pipeline on a folder

After the smoke test and manual output review, run the full folder:

```bash
python extract_constraints_qwen_local.py \
  --backend local \
  --local_model greghavens/Qwen3.8-27B-bnb-4bit \
  --input_dir ./bid_pdfs \
  --output_dir ./output \
  --llm_context_chars 60000 \
  --chunk_overlap_chars 3000 \
  --concurrency 1 \
  --no_think_mode none
```

Local GPU inference is serialized; leave concurrency at `1`. Unlike the API setup, you do not need `OPENAI_API_KEY`, `OPENAI_BASE_URL`, or an OpenRouter model name.

### Suggested starting settings

| Setting | Starting value | Why |
|---|---:|---|
| `--llm_context_chars` | `60000` | Initial character budget for the LLM context; verify actual token length and memory usage |
| `--chunk_overlap_chars` | `3000` | Preserves some context at chunk boundaries |
| `--concurrency` | `1` | Avoids concurrent generations competing for VRAM |
| `--local_max_memory_gib` | `32` (script default) | Memory budget used for model placement; not a hard cap on total runtime VRAM |
| `--local_max_input_tokens` | `40000` (script default) | Input-token guardrail; not a guarantee that a prompt of this size fits in VRAM |
| `--validator_evidence_chars` | `70000` (script default) | Limits the targeted evidence text supplied to validation |
| `--validator_risk_threshold` | `3` (script default) | Controls which constraints are routed to Agent 2 versus automatically kept |

Confirm the defaults and available options with:

```bash
python extract_constraints_qwen_local.py --help
```

The validator batch-size options are legacy settings and should not be relied on to control actual validation batching or GPU memory. Agent 2 validation can still consume substantial context and generation memory.

`--llm_context_chars` counts characters, not tokens. The number of tokens varies with the document and tokenizer. Increase it gradually only after successful runs. Very long prompts and outputs can exhaust VRAM even when the quantized weights fit.

**Important:** `--local_max_memory_gib` provides a model-placement budget through Accelerate. It is not a hard cap on the GPU memory used by model weights, activations, the KV cache, or other runtime allocations. Leave headroom and measure actual usage.

## 11. Outputs

The output directory typically contains:

- `extraction.json`: extraction and validation records.
- `final_constraints.csv`: canonicalized constraints intended for downstream matching.
- `bids.csv`: bid-level records.
- `bid_constraints.csv`, `matching_constraints.csv`, and evidence/document CSVs, depending on the run.
- `failures.json`: per-bid failures, if any.
- Per-bid folders and linked documents, when downloaded.

Inspect logs and `failures.json`; do not assume that a process exit or a populated CSV means every constraint was validated successfully.

The pipeline may skip existing output JSON files to support resuming. The exact skip behavior depends on the output path and bid identifier. Use `--force` when you intentionally want to rerun bids in the same output directory. Prefer a new output directory when changing the model, prompts, or settings so outputs do not get mixed.

For research use, preserve the raw extraction records and source documents alongside the canonical CSVs. This makes it possible to audit individual constraints and distinguish extraction errors from matching errors.

## 12. Troubleshooting

### `torch.cuda.is_available()` is `False`

The installed PyTorch wheel may be CPU-only, the NVIDIA driver may be missing or incompatible, or the process may be running outside the GPU-enabled environment. Reinstall PyTorch using the official selector and verify `nvidia-smi`.

### CUDA out of memory

1. Stop the process and check whether another job is occupying the GPU with `nvidia-smi`.
2. Lower `--llm_context_chars` to `30000`–`40000`.
3. Keep `--concurrency 1`.
4. Lower `--local_max_memory_gib` to `28` or `30` if needed, understanding that this affects model placement and is not a hard runtime-memory cap.
5. Use a shorter single-PDF smoke test before retrying a full folder.
6. Check whether the failure occurs during model loading, extraction, or validation; the appropriate fix can differ by stage.

### Tesseract executable not found

Run:

```bash
tesseract --version
```

Install the OS-level executable if OCR fallback is needed. `pip install pytesseract` alone is not enough. On Windows, set `TESSERACT_CMD` to the executable path.

### Model download fails

Check disk space, internet access, Hugging Face status, and the model ID. If required by current access settings, authenticate using your own Hugging Face token. Do not place tokens in source code or commit them to the repository.

### Transformers cannot load the model

Check the model card for the current recommended Transformers version and model class. Upgrade `transformers`, `accelerate`, and `bitsandbytes` in the virtual environment if needed. Avoid mixing incompatible CUDA/PyTorch wheels or installing random nightly packages without a specific compatibility reason.

The script's model-loading path uses Transformers classes and does not enable `trust_remote_code` by default. Confirm the checkpoint's architecture and documented loading requirements before changing this behavior.

### Output is missing or incomplete

Check the console log, `failures.json`, and the per-bid JSON. Confirm that the expected PDF was discovered, the correct output directory was used, and the run was not skipped because an output already existed. Use a clean output directory or `--force` when intentionally rerunning.

### Extraction runs but constraints look wrong

Successful inference is not proof of accurate extraction. Compare a sample of constraints against the original bid documents. Pay particular attention to numerical values, units, mandatory versus optional requirements, contradictory specifications, certificates, and constraints whose evidence anchors are missing.

If quality is inadequate, fix the extraction or validation prompts and rerun a small reviewed set before scaling. Preserve the original outputs so changes can be evaluated rather than judged from memory.

## 13. Security and reproducibility

- Pin exact package versions after the first successful GPU run if the environment will be shared or reused.
- Record the GPU model, driver version, PyTorch/CUDA wheel, Transformers version, model ID/revision, and CLI arguments.
- Treat downloaded bid documents as untrusted inputs and run the pipeline in a dedicated environment.
- Never commit API keys or Hugging Face tokens.
- Preserve raw extraction outputs alongside canonical CSVs so the matching dataset can be audited.
- Record any manual corrections separately from model-generated constraints, so downstream experiments can distinguish raw extraction from human-reviewed ground truth.

## 14. Known limitations

- The default checkpoint is a third-party quantized model, not an official Qwen quantization.
- Compatibility with the exact GPU, model revision, and package versions must be established by a smoke test.
- A successful run does not guarantee that every constraint is correct or that all relevant linked documents were retrieved.
- Automatic retention by the validator does not mean every constraint received independent validation.
- Missing evidence anchors, contradictory constraints, and ambiguous certificate fields require review before treating them as verified facts.
- Memory usage depends on the model, prompt length, output length, document content, and runtime state. The configured memory budget is not a hard cap on total GPU memory.

The recommended workflow is to validate the installation, smoke-test one bid, review the resulting constraints against the source, and only then scale to the full dataset.
