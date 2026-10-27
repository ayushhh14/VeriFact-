# AI Hallucination Detector & Fact Verification System

A local research-assistant prototype that checks claims in an AI-generated response against uploaded research PDFs.

## Features
- Extract text from PDFs and preserve page numbers.
- Split documents into overlapping chunks.
- Retrieve evidence using BM25 keyword search and dense embeddings.
- Use an NLI model to estimate whether evidence supports, contradicts, or is neutral toward each claim.
- Display evidence snippets and source-page references.
- Export a claim-level verification report as CSV.

## Important limitation
This is a decision-support prototype, not a guarantee of truth. NLI scores are model outputs, not calibrated probabilities that a claim is true. "Unverified" means the available retrieved evidence did not establish support; it does not prove the claim false. Citation checks in this first version verify references to uploaded source documents/page evidence, not the existence or authenticity of every external DOI/URL.

## Setup (macOS)
```bash
python3 --version
mkdir ai_hallucination_detector
cd ai_hallucination_detector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads embedding and NLI models from Hugging Face, so an internet connection is required initially. Model downloads can be several hundred MB. Use a Python version supported by the installed PyTorch wheels (Python 3.11 is a safe choice if your current Python version has package compatibility issues).

## Current scope
- Upload one or more text-based PDFs.
- Paste an AI-generated response.
- Extract simple sentence-level claims.
- Retrieve evidence from the uploaded PDFs.
- Show per-claim NLI verdicts and export a CSV report.


