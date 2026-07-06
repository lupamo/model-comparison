# Model Comaparison Harnes
A Minimal, framework tool to compare foundation models on the same prompts
The goal isn't the tool itself, it's to build intuition for what actually happens when you "call a model": tokens in, tokens out, latency cost, and how much quality varies across providers for the exact same input.

## Setup

```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```

Set API keys as environment variables (get free/cheap tiers from each):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GROQ_API_KEY=gsk_...
```

You don't need to run all three. Run with just what you have:
```bash
python compare.py "Explain the CAP theorem in two sentences." --providers claude,groq
```
## Usage

Single prompt across all configured providers:
```bash
python compare.py "What are the tradeoffs of RAG vs fine-tuning?"
```

Batch mode — run a whole file of prompts (one per line):
```bash
python compare.py --file prompts.txt
```

Every run appends structured JSON records to `results.jsonl`. Nothing is
overwritten, so this file grows into a dataset over time.
