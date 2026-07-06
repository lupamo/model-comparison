"""
Multi-model comparison harness.
Sends the same prompt to several foundation models, times each call, estimates cost and appends a structured record to results.jsonl.

This file is intentionally framework-free, so as to understand what 'model call' actually is before LangChain/llamaIndex hides it

usage:
	python compare.py "Explains the CAP theorem in two simple sentences."
	python compare.py --file prompts.txt #run a batch of prompts
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

RESULTS_FILE = Path("results.jsonl")


#----------------------------------------------------------------------------------------
# Model adapters, each function takes a prompt string and returns a dict:
# { "text": str, "input_tokens": int, "output_tokens": int}
#----------------------------------------------------------------------------------------

def call_claude(prompt: str, model:str = "claude-sonnet-4.6") -> dict:
	import anthropic
	client = anthropic.Anthropic()
	resp = client.message.create(
		model=model,
		max_tokens=1024,
		message={"role": "user", "content": prompt}
	)
	text = "".join(block.text for block in resp.content if block.type == "text")
	return {
		"text": text,
		"input_tokens": resp.usage.prompt_tokens,
		"output_tokens": resp.usage.completion_tokens
	}


def call_openai(prompt: str, model: str = "gpt-4o-mini") -> dict:
	from openai import OpenAI
	client = OpenAI()
	resp = client.chat.completions.create(
		model=model,
		messages=[{"role": "user", "content": prompt}],
	)
	return {
		"text": resp.choices[0].message.content,
		"input_tokens": resp.usage.prompt_tokens,
		"output_tokens": resp.usage.completion_tokens
	}

def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> dict:
    from groq import Groq
    client = Groq()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "text": resp.choices[0].message.content,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }

#----------------------------------------------------------------------------------------
# Rough $ per 1M tokens (input, output). Update these as pricing changes —
# the point isn't precision, it's building the instinct to always ask
# "what did that call cost?"
#----------------------------------------------------------------------------------------

PRICING = {
	"claude": (3.00, 15.00),
	"openai": (0.15, 0.60),
	"groq": (0.59, 0.79)
}

MODELS = {
	"claude":  call_claude,
	"openai":  call_openai,
	"groq":    call_groq,
}


def estimate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
	"""
	Estimate the cost of a model call in USD.
	"""
	in_price, out_price = PRICING[provider]
	return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price

def run_prompt(prompt: str, providers: list[str]) -> list[dict]:
    records = []
    for provider in providers:
        fn = MODELS[provider]
        start = time.perf_counter()
        try:
            result = fn(prompt)
            error = None
        except Exception as e:
            result = {"text": None, "input_tokens": 0, "output_tokens": 0}
            error = str(e)
        latency = time.perf_counter() - start
 
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "prompt": prompt,
            "output": result["text"],
            "latency_seconds": round(latency, 3),
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "estimated_cost_usd": round(
                estimate_cost(provider, result["input_tokens"], result["output_tokens"]), 6
            ),
            "error": error,
        }
        records.append(record)
        print_record(record)
    return records

def print_record(record: dict) -> None:
    print(f"\n--- {record['provider']} ---")
    if record["error"]:
        print(f"ERROR: {record['error']}")
        return
    print(record["output"])
    print(
        f"[{record['latency_seconds']}s | "
        f"{record['input_tokens']}in/{record['output_tokens']}out tokens | "
        f"${record['estimated_cost_usd']:.6f}]"
    )

def append_results(records: list[dict]) -> None:
    with RESULTS_FILE.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
def main():
    parser = argparse.ArgumentParser(description="Compare foundation models on the same prompt(s).")
    parser.add_argument("prompt", nargs="?", help="A single prompt to run.")
    parser.add_argument("--file", help="Path to a text file with one prompt per line.")
    parser.add_argument(
        "--providers",
        default="claude,openai,groq",
        help="Comma-separated list of providers to query (default: all).",
    )
    args = parser.parse_args()
 
    providers = [p.strip() for p in args.providers.split(",") if p.strip() in MODELS]
    if not providers:
        raise SystemExit(f"No valid providers. Choose from: {list(MODELS)}")
 
    if args.file:
        prompts = [line.strip() for line in Path(args.file).read_text().splitlines() if line.strip()]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        raise SystemExit("Provide a prompt as an argument or use --file prompts.txt")
 
    all_records = []
    for prompt in prompts:
        print(f"\n=== PROMPT: {prompt} ===")
        all_records.extend(run_prompt(prompt, providers))
 
    append_results(all_records)
    print(f"\nSaved {len(all_records)} records to {RESULTS_FILE.resolve()}")
 
 
if __name__ == "__main__":
    main()
 
