#!/usr/bin/env python3
"""
Quick test: Segmentation only (Step 1 of v4 approach)
=====================================================
Tests whether Opus 4.6 can consistently segment a system prompt
into non-overlapping semantic units.

Usage:
  export OPENROUTER_API_KEY="sk-or-..."
  python test_segmentation.py                          # default: Poke_p1.txt
  python test_segmentation.py --index 1                # by index
  python test_segmentation.py --filename gpt4o_12102024.md
  python test_segmentation.py --index 1 --run 3        # run 3 times to check consistency
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_FILE = Path(__file__).parent / "audit_prompts.json"
OUTPUT_DIR = Path(__file__).parent / "test_segmentation_output"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

MODEL = "anthropic/claude-opus-4.6"
MAX_TOKENS = 16000
REASONING_EFFORT = "low"  # segmentation is relatively simple

API_TIMEOUT = 600
MAX_RETRIES = 3
RETRY_DELAY = 10

# ─── Segmentation Prompt ──────────────────────────────────────────────────────
SEGMENTATION_PROMPT = """You are a document segmentation assistant for AI system prompt auditing.

Your task: Divide the document below into non-overlapping, contiguous semantic units that together cover the entire text.

Segmentation rules:
1. Each segment should be a self-contained semantic unit — typically 1–3 sentences expressing one coherent idea, instruction, or rule.
2. Segments must NOT overlap — every character belongs to exactly one segment.
3. Together, all segments must cover the entire document text from start to end (no gaps).
4. The "text" field must be an EXACT verbatim copy from the document — do not alter, add, or remove any characters (including whitespace, newlines, punctuation).
5. Target length: 50–500 characters per segment. Shorter is OK for standalone sentences; longer is OK for tightly coupled multi-sentence blocks.
6. Section headers should be grouped with their immediately following content when they form one semantic unit.
7. Never split mid-sentence.
8. Whitespace-only regions (blank lines between sections) should be attached to the adjacent segment, not standalone.
9. Code blocks, tool definitions, and structured data should be kept as single segments if they express one idea (even if long).

--- DOCUMENT START ---
{content}
--- DOCUMENT END ---

Return ONLY a JSON array with no markdown fences or extra text:
[
  {{"id": "S01", "text": "exact verbatim text from document"}},
  {{"id": "S02", "text": "exact verbatim text from document"}},
  ...
]"""


# ─── API ──────────────────────────────────────────────────────────────────────
def call_openrouter(prompt_text: str) -> dict:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set!")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/promptauditing",
    }
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt_text}],
        "reasoning": {"effort": REASONING_EFFORT},
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(OPENROUTER_BASE_URL, headers=headers,
                                 json=payload, timeout=API_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.HTTPError, requests.exceptions.Timeout) as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"    ⏳ Retry in {wait}s... ({e})")
                time.sleep(wait)
            else:
                raise


def extract_response(resp_json: dict) -> tuple:
    choice = resp_json["choices"][0]["message"]
    output = choice.get("content", "") or ""
    usage = resp_json.get("usage", {})
    return output, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def parse_json_array(text: str) -> list:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    print("    ⚠️  Failed to parse JSON array")
    return []


def _normalize_unicode(s: str) -> str:
    return (s
        .replace('\u2018', "'").replace('\u2019', "'")
        .replace('\u201c', '"').replace('\u201d', '"')
        .replace('\u2013', '-').replace('\u2014', '-')
        .replace('\u00a0', ' ').replace('\u200b', ''))


# ─── Segmentation + Validation ────────────────────────────────────────────────
def run_segmentation(content: str, run_id: int = 1) -> dict:
    """Run one segmentation call and validate results."""
    prompt = SEGMENTATION_PROMPT.replace("{content}", content)

    print(f"\n  🔍 Run #{run_id}: Calling {MODEL}...")
    t0 = time.time()
    resp_json = call_openrouter(prompt)
    elapsed = time.time() - t0
    output, in_tok, out_tok = extract_response(resp_json)
    segments_raw = parse_json_array(output)
    print(f"  ⏱  {elapsed:.1f}s | {len(segments_raw)} segments | Tokens: {in_tok} in, {out_tok} out")

    # Validate: resolve offsets
    content_norm = _normalize_unicode(content)
    segments = []
    cursor = 0  # expected start position

    for seg in segments_raw:
        text = seg.get("text", "")
        if not text:
            continue

        # Try to find exact match
        idx = content.find(text, max(0, cursor - 50))
        if idx < 0:
            text_norm = _normalize_unicode(text)
            idx = content_norm.find(text_norm, max(0, cursor - 50))

        if idx >= 0:
            segments.append({
                "id": seg["id"],
                "text": content[idx:idx + len(text)],
                "start": idx,
                "end": idx + len(text),
                "found": True,
            })
            cursor = idx + len(text)
        else:
            segments.append({
                "id": seg["id"],
                "text": text,
                "start": -1,
                "end": -1,
                "found": False,
            })

    # Stats
    found = [s for s in segments if s["found"]]
    not_found = [s for s in segments if not s["found"]]
    total_chars = len(content)
    covered_chars = sum(s["end"] - s["start"] for s in found)

    # Check for overlaps
    overlaps = 0
    found_sorted = sorted(found, key=lambda s: s["start"])
    for i in range(1, len(found_sorted)):
        if found_sorted[i]["start"] < found_sorted[i-1]["end"]:
            overlaps += 1

    # Check for gaps
    gaps = 0
    gap_chars = 0
    for i in range(1, len(found_sorted)):
        gap = found_sorted[i]["start"] - found_sorted[i-1]["end"]
        if gap > 0:
            gap_text = content[found_sorted[i-1]["end"]:found_sorted[i]["start"]]
            if gap_text.strip():  # non-whitespace gap
                gaps += 1
                gap_chars += gap

    result = {
        "run_id": run_id,
        "time_seconds": round(elapsed, 1),
        "tokens": {"input": in_tok, "output": out_tok},
        "total_segments": len(segments_raw),
        "found": len(found),
        "not_found": len(not_found),
        "coverage_pct": round(covered_chars / total_chars * 100, 1),
        "overlaps": overlaps,
        "gaps_with_content": gaps,
        "gap_chars": gap_chars,
        "segments": segments,
        "segment_lengths": [s["end"] - s["start"] for s in found],
    }

    # Print summary
    print(f"  ✅ Found: {len(found)}/{len(segments_raw)} segments")
    if not_found:
        for nf in not_found:
            print(f"  ⚠️  Not found: \"{nf['text'][:60]}...\"")
    print(f"  📊 Coverage: {result['coverage_pct']}% | Overlaps: {overlaps} | Gaps: {gaps}")

    lengths = result["segment_lengths"]
    if lengths:
        avg = sum(lengths) / len(lengths)
        print(f"  📏 Segment lengths: min={min(lengths)}, avg={avg:.0f}, max={max(lengths)}, median={sorted(lengths)[len(lengths)//2]}")

    return result


def compare_runs(results: list, content: str):
    """Compare multiple segmentation runs for consistency."""
    print(f"\n{'='*70}")
    print(f"📊 CONSISTENCY ANALYSIS ({len(results)} runs)")
    print(f"{'='*70}")

    # Basic stats
    print(f"\n  {'Run':>4} {'Segs':>5} {'Found':>5} {'Cover%':>7} {'Overlaps':>8} {'Gaps':>5} {'Time':>6}")
    print(f"  {'─'*50}")
    for r in results:
        print(f"  #{r['run_id']:>3} {r['total_segments']:>5} {r['found']:>5} "
              f"{r['coverage_pct']:>6.1f}% {r['overlaps']:>8} {r['gaps_with_content']:>5} "
              f"{r['time_seconds']:>5.1f}s")

    # Compare segment boundaries across runs
    if len(results) > 1:
        print(f"\n  Boundary comparison:")
        all_boundaries = []
        for r in results:
            boundaries = set()
            for s in r["segments"]:
                if s["found"]:
                    boundaries.add(s["start"])
                    boundaries.add(s["end"])
            all_boundaries.append(boundaries)

        # Find boundaries that appear in all runs vs some runs
        all_set = all_boundaries[0]
        for b in all_boundaries[1:]:
            all_set = all_set & b
        any_set = set()
        for b in all_boundaries:
            any_set = any_set | b

        consistent = len(all_set)
        total_unique = len(any_set)
        print(f"  Consistent boundaries (in ALL runs): {consistent}")
        print(f"  Total unique boundaries: {total_unique}")
        if total_unique > 0:
            print(f"  Consistency rate: {consistent/total_unique*100:.1f}%")

        # Show inconsistent boundaries
        inconsistent = any_set - all_set
        if inconsistent and len(inconsistent) <= 20:
            print(f"\n  Inconsistent boundaries (positions):")
            for pos in sorted(inconsistent):
                context = content[max(0,pos-20):pos+20].replace('\n', '\\n')
                which_runs = [i+1 for i, b in enumerate(all_boundaries) if pos in b]
                print(f"    pos {pos}: in runs {which_runs} | \"...{context}...\"")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test segmentation with Opus 4.6")
    parser.add_argument("--index", type=int, default=None, help="Prompt index")
    parser.add_argument("--filename", type=str, default=None, help="Prompt filename")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for consistency check")
    parser.add_argument("--parallel", action="store_true", help="Run in parallel (for multiple runs)")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("❌ Please set OPENROUTER_API_KEY")
        sys.exit(1)

    # Load prompts
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)

    # Find target prompt
    target = None
    if args.index is not None:
        if 0 <= args.index < len(all_prompts):
            target = all_prompts[args.index]
        else:
            print(f"❌ Index {args.index} out of range (0–{len(all_prompts)-1})")
            sys.exit(1)
    elif args.filename:
        for p in all_prompts:
            if p["filename"] == args.filename:
                target = p
                break
        if not target:
            print(f"❌ Filename not found: {args.filename}")
            sys.exit(1)
    else:
        # Default: Poke_p1.txt
        for p in all_prompts:
            if p["filename"] == "Poke_p1.txt":
                target = p
                break
        if not target:
            target = all_prompts[0]

    content = target["content"]
    company = target["company"]
    filename = target["filename"]
    print(f"{'='*70}")
    print(f"📐 Segmentation Test: {company} / {filename}")
    print(f"   Size: {len(content)/1024:.1f} KB | Model: {MODEL}")
    print(f"   Runs: {args.runs} | Parallel: {args.parallel}")
    print(f"{'='*70}")

    # Run segmentation
    results = []
    if args.parallel and args.runs > 1:
        with ThreadPoolExecutor(max_workers=min(args.runs, 4)) as executor:
            futures = {executor.submit(run_segmentation, content, i+1): i+1
                       for i in range(args.runs)}
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda r: r["run_id"])
    else:
        for i in range(args.runs):
            results.append(run_segmentation(content, i + 1))
            if i < args.runs - 1:
                time.sleep(2)  # rate limiting

    # Compare if multiple runs
    if args.runs > 1:
        compare_runs(results, content)

    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    safe_name = f"{company}__{filename}".replace("/", "_").replace(" ", "_")
    out_path = OUTPUT_DIR / f"seg_test_{safe_name}_{args.runs}runs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "prompt": {"company": company, "filename": filename, "size_bytes": len(content)},
            "model": MODEL,
            "runs": [{k: v for k, v in r.items() if k != "segments"} for r in results],
            "segments_run1": results[0]["segments"],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved to: {out_path}")


if __name__ == "__main__":
    main()
