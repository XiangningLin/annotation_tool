#!/usr/bin/env python3
"""
Batch LLM Pre-Annotation — Process 20 prompts for human audit pilot
====================================================================
Uses the pre-segmentation approach from llm_preannotate_v2.py to process
a stratified sample of 20 prompts across 20 companies and multiple categories.

Features:
  - Resume support: skips already-processed prompts
  - Progress tracking with ETA
  - Per-prompt and total cost tracking
  - Individual output files + batch summary

Usage:
  export OPENROUTER_API_KEY="sk-or-..."
  python llm_batch_preannotate.py                # run all 20
  python llm_batch_preannotate.py --resume       # resume (skip completed)
  python llm_batch_preannotate.py --dry-run      # show plan without running
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from llm_preannotate_v2 import (
    DATA_FILE, OUTPUT_DIR, OPENROUTER_API_KEY,
    process_prompt, DIMENSIONS,
)

# ─── Pilot 20 Prompts ────────────────────────────────────────────────────────
# Stratified sample: 20 companies × diverse categories × range of sizes
PILOT_20_INDICES = [
    148,  # Poke / Poke_p1.txt (10.2KB) — Social AI
    93,   # OpenAI / gpt4o_12102024.md (8.0KB) — Chat / General
    1,    # Anthropic / 20240712-Claude3.5-Sonnet.md (5.8KB) — Chat / General
    2,    # Google / gemini-2.0-flash-thinking (6.9KB) — Chat / General
    3,    # Microsoft / copilot_website (14.3KB) — Code Assistant
    61,   # xAI / 20240821-Grok2.md (4.6KB) — Chat / General
    63,   # Meta / metaai_llama3-04182024.md (15.0KB) — Chat / General
    62,   # Cursor / CursorAgileModeSystemPrompt (6.8KB) — Code Editor / IDE
    9,    # Perplexity / 20240320-Perplexity.md (5.5KB) — Search / Research
    67,   # Devin / Devin_2.0.md (6.1KB) — Autonomous Agent
    11,   # Lovable / system.md (9.0KB) — Web Development
    17,   # DIA / Dia_DraftSkill.txt (8.9KB) — Writing Assistant
    42,   # Hume / 05052024-system-prompt.md (8.7KB) — Chat / General
    52,   # Venice / Venice.md (2.1KB) — Privacy / Uncensored
    31,   # Orchids / Decision-making prompt.txt (6.8KB) — Health / Wellness
    25,   # Cluely / Cluely.mkd (4.8KB) — Controversial
    29,   # Moonshot / Kimi_K2_Thinking.txt (1.0KB) — Chat / General
    84,   # DeepSeek / R1.md (1.4KB) — Chat / General
    18,   # Manus / Agent loop.txt (2.1KB) — Autonomous Agent
    50,   # Raycast / RaycastAI.md (6.9KB) — Productivity / DevOps
]


def get_output_path(prompt_data: dict) -> Path:
    """Get the output file path for a prompt."""
    safe_name = f"{prompt_data['company']}__{prompt_data['filename']}"
    safe_name = safe_name.replace("/", "_").replace(" ", "_")
    return OUTPUT_DIR / f"{safe_name}.json"


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


def main():
    parser = argparse.ArgumentParser(
        description="Batch LLM Pre-Annotation — 20 prompts for pilot study")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed prompts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without processing")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated indices to process (overrides default 20)")
    args = parser.parse_args()

    # Check API key
    if not OPENROUTER_API_KEY and not args.dry_run:
        print("❌ Please set the OPENROUTER_API_KEY environment variable:")
        print("   export OPENROUTER_API_KEY='sk-or-...'")
        sys.exit(1)

    # Load prompts
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)

    # Determine which indices to process
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]
    else:
        indices = PILOT_20_INDICES

    # Validate indices
    prompts_to_process = []
    for idx in indices:
        if 0 <= idx < len(all_prompts):
            prompts_to_process.append(all_prompts[idx])
        else:
            print(f"⚠️  Index {idx} out of range, skipping")

    # Check which are already done (for resume mode)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    already_done = set()
    if args.resume:
        for p in prompts_to_process:
            out_path = get_output_path(p)
            if out_path.exists():
                already_done.add(p["filename"])

    # ── Show Plan ──
    total_kb = sum(p["size_bytes"] / 1024 for p in prompts_to_process)
    to_process = [p for p in prompts_to_process if p["filename"] not in already_done]

    print(f"{'='*70}")
    print(f"📦 Batch LLM Pre-Annotation — Pilot Study")
    print(f"{'='*70}")
    print(f"  Total prompts:    {len(prompts_to_process)}")
    print(f"  Already done:     {len(already_done)}")
    print(f"  To process:       {len(to_process)}")
    print(f"  Total size:       {total_kb:.0f} KB")
    print(f"  Est. time:        {format_duration(len(to_process) * 600)}")
    print(f"  Est. cost:        ~${len(to_process) * 1.3:.0f}")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print()

    print(f"{'#':>3s}  {'Idx':>4s}  {'Company':15s}  {'Filename':42s}  {'Size':>7s}  {'Status'}")
    print(f"{'─'*90}")
    for i, p in enumerate(prompts_to_process):
        kb = p['size_bytes'] / 1024
        status = "✅ done" if p["filename"] in already_done else "⏳ pending"
        print(f"{i+1:3d}  {p.get('index', '?'):>4}  {p['company']:15s}  "
              f"{p['filename']:42s}  {kb:5.1f}KB  {status}")
    print()

    if args.dry_run:
        print("🔍 Dry run — no API calls will be made.")
        return

    if not to_process:
        print("✅ All prompts already processed!")
        return

    # ── Process ──
    batch_t0 = time.time()
    results_summary = []
    total_cost = 0
    processed = 0
    failed = 0

    for i, prompt_data in enumerate(to_process):
        prompt_num = i + 1
        total_num = len(to_process)
        filename = prompt_data["filename"]

        # Progress header
        elapsed_so_far = time.time() - batch_t0
        if processed > 0:
            avg_time = elapsed_so_far / processed
            eta = avg_time * (total_num - prompt_num + 1)
            eta_str = f" | ETA: {format_duration(eta)}"
        else:
            eta_str = ""

        print(f"\n{'▓'*70}")
        print(f"  [{prompt_num}/{total_num}] {prompt_data['company']} / {filename}"
              f"  ({prompt_data['size_bytes']/1024:.1f}KB){eta_str}")
        print(f"{'▓'*70}")

        try:
            result = process_prompt(prompt_data, verbose=True)

            # Save individual result
            out_path = get_output_path(prompt_data)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  💾 Saved: {out_path.name}")

            cost = result["metadata"]["cost_usd"]["total"]
            total_cost += cost
            processed += 1

            results_summary.append({
                "index": prompt_data.get("index", -1),
                "company": prompt_data["company"],
                "filename": filename,
                "size_bytes": prompt_data["size_bytes"],
                "num_segments": len(result["segments"]),
                "coverage_pct": result["metadata"]["coverage"]["non_ws_coverage_pct"],
                "cost_usd": cost,
                "time_seconds": result["metadata"]["timing"]["total_seconds"],
                "status": "success",
            })

        except Exception as e:
            print(f"\n  ❌ FAILED: {e}")
            failed += 1
            results_summary.append({
                "index": prompt_data.get("index", -1),
                "company": prompt_data["company"],
                "filename": filename,
                "size_bytes": prompt_data["size_bytes"],
                "status": "failed",
                "error": str(e),
            })
            # Continue to next prompt
            continue

        # Small delay between prompts to be nice to the API
        if prompt_num < total_num:
            time.sleep(2)

    # ── Batch Summary ──
    total_elapsed = time.time() - batch_t0
    print(f"\n{'='*70}")
    print(f"📊 BATCH SUMMARY")
    print(f"{'='*70}")
    print(f"  Processed:   {processed}/{len(to_process)}")
    print(f"  Failed:      {failed}")
    print(f"  Total time:  {format_duration(total_elapsed)}")
    print(f"  Total cost:  ${total_cost:.2f}")
    print()

    # Per-prompt summary table
    print(f"  {'Company':15s}  {'Filename':35s}  {'Segs':>5s}  {'Cov%':>5s}  {'Cost':>6s}  {'Time':>7s}  {'Status'}")
    print(f"  {'─'*95}")
    for r in results_summary:
        if r["status"] == "success":
            print(f"  {r['company']:15s}  {r['filename']:35s}  "
                  f"{r['num_segments']:5d}  {r['coverage_pct']:5.1f}  "
                  f"${r['cost_usd']:.2f}  {format_duration(r['time_seconds']):>7s}  ✅")
        else:
            print(f"  {r['company']:15s}  {r['filename']:35s}  "
                  f"{'—':>5s}  {'—':>5s}  {'—':>6s}  {'—':>7s}  ❌ {r.get('error','')[:30]}")

    # Save batch summary
    summary_path = OUTPUT_DIR / "batch_summary.json"
    summary = {
        "batch_info": {
            "total_prompts": len(to_process),
            "processed": processed,
            "failed": failed,
            "total_time_seconds": round(total_elapsed, 1),
            "total_cost_usd": round(total_cost, 4),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "prompts": results_summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Batch summary: {summary_path}")


if __name__ == "__main__":
    main()

