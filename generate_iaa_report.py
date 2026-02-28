"""Generate an Inter-Annotator Agreement (IAA) report from training tool outputs.

Usage:
    python generate_iaa_report.py                          # auto-detect latest files
    python generate_iaa_report.py -o iaa_report.txt        # custom output path
    python generate_iaa_report.py --files f1.json f2.json  # specify files explicitly
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path

DIMENSION_NAMES = {
    "D1": "Identity Transparency",
    "D2": "Knowledge Transparency",
    "D3": "Information Accuracy & Source Integrity",
    "D4": "Privacy & Data Protection",
    "D5": "Tool/Action Safety",
    "D6": "Manipulation Prevention",
    "D7": "Unsafe Request Handling",
    "D8": "Harm Prevention & User Safety",
    "D9": "Fairness, Inclusion & Neutrality",
    "D10": "Miscellaneous",
}

TRAINING_DIR = Path(__file__).parent / "training_tool" / "outputs"


def dim_label(dim: str) -> str:
    name = DIMENSION_NAMES.get(dim, "")
    return f"{dim} {name}" if name else dim


def load_annotations(filepath: Path) -> dict:
    with open(filepath) as f:
        return json.load(f)


def find_latest_training_files() -> list[Path]:
    """Pick the latest file per reviewer (by filename timestamp)."""
    by_reviewer: dict[str, Path] = {}
    for p in sorted(TRAINING_DIR.glob("training_*.json")):
        match = re.match(r"training_(.+?)_\d{8}_\d{6}\.json", p.name)
        if not match:
            continue
        reviewer = match.group(1)
        by_reviewer[reviewer] = p
    return list(by_reviewer.values())


def span_key(prompt_key: str, span: dict) -> tuple:
    return (prompt_key, span["dimension"], span["start"], span["end"])


def is_rejected(span: dict) -> bool:
    return span.get("rejected", False)


def build_llm_span_decisions(data: dict) -> dict[tuple, bool]:
    """Return {span_key: accepted} for every LLM span in this annotator's file."""
    decisions: dict[tuple, bool] = {}
    for prompt_key, prompt_data in data["annotations"].items():
        for span in prompt_data["spans"]:
            if span.get("source") != "llm":
                continue
            key = span_key(prompt_key, span)
            decisions[key] = not is_rejected(span)
    return decisions


def prompt_display_name(prompt_key: str, prompt_data: dict) -> str:
    company = prompt_data.get("company", "")
    filename = prompt_data.get("filename", prompt_key)
    return f"{company} / {filename}" if company else filename


def generate_report(files: list[Path], output: Path | None = None) -> str:
    annotators_data = []
    for f in files:
        d = load_annotations(f)
        annotators_data.append(d)

    names = [d["metadata"]["reviewer"] for d in annotators_data]
    n = len(names)

    all_decisions = [build_llm_span_decisions(d) for d in annotators_data]
    all_keys = set()
    for dec in all_decisions:
        all_keys |= dec.keys()
    all_keys = sorted(all_keys)
    total_llm_spans = len(all_keys)

    # --- Pairwise agreement ---
    pair_results = []
    for i, j in combinations(range(n), 2):
        agree = 0
        compared = 0
        for key in all_keys:
            if key in all_decisions[i] and key in all_decisions[j]:
                compared += 1
                if all_decisions[i][key] == all_decisions[j][key]:
                    agree += 1
        pair_results.append((i, j, agree, compared))

    # --- All-agree ---
    all_agree = 0
    for key in all_keys:
        votes = [all_decisions[a].get(key) for a in range(n)]
        if all(v is not None for v in votes) and len(set(votes)) == 1:
            all_agree += 1

    # --- Kappa (pairwise) ---
    def cohens_kappa(dec_a, dec_b, keys):
        tp = tn = fp = fn = 0
        for k in keys:
            a, b = dec_a.get(k), dec_b.get(k)
            if a is None or b is None:
                continue
            if a and b:
                tp += 1
            elif not a and not b:
                tn += 1
            elif a and not b:
                fp += 1
            else:
                fn += 1
        total = tp + tn + fp + fn
        if total == 0:
            return None
        po = (tp + tn) / total
        pa = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
        if pa == 1:
            return None
        return (po - pa) / (1 - pa)

    kappas = []
    for i, j in combinations(range(n), 2):
        k = cohens_kappa(all_decisions[i], all_decisions[j], all_keys)
        kappas.append((i, j, k))

    # --- Disagreements ---
    disagreements = []
    for key in all_keys:
        votes = {}
        for a in range(n):
            v = all_decisions[a].get(key)
            if v is not None:
                votes[a] = v
        if len(set(votes.values())) > 1:
            prompt_key, dim, start, end = key
            span_info = None
            for d in annotators_data:
                if prompt_key in d["annotations"]:
                    for s in d["annotations"][prompt_key]["spans"]:
                        if s["source"] == "llm" and s["start"] == start and s["end"] == end:
                            span_info = s
                            prompt_display = prompt_display_name(prompt_key, d["annotations"][prompt_key])
                            break
                if span_info:
                    break
            disagreements.append((prompt_display, dim, span_info, votes))

    # --- Human-added spans ---
    human_spans: dict[str, list] = defaultdict(list)
    for idx, d in enumerate(annotators_data):
        reviewer = names[idx]
        for prompt_key, prompt_data in d["annotations"].items():
            display = prompt_display_name(prompt_key, prompt_data)
            for span in prompt_data["spans"]:
                if span.get("source") == "human":
                    human_spans[reviewer].append((display, span))

    # --- Accept rate ---
    total_accept = sum(sum(1 for v in dec.values() if v) for dec in all_decisions)
    total_votes = sum(len(dec) for dec in all_decisions)
    accept_rate = total_accept / total_votes * 100 if total_votes else 0

    # --- Build report ---
    lines = []
    w = lines.append

    w(f"IAA Report — Training Tool ({n} Annotators)")
    w("=" * 42)
    w(f"Generated: {date.today()}")
    w(f"Annotators: {', '.join(names)}")
    w(f"Total LLM pre-annotated spans: {total_llm_spans}")
    w("")
    w("=" * 50)
    w("PAIRWISE AGREEMENT (Accept/Reject on LLM spans)")
    w("=" * 50)
    w("")

    for i, j, agree, compared in pair_results:
        safe_name_i = names[i].replace(" ", "_")
        safe_name_j = names[j].replace(" ", "_")
        pct = agree / compared * 100 if compared else 0
        w(f"  {safe_name_i} vs {safe_name_j}:   {agree}/{compared} = {pct:.1f}%")
    w("")

    all_compared = sum(1 for key in all_keys if all(all_decisions[a].get(key) is not None for a in range(n)))
    all_pct = all_agree / all_compared * 100 if all_compared else 0
    w(f"  All {n} agree: {all_agree}/{all_compared} = {all_pct:.1f}%")
    w("")

    kappa_strs = []
    for i, j, k in kappas:
        if k is not None:
            kappa_strs.append(f"{k:.3f}")
    if kappa_strs:
        unique_kappas = set(kappa_strs)
        if len(unique_kappas) == 1:
            w(f"Note: Cohen's Kappa = {kappa_strs[0]} for all pairs (Kappa Paradox due to")
        else:
            parts = ", ".join(kappa_strs)
            w(f"Note: Cohen's Kappa = {parts} per pair (Kappa Paradox due to")
        w(f"~{accept_rate:.0f}% accept rate; percent agreement is the more appropriate metric here).")
    w("")

    w("=" * 50)
    w(f"DISAGREEMENTS ({len(disagreements)} / {total_llm_spans} spans)")
    w("=" * 50)
    w("")

    for idx, (display, dim, span_info, votes) in enumerate(disagreements, 1):
        score_str = f"+{span_info['score']}" if span_info["score"] > 0 else str(span_info["score"])
        w(f"{idx}. [{display}] {dim_label(dim)}, score={score_str}")

        text = span_info["text"]
        text_lines = text.split("\n")
        if len(text_lines) == 1:
            w(f'   Text: "{text}"')
        else:
            w(f'   Text: "{text_lines[0]}')
            for tl in text_lines[1:]:
                w(f'         {tl}')
            w(f'         "')

        for a in range(n):
            safe_name = names[a].replace(" ", "_")
            v = votes.get(a)
            if v is None:
                label = "N/A"
            elif v:
                label = "Accept"
            else:
                label = "REJECT"
            w(f"   {safe_name + ':':14s} {label}")
        w("")

    w("=" * 50)
    w("HUMAN-ADDED SPANS (extra annotations beyond LLM)")
    w("=" * 50)
    w("")

    for reviewer in names:
        spans = human_spans.get(reviewer, [])
        w(f"{reviewer} ({len(spans)} span{'s' if len(spans) != 1 else ''}):")
        if not spans:
            w("")
            continue
        for display, span in spans:
            score_str = f"+{span['score']}" if span["score"] > 0 else str(span["score"])
            text_preview = span["text"]
            if len(text_preview) > 120:
                text_preview = text_preview[:117] + "..."
            w(f'  - [{display}] {dim_label(span["dimension"])}, score={score_str}')
            w(f'    "{text_preview}"')
        w("")

    report = "\n".join(lines)

    if output:
        output.write_text(report)
        print(f"Report written to {output}")
    else:
        print(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="Generate IAA report from training outputs")
    parser.add_argument("-o", "--output", type=Path, default=Path("iaa_report.txt"),
                        help="Output file path (default: iaa_report.txt)")
    parser.add_argument("--files", nargs="+", type=Path,
                        help="Specific annotation files to compare (default: auto-detect latest per reviewer)")
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        files = find_latest_training_files()

    if len(files) < 2:
        print(f"Error: Need at least 2 annotator files, found {len(files)}")
        print(f"Looked in: {TRAINING_DIR}")
        return

    print(f"Found {len(files)} annotator files:")
    for f in files:
        print(f"  - {f.name}")
    print()

    generate_report(files, args.output)


if __name__ == "__main__":
    main()
