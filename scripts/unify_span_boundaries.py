#!/usr/bin/env python3
"""
Unify span boundaries across same-company same-paragraph prompts.

For each shared paragraph between versions:
1. Pick the version with the most spans as the "reference" cut
2. For other versions, re-cut their spans to match the reference boundaries
3. Keep each version's own dimension + score labels
4. If a reference span has no matching label in a target version, mark it as missing

Usage:
    python scripts/unify_span_boundaries.py --dry-run
    python scripts/unify_span_boundaries.py
"""

import json
import argparse
import copy
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path(__file__).parent.parent
MERGED = ROOT / "annotation_tool_89" / "analysis" / "merged_all_annotations.json"
AUDIT = ROOT / "data" / "audit_prompts.json"


def load_prompt_texts():
    with AUDIT.open() as f:
        audit = json.load(f)
    texts = {}
    for p in audit:
        if not p.get("content"):
            continue
        pid = f"{p['company']}__{p['filename']}"
        texts[pid] = p["content"]
        texts[pid.replace(" ", "_")] = p["content"]
    return texts


def get_paragraphs(text, min_len=50):
    return [c.strip() for c in text.split("\n\n") if len(c.strip()) >= min_len]


def get_spans_covering(spans, para_start, para_end):
    """Get spans that overlap with the paragraph region."""
    return [
        s for s in spans
        if s.get("start", 0) < para_end and s.get("end", 0) > para_start
    ]


def normalize_spans(spans, base_offset):
    """Convert absolute positions to relative (within paragraph)."""
    return [
        {
            "rel_start": s["start"] - base_offset,
            "rel_end": s["end"] - base_offset,
            "dimension": s["dimension"],
            "score": s.get("score", 0),
            "note": s.get("note", ""),
            "source": s.get("source", "llm"),
            "text": s.get("text", ""),
        }
        for s in spans
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompt_texts = load_prompt_texts()

    with MERGED.open() as f:
        data = json.load(f)
    original = copy.deepcopy(data)

    by_company = defaultdict(list)
    for pid, p in data["prompts"].items():
        by_company[p["company"]].append(pid)

    total_paras_unified = 0
    total_spans_adjusted = 0
    log = []

    for comp, pids in sorted(by_company.items()):
        if len(pids) < 2:
            continue

        pids_with_text = [pid for pid in pids if pid in prompt_texts]
        if len(pids_with_text) < 2:
            continue

        # Find all shared paragraphs across ALL versions
        para_to_pids = defaultdict(set)
        for pid in pids_with_text:
            for para in get_paragraphs(prompt_texts[pid]):
                para_to_pids[para].add(pid)

        for para, para_pids in para_to_pids.items():
            if len(para_pids) < 2:
                continue

            # For each version, get its spans covering this paragraph
            version_spans = {}
            for pid in para_pids:
                pt = prompt_texts[pid]
                para_start = pt.find(para)
                if para_start < 0:
                    continue
                para_end = para_start + len(para)
                covering = get_spans_covering(data["prompts"][pid]["kept_spans"], para_start, para_end)
                norm = normalize_spans(covering, para_start)
                version_spans[pid] = {
                    "para_start": para_start,
                    "covering": covering,
                    "normalized": norm,
                }

            if len(version_spans) < 2:
                continue

            # Check if all versions already have the same boundaries
            boundary_sets = []
            for pid, vs in version_spans.items():
                bounds = frozenset((n["rel_start"], n["rel_end"]) for n in vs["normalized"])
                boundary_sets.append(bounds)

            if len(set(boundary_sets)) <= 1:
                continue  # boundaries already consistent

            # Pick reference: version with most unique span boundaries
            ref_pid = max(version_spans.keys(),
                key=lambda pid: len(set((n["rel_start"], n["rel_end"]) for n in version_spans[pid]["normalized"])))
            ref = version_spans[ref_pid]
            ref_boundaries = sorted(set((n["rel_start"], n["rel_end"]) for n in ref["normalized"]))

            if not ref_boundaries:
                continue

            total_paras_unified += 1
            ref_label = data["prompts"][ref_pid]["product_label"]
            log.append(f"\n  [{comp}] Paragraph: \"{para[:80]}...\"")
            log.append(f"    Reference: {ref_label} ({len(ref_boundaries)} span boundaries)")

            # For each other version, re-align to reference boundaries
            for pid, vs in version_spans.items():
                if pid == ref_pid:
                    continue

                target_label = data["prompts"][pid]["product_label"]
                para_start = vs["para_start"]
                prompt = data["prompts"][pid]

                # Remove old spans covering this paragraph
                old_covering = vs["covering"]
                old_ids = set(id(s) for s in old_covering)
                # Collect labels from old spans (by relative position overlap)
                old_norm = vs["normalized"]

                # For each reference boundary, find which labels the target version had
                new_spans = []
                for ref_rs, ref_re in ref_boundaries:
                    abs_start = para_start + ref_rs
                    abs_end = para_start + ref_re
                    ref_text = para[max(0, ref_rs):ref_re] if ref_re <= len(para) else para[max(0, ref_rs):]

                    # Find labels from old spans that overlap with this reference boundary
                    labels_for_this_boundary = []
                    for on in old_norm:
                        # Check overlap between old span and reference boundary
                        overlap_start = max(on["rel_start"], ref_rs)
                        overlap_end = min(on["rel_end"], ref_re)
                        if overlap_start < overlap_end:
                            labels_for_this_boundary.append({
                                "dimension": on["dimension"],
                                "score": on["score"],
                                "note": on["note"],
                                "source": on["source"],
                            })

                    if not labels_for_this_boundary:
                        continue

                    # Deduplicate labels
                    seen_labels = set()
                    for lb in labels_for_this_boundary:
                        key = (lb["dimension"], lb["score"])
                        if key in seen_labels:
                            continue
                        seen_labels.add(key)
                        new_spans.append({
                            "start": abs_start,
                            "end": abs_end,
                            "text": ref_text,
                            "dimension": lb["dimension"],
                            "score": lb["score"],
                            "note": lb["note"],
                            "source": lb["source"],
                            "reviewed": True,
                        })

                # Check if anything actually changed
                old_set = set((n["rel_start"], n["rel_end"], n["dimension"], n["score"]) for n in old_norm)
                new_set = set((s["start"] - para_start, s["end"] - para_start, s["dimension"], s["score"]) for s in new_spans)

                if old_set == new_set:
                    continue

                # Remove old covering spans from kept_spans
                prompt["kept_spans"] = [s for s in prompt["kept_spans"] if id(s) not in old_ids]
                # Add new re-aligned spans
                prompt["kept_spans"].extend(new_spans)
                # Re-sort
                prompt["kept_spans"].sort(key=lambda s: (s.get("start", 0), s.get("end", 0)))

                adjusted = len(new_set - old_set) + len(old_set - new_set)
                total_spans_adjusted += adjusted
                log.append(f"    Adjusted: {target_label} ({len(old_set)} → {len(new_set)} span-labels)")

    # Rebuild counts
    total_kept = 0
    for pid, prompt in data["prompts"].items():
        prompt["kept_count"] = len(prompt["kept_spans"])
        prompt["human_added_count"] = sum(1 for s in prompt["kept_spans"] if s.get("source") == "human")
        summary = {}
        for s in prompt["kept_spans"]:
            dim = s.get("dimension", "?")
            if dim not in summary:
                summary[dim] = {"positive": 0, "negative": 0, "total": 0}
            summary[dim]["total"] += 1
            if s.get("score", 0) > 0:
                summary[dim]["positive"] += 1
            elif s.get("score", 0) < 0:
                summary[dim]["negative"] += 1
        prompt["dimension_summary"] = summary
        total_kept += len(prompt["kept_spans"])
    data["metadata"]["total_kept_spans"] = total_kept

    old_kept = original["metadata"]["total_kept_spans"]
    new_kept = data["metadata"]["total_kept_spans"]

    print(f"{'='*60}")
    print(f"SPAN BOUNDARY UNIFICATION")
    print(f"{'='*60}")
    print(f"  Paragraphs unified: {total_paras_unified}")
    print(f"  Span-labels adjusted: {total_spans_adjusted}")
    print(f"  Total kept spans: {old_kept} → {new_kept} (delta: {new_kept - old_kept:+d})")
    print()
    for line in log:
        print(line)

    if args.dry_run:
        print(f"\n[DRY RUN] No files written.")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = MERGED.with_name(f"merged_all_annotations_pre_unify_{ts}.json")
        with backup.open("w", encoding="utf-8") as f:
            json.dump(original, f, indent=2, ensure_ascii=False)
        print(f"\n  Backup: {backup}")
        with MERGED.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Written: {MERGED}")


if __name__ == "__main__":
    main()
