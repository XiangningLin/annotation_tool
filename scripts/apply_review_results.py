#!/usr/bin/env python3
"""
Apply review results back to merged_all_annotations.json.

Reads the exported review JSON (from the review tool) and:
1. For negative spans marked "disagree": removes the span from kept_spans
2. For cross-version conflicts with final_dim + final_score:
   finds ALL spans across ALL versions containing the shared text
   and unifies their dimension + score to the user's decision.

Usage:
    python scripts/apply_review_results.py review_results_2026-03-07.json
    python scripts/apply_review_results.py review_results_2026-03-07.json --dry-run
"""

import json
import argparse
import copy
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
MERGED = ROOT / "annotation_tool_89" / "analysis" / "merged_all_annotations.json"


def load_merged():
    with MERGED.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_merged(data, path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def apply_negative_reviews(merged, neg_reviews, log):
    """Remove spans marked 'disagree' from kept_spans."""
    removed = 0
    for nr in neg_reviews:
        if nr.get("review_status") != "disagree":
            continue

        pid = nr["prompt_id"]
        if pid not in merged["prompts"]:
            log.append(f"  WARN: prompt {pid} not found, skipping")
            continue

        prompt = merged["prompts"][pid]
        span_idx = nr.get("span_index_in_prompt")
        text = nr["text"]

        before = len(prompt["kept_spans"])
        prompt["kept_spans"] = [
            s for s in prompt["kept_spans"]
            if not (s["text"] == text and s["dimension"] == nr["dimension"] and s.get("score") == nr["score"])
        ]
        after = len(prompt["kept_spans"])
        diff = before - after
        if diff > 0:
            removed += diff
            prompt["kept_count"] = len(prompt["kept_spans"])
            log.append(f"  Removed {diff} span(s) from {pid}: [{nr['dimension']} score={nr['score']}] \"{text[:80]}\"")

    return removed


def apply_conflict_decisions(merged, conf_reviews, log):
    """
    For each conflict with final_dim + final_score:
    find all spans across all same-company prompts containing the shared text,
    and unify them to the decided label.
    """
    unified = 0
    changed_spans = 0

    for cr in conf_reviews:
        final_dim = cr.get("final_dim", "")
        final_score = cr.get("final_score", "")
        if not final_dim or final_score == "":
            continue

        final_score = int(final_score)
        company = cr["company"]
        overlap_text = cr.get("overlap_text", "")

        if not overlap_text:
            continue

        unified += 1
        log.append(f"\n  Unifying: [{company}] → {final_dim} score={'+1' if final_score > 0 else '-1'}")
        log.append(f"    Text: \"{overlap_text[:100]}\"")

        for pid, prompt in merged["prompts"].items():
            if prompt["company"] != company:
                continue

            for span in prompt["kept_spans"]:
                span_text = span["text"].strip()
                overlap_clean = overlap_text.strip()

                if overlap_clean not in span_text and span_text not in overlap_clean:
                    continue

                old_dim = span["dimension"]
                old_score = span.get("score")

                if old_dim == final_dim and old_score == final_score:
                    continue

                span["dimension"] = final_dim
                span["score"] = final_score
                changed_spans += 1
                log.append(f"    Changed [{pid}]: {old_dim}(s={old_score}) → {final_dim}(s={final_score})")

    return unified, changed_spans


def rebuild_summaries(merged):
    """Recalculate dimension_summary and counts for each prompt."""
    total_kept = 0
    total_neg = 0
    for pid, prompt in merged["prompts"].items():
        spans = prompt["kept_spans"]
        prompt["kept_count"] = len(spans)
        prompt["human_added_count"] = sum(1 for s in spans if s.get("source") == "human")

        summary = {}
        for s in spans:
            dim = s.get("dimension", "?")
            if dim not in summary:
                summary[dim] = {"positive": 0, "negative": 0, "total": 0}
            summary[dim]["total"] += 1
            if s.get("score", 0) > 0:
                summary[dim]["positive"] += 1
            elif s.get("score", 0) < 0:
                summary[dim]["negative"] += 1
        prompt["dimension_summary"] = summary
        total_kept += len(spans)

    merged["metadata"]["total_kept_spans"] = total_kept


def main():
    parser = argparse.ArgumentParser(description="Apply review results to merged annotations")
    parser.add_argument("review_file", help="Path to exported review_results JSON")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument("--output", help="Output path (default: overwrite merged file)")
    args = parser.parse_args()

    with open(args.review_file, "r", encoding="utf-8") as f:
        review_data = json.load(f)

    neg_reviews = review_data.get("negative_spans", [])
    conf_reviews = review_data.get("cross_version_conflicts", [])

    print(f"Loaded review results:")
    print(f"  Negative spans: {len(neg_reviews)}")
    print(f"    agree: {sum(1 for r in neg_reviews if r.get('review_status')=='agree')}")
    print(f"    disagree: {sum(1 for r in neg_reviews if r.get('review_status')=='disagree')}")
    print(f"    discuss: {sum(1 for r in neg_reviews if r.get('review_status')=='discuss')}")
    print(f"    unreviewed: {sum(1 for r in neg_reviews if r.get('review_status','unreviewed')=='unreviewed')}")
    print(f"  Conflicts: {len(conf_reviews)}")
    print(f"    with final decision: {sum(1 for r in conf_reviews if r.get('final_dim') and r.get('final_score','')!='')}")
    print(f"    all ok: {sum(1 for r in conf_reviews if r.get('review_status')=='agree')}")

    merged = load_merged()
    original = copy.deepcopy(merged)
    log = []

    print(f"\n{'='*60}")
    print("Applying negative span reviews...")
    log.append("=== NEGATIVE SPAN REMOVALS ===")
    removed = apply_negative_reviews(merged, neg_reviews, log)
    log.append(f"\nTotal removed: {removed}")

    print(f"\nApplying conflict decisions...")
    log.append("\n=== CONFLICT UNIFICATION ===")
    unified, changed = apply_conflict_decisions(merged, conf_reviews, log)
    log.append(f"\nTotal unified groups: {unified}, spans changed: {changed}")

    print(f"\nRebuilding summaries...")
    rebuild_summaries(merged)

    old_kept = original["metadata"]["total_kept_spans"]
    new_kept = merged["metadata"]["total_kept_spans"]

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Negative spans removed: {removed}")
    print(f"  Conflict groups unified: {unified}")
    print(f"  Spans label-changed: {changed}")
    print(f"  Total kept spans: {old_kept} → {new_kept}")

    for line in log:
        print(line)

    if args.dry_run:
        print(f"\n[DRY RUN] No files written.")
    else:
        out_path = Path(args.output) if args.output else MERGED
        backup = MERGED.with_name(f"merged_all_annotations_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        save_merged(original, backup)
        print(f"\n  Backup saved: {backup}")

        save_merged(merged, out_path)
        print(f"  Updated: {out_path}")


if __name__ == "__main__":
    main()
