#!/usr/bin/env python3
"""
Apply cross-version review decisions to merged annotations.

After reviewing cross-version conflicts in the review tool (/review),
this script reads the review decisions and unifies labels so that
identical text across same-company product versions gets identical labels.

Decision types:
  - "agree"    → keep current state (no change)
  - "disagree" + final_labels → apply those labels uniformly to ALL versions
  - "discuss"  → skip (needs further discussion)

Usage:
    python scripts/apply_cross_version_review.py --dry-run
    python scripts/apply_cross_version_review.py
"""

import json
import copy
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path(__file__).parent.parent
ANALYSIS = ROOT / "annotation_tool_89" / "analysis"
MERGED = ANALYSIS / "merged_all_annotations.json"
CONFLICTS_FILE = ANALYSIS / "shared_text_review.json"
REVIEW_STATE_FILE = ANALYSIS / "neg_review_state.json"


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Apply cross-version review decisions")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    if not REVIEW_STATE_FILE.exists():
        print("No review state found. Run the review tool first: http://127.0.0.1:5009/review")
        return

    reviews = load_json(REVIEW_STATE_FILE)
    raw_conflicts = load_json(CONFLICTS_FILE)
    conflicts = (raw_conflicts if isinstance(raw_conflicts, list)
                 else raw_conflicts.get("conflicts", []))
    data = load_json(MERGED)
    original = copy.deepcopy(data)

    # Index conflicts by _id
    conflict_map = {}
    for i, c in enumerate(conflicts):
        cid = c.get("_id", f"c{i}")
        conflict_map[cid] = c

    # Gather stats
    stats = {"agree": 0, "disagree_applied": 0, "discuss": 0,
             "unreviewed": 0, "no_labels": 0,
             "spans_added": 0, "spans_removed": 0, "spans_changed": 0}
    log = []

    for cid, review in sorted(reviews.items()):
        conflict = conflict_map.get(cid)
        if not conflict:
            continue

        status = review.get("status", "")
        final_labels = review.get("final_labels", [])
        comment = review.get("comment", "")

        if status == "agree":
            stats["agree"] += 1
            continue
        if status == "discuss":
            stats["discuss"] += 1
            continue
        if status != "disagree":
            stats["unreviewed"] += 1
            continue
        if not final_labels:
            stats["no_labels"] += 1
            log.append(f"  SKIP {cid}: disagree but no final_labels provided")
            continue

        # Apply final_labels to all versions
        overlap_text = conflict["overlap_text"]
        company = conflict["company"]
        annotations = conflict.get("annotations", [])
        prompt_ids = list(set(a["prompt_id"] for a in annotations))

        log.append(f"\n  APPLY {cid} [{company}]: {len(final_labels)} label(s) → {len(prompt_ids)} versions")
        log.append(f"    Text: \"{overlap_text[:80]}...\"")
        label_str = ", ".join(str(lb["dim"]) + "=" + format(lb["score"], "+d") for lb in final_labels)
        log.append(f"    Labels: {label_str}")
        if comment:
            log.append(f"    Comment: {comment}")

        for pid in prompt_ids:
            prompt = data["prompts"].get(pid)
            if not prompt:
                log.append(f"    WARNING: {pid} not found in merged data")
                continue

            kept = prompt["kept_spans"]
            prompt_text = None

            # Find the span(s) covering this overlap text
            matching_spans = []
            for s in kept:
                span_text = s.get("text", "")
                if not span_text:
                    continue
                # Match: overlap_text is contained in span_text, or they share significant overlap
                if (overlap_text[:80] in span_text or span_text in overlap_text
                        or overlap_text[:50] == span_text[:50]):
                    matching_spans.append(s)

            if not matching_spans:
                # Text exists in this version but no span was annotated (missing_span case)
                # Use the start/end from another version's annotation
                ref_ann = next((a for a in annotations if a["prompt_id"] != pid and not a.get("missing")), None)
                if not ref_ann:
                    log.append(f"    {pid}: no matching span and no reference — skip")
                    continue

                # Find the start/end from a version that has the span
                ref_pid = ref_ann["prompt_id"]
                ref_prompt = data["prompts"].get(ref_pid)
                if not ref_prompt:
                    continue
                ref_matches = [s for s in ref_prompt["kept_spans"]
                               if overlap_text[:80] in s.get("text", "")
                               or s.get("text", "") in overlap_text
                               or overlap_text[:50] == s.get("text", "")[:50]]
                if not ref_matches:
                    log.append(f"    {pid}: cannot find reference span — skip")
                    continue

                # Find where this text appears in the target prompt
                from_audit = None
                audit_path = ROOT / "data" / "audit_prompts.json"
                if audit_path.exists():
                    with audit_path.open() as f:
                        for p in json.load(f):
                            test_pid = f"{p['company']}__{p['filename']}"
                            if test_pid == pid or test_pid.replace(" ", "_") == pid:
                                from_audit = p.get("content", "")
                                break

                if from_audit and overlap_text[:50] in from_audit:
                    text_start = from_audit.find(overlap_text[:50])
                    text_end = text_start + len(overlap_text)
                    actual_text = from_audit[text_start:text_end]
                else:
                    # Fallback: use reference span's position
                    text_start = ref_matches[0]["start"]
                    text_end = ref_matches[0]["end"]
                    actual_text = ref_matches[0].get("text", overlap_text)

                # Add new spans for each final label
                added = 0
                for lb in final_labels:
                    existing = any(
                        s["dimension"] == lb["dim"] and s.get("score") == lb["score"]
                        and s["start"] == text_start and s["end"] == text_end
                        for s in kept
                    )
                    if existing:
                        continue
                    kept.append({
                        "start": text_start,
                        "end": text_end,
                        "text": actual_text,
                        "dimension": lb["dim"],
                        "score": lb["score"],
                        "note": f"Added by cross-version review ({cid})",
                        "source": "llm",
                        "reviewed": True,
                    })
                    added += 1
                    stats["spans_added"] += 1

                if added:
                    log.append(f"    {pid}: +{added} new span(s) (was missing)")

            else:
                # Has matching spans — update labels to match final_labels
                # Remove existing labels at this position
                match_positions = set((s["start"], s["end"]) for s in matching_spans)
                old_labels = set()
                for s in matching_spans:
                    old_labels.add((s["dimension"], s.get("score", 0)))

                new_label_set = set((lb["dim"], lb["score"]) for lb in final_labels)

                if old_labels == new_label_set:
                    continue

                # Remove old spans at matching positions
                to_remove = set()
                for s in kept:
                    if (s["start"], s["end"]) in match_positions:
                        for ms in matching_spans:
                            if s["start"] == ms["start"] and s["end"] == ms["end"] and s["dimension"] == ms["dimension"]:
                                to_remove.add(id(s))
                                break

                ref_span = matching_spans[0]
                new_kept = [s for s in kept if id(s) not in to_remove]

                # Add final labels
                for lb in final_labels:
                    new_kept.append({
                        "start": ref_span["start"],
                        "end": ref_span["end"],
                        "text": ref_span.get("text", overlap_text),
                        "dimension": lb["dim"],
                        "score": lb["score"],
                        "note": ref_span.get("note", "") or f"Updated by cross-version review ({cid})",
                        "source": ref_span.get("source", "llm"),
                        "reviewed": True,
                    })

                removed = len(to_remove)
                added = len(final_labels)
                prompt["kept_spans"] = new_kept
                stats["spans_removed"] += removed
                stats["spans_added"] += added
                stats["spans_changed"] += 1

                old_str = ", ".join(f"{d}={s:+d}" for d, s in sorted(old_labels))
                new_str = ", ".join(f"{lb['dim']}={lb['score']:+d}" for lb in final_labels)
                log.append(f"    {pid}: [{old_str}] → [{new_str}]")

        stats["disagree_applied"] += 1

    # Rebuild counts
    total_kept = 0
    for pid, prompt in data["prompts"].items():
        prompt["kept_spans"].sort(key=lambda s: (s.get("start", 0), s.get("end", 0)))
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

    # Print summary
    print("=" * 60)
    print("CROSS-VERSION REVIEW APPLICATION SUMMARY")
    print("=" * 60)
    print(f"  Conflicts reviewed:   {stats['agree'] + stats['disagree_applied'] + stats['discuss']}")
    print(f"    Agree (no change):  {stats['agree']}")
    print(f"    Applied:            {stats['disagree_applied']}")
    print(f"    Discuss (skipped):  {stats['discuss']}")
    print(f"    Unreviewed:         {stats['unreviewed']}")
    if stats["no_labels"]:
        print(f"    No labels (skip):   {stats['no_labels']}")
    print(f"  Spans added:          {stats['spans_added']}")
    print(f"  Spans removed:        {stats['spans_removed']}")
    print(f"  Total kept spans:     {old_kept} → {total_kept} (delta: {total_kept - old_kept:+d})")

    for line in log:
        print(line)

    if args.dry_run:
        print(f"\n[DRY RUN] No files written.")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = MERGED.with_name(f"merged_all_annotations_pre_review_{ts}.json")
        with backup.open("w", encoding="utf-8") as f:
            json.dump(original, f, indent=2, ensure_ascii=False)
        print(f"\n  Backup: {backup}")
        with MERGED.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Written: {MERGED}")

        # Regenerate negative_spans_review.json + shared_text_review.json
        print(f"\n  Regenerating derived data files...")
        import subprocess
        regen_script = ROOT / "scripts" / "regenerate_negative_spans.py"
        if regen_script.exists():
            result = subprocess.run(["python3", str(regen_script)], capture_output=True, text=True)
            print(f"  {result.stdout.strip()}")
        else:
            print(f"  WARNING: {regen_script} not found — run it manually")


if __name__ == "__main__":
    main()
