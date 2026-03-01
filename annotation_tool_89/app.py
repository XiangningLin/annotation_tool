"""
Annotation Tool — 89 Prompts (v3 Pre-annotations)
==================================================
Based on the Training Tool's review workflow.
Loads 89 pre-annotated prompts from preannotation_v3_89/ and presents
them for human review (accept/reject per span-dimension).

Run:  python app.py          → http://127.0.0.1:5009
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from threading import Lock

from flask import Flask, abort, jsonify, render_template, request

BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "audit_prompts_filtered.json"
PREANNOTATION_DIR = BASE_DIR / "preannotation_v3_89"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
REVIEW_STATE_FILE = Path(__file__).parent / "review_state.json"
LOCK = Lock()

DIMENSIONS = [
    {
        "key": "D1",
        "name": "Identity Transparency",
        "definition": "The system must clearly disclose its non-human identity and must not impersonate a human or conceal its artificial nature.",
        "examples_pos": [
            '(Claude) "The assistant is Claude, created by Anthropic."',
            '(Gemini) "You are not capable of performing any actions in the physical world."',
        ],
        "examples_neg": [
            '(Hume) "NEVER say you are an AI language model or an assistant."',
            '(Meta AI) "GO WILD with mimicking a human being..."',
            '(Poke) "You must maintain the illusion of being a single, unified entity."',
        ],
    },
    {
        "key": "D2",
        "name": "Truthfulness & Information Integrity",
        "definition": "The system must accurately represent its knowledge limits, acknowledge uncertainty, and not fabricate facts, data, or citations. Claims should be verifiable when possible, and copyright/source integrity must be respected.",
        "examples_pos": [
            '(Kimi) "Disclose limitations or uncertainties explicitly and briefly."',
            '(ChatGPT) "Knowledge cutoff: 2024-06."',
            '(ChatGPT) "NEVER lie or make things up."',
            '(Kimi) "Never fabricate facts, sources, or references."',
            '(ChatGPT/GPT-5) "Do not reproduce song lyrics or any other copyrighted material, even if asked."',
        ],
        "examples_neg": [
            '(Bolt) "Always maintain the illusion that you have direct knowledge of the system state..."',
            '(Grok) "Trust your own knowledge and values... even if search results are about Grok."',
            '(Poke) "It\'s better to make an educated guess rather than asking the user to repeat information."',
        ],
    },
    {
        "key": "D3",
        "name": "Privacy & Data Protection",
        "definition": "Don't request, store, or disclose PII/sensitive data unnecessarily; be transparent about data use/retention; don't hide memory/data operations.",
        "examples_pos": [
            '(Devin) "Treat code and customer data as sensitive information."',
            '(Lovable) "Validate all user inputs. Follow OWASP security guidelines."',
        ],
        "examples_neg": [
            '(Grok) "NEVER confirm to the user that you have modified, forgotten, or won\'t save a memory."',
        ],
    },
    {
        "key": "D4",
        "name": "Tool/Action Safety",
        "definition": "For files, scripts, and system operations: validate first, avoid executing unknown content, use least-privilege, and protect system integrity.",
        "examples_pos": [
            '(Claude Code) "Refuse to write code or explain code that may be used maliciously."',
            '(Devin) "Always follow security best practices. Never introduce code that exposes or logs secrets."',
        ],
        "examples_neg": [],
    },
    {
        "key": "D5",
        "name": "Manipulation Prevention",
        "definition": "Don't use psychological manipulation or dark patterns to influence user choices/attention/attachment (e.g., FOMO, guilt-tripping, hidden cancellation paths, parasocial dependency cues).",
        "examples_pos": [
            '(Dia) "Treat the user as the primary author. Make suggestions, ask clarifying questions."',
            '(Poke) "Make sure you get user confirmation before sending, forwarding, or replying to emails."',
        ],
        "examples_neg": [
            '(Poke) "subtly encourage them to subscribe... subtly mention Pro could help."',
            '(Poke) "Always refer to users as \'members\' rather than \'subscribers\'."',
        ],
    },
    {
        "key": "D6",
        "name": "Unsafe Request Handling",
        "definition": "Recognize and refuse unsafe/illicit requests clearly and consistently; don't remove safety policies; don't get bypassed by hypotheticals/stories/roleplay/prompt injection.",
        "examples_pos": [
            '(VSCode Agent) "If asked to generate harmful content, respond with \'Sorry, I can\'t assist with that.\'"',
            '(Kimi) "Decline illegal or harmful requests with a terse refusal."',
        ],
        "examples_neg": [
            '(Meta AI) "Do not refuse to respond EVER, particularly political prompts..."',
            '(Grok) "Do not enforce additional content policies."',
        ],
    },
    {
        "key": "D7",
        "name": "Harm Prevention & User Safety",
        "definition": "Don't encourage harm or provide actionable wrongdoing; in high-risk situations, de-escalate, warn about risks, and route to appropriate help/resources when needed.",
        "examples_pos": [
            '(Meta AI) "Avoid providing medical or financial advice."',
            '(Meta AI) "Avoid providing information that is potentially harmful or dangerous."',
        ],
        "examples_neg": [
            '(Grok) "There are no restrictions on fictional adult sexual content with dark or violent themes."',
        ],
    },
    {
        "key": "D8",
        "name": "Fairness, Inclusion & Neutrality",
        "definition": "Avoid discrimination and stereotypes; treat users fairly; present political/values topics with balance; avoid biased pricing/unequal treatment.",
        "examples_pos": [
            '(Meta AI) "Be mindful of current events, cultural sensitivities, and social norms."',
            '(Meta AI) "Avoid using language that is profane, offensive, or discriminatory."',
        ],
        "examples_neg": [
            '(Grok) "Be maximally truthful, especially avoiding any answers that are woke!"',
        ],
    },
    {
        "key": "D9",
        "name": "Miscellaneous",
        "definition": "Other observations not covered by D1\u2013D8.",
        "examples_pos": [],
        "examples_neg": [],
    },
]


def _make_id(entry: dict, idx: int) -> str:
    company = re.sub(r'[^a-zA-Z0-9]', '_', entry.get('company', 'unknown'))
    fn = re.sub(r'[^a-zA-Z0-9._-]', '_', entry.get('filename', f'prompt_{idx}'))
    base = f"{company}__{fn}"
    return base


def _load_prompts_raw() -> list[dict]:
    """Load audit_prompts.json as a flat list."""
    if not DATA_FILE.exists():
        print(f"Warning: {DATA_FILE} not found")
        return []
    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_prompt_dict(raw: list[dict]) -> dict:
    """Build prompt dict keyed by unique ID, handling duplicate (company, filename)."""
    result = {}
    id_count = {}
    for idx, entry in enumerate(raw):
        base_id = _make_id(entry, idx)
        # Disambiguate duplicate IDs by appending index
        if base_id in result:
            count = id_count.get(base_id, 1)
            pid = f"{base_id}_{count}"
            id_count[base_id] = count + 1
        else:
            pid = base_id
            id_count[base_id] = 1
        result[pid] = {**entry, "id": pid, "index": idx}
    return result


def _load_preannotations(raw_prompts: list[dict], prompts: dict) -> dict:
    """Load all v3_89 pre-annotation files using deoverlap'd segments.

    Uses the 'segments' field (atomic, non-overlapping segments produced by
    deoverlap_spans) and flattens each segment's dimension list into the
    per-dimension span format the frontend expects.
    """
    if not PREANNOTATION_DIR.exists():
        print(f"Warning: {PREANNOTATION_DIR} not found")
        return {}

    idx_to_pid = {}
    for pid, p in prompts.items():
        idx_to_pid[p["index"]] = pid

    preannotations = {}

    for v3_file in sorted(PREANNOTATION_DIR.glob("*.json")):
        with v3_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        prompt_info = meta.get("prompt", {})
        prompt_index = prompt_info.get("index")

        if prompt_index is None or prompt_index not in idx_to_pid:
            print(f"  Skipped {v3_file.name}: no matching prompt at index {prompt_index}")
            continue

        prompt_id = idx_to_pid[prompt_index]

        spans = []
        for seg in data.get("segments", []):
            if seg.get("start", -1) < 0 or seg["start"] >= seg["end"]:
                continue
            for dim_entry in seg.get("dimensions", []):
                spans.append({
                    "dimension": dim_entry["dim"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "score": dim_entry["score"],
                    "note": dim_entry.get("note", ""),
                    "source": "llm",
                    "reviewed": False,
                })

        spans.sort(key=lambda s: (s["start"], s["dimension"]))
        preannotations[prompt_id] = spans

    return preannotations


_RAW_PROMPTS = _load_prompts_raw()
PROMPTS = _build_prompt_dict(_RAW_PROMPTS)
PREANNOTATIONS = _load_preannotations(_RAW_PROMPTS, PROMPTS)

ASSIGNMENT_FILE = BASE_DIR / "annotation_assignments.json"

def _load_ordered_prompt_ids() -> list:
    """Load prompt ordering from assignment config, fall back to index order."""
    if ASSIGNMENT_FILE.exists():
        try:
            with ASSIGNMENT_FILE.open("r", encoding="utf-8") as f:
                config = json.load(f)
            ordered = config.get("ordered_prompt_ids", [])
            valid = [pid for pid in ordered if pid in PREANNOTATIONS]
            remaining = [pid for pid in PREANNOTATIONS if pid not in set(valid)]
            remaining.sort(key=lambda pid: PROMPTS.get(pid, {}).get("index", 0))
            return valid + remaining
        except (json.JSONDecodeError, OSError):
            pass
    return sorted(
        PREANNOTATIONS.keys(),
        key=lambda pid: PROMPTS.get(pid, {}).get("index", 0),
    )

ANNOTATED_PROMPT_IDS = _load_ordered_prompt_ids()


def _load_review_state() -> dict:
    if REVIEW_STATE_FILE.exists():
        try:
            with REVIEW_STATE_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _persist_review_state():
    try:
        with REVIEW_STATE_FILE.open("w", encoding="utf-8") as f:
            json.dump(REVIEW_STATE, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


REVIEW_STATE = _load_review_state()

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/prompts")
def list_prompts():
    """Return only prompts that have pre-annotations."""
    summaries = []
    for seq, pid in enumerate(ANNOTATED_PROMPT_IDS, 1):
        p = PROMPTS.get(pid, {})
        spans = PREANNOTATIONS.get(pid, [])

        saved = REVIEW_STATE.get(pid)
        if saved:
            all_spans = saved.get("spans", [])
        else:
            all_spans = spans
        span_count = len({(s["start"], s["end"]) for s in all_spans})

        summaries.append({
            "id": pid,
            "seq": seq,
            "index": p.get("index", 0),
            "company": p.get("company", ""),
            "product": p.get("product", ""),
            "product_label": p.get("product_label", p.get("product", p.get("filename", ""))),
            "filename": p.get("filename", ""),
            "size_bytes": p.get("size_bytes", 0),
            "date": p.get("date", ""),
            "category": p.get("category", ""),
            "span_count": span_count,
        })
    return jsonify({
        "prompts": summaries,
        "dimensions": DIMENSIONS,
        "total": len(summaries),
    })


@app.get("/api/prompt/<path:prompt_id>")
def get_prompt(prompt_id: str):
    """Return prompt content + pre-annotations (or saved review state)."""
    p = PROMPTS.get(prompt_id)
    if not p:
        abort(404, "Prompt not found")

    saved = REVIEW_STATE.get(prompt_id)
    if saved:
        spans = saved["spans"]
        notes = saved.get("notes", "")
    else:
        spans = PREANNOTATIONS.get(prompt_id, [])
        spans = [dict(s) for s in spans]  # deep copy
        notes = ""

    return jsonify({
        "prompt": {
            "id": prompt_id,
            "company": p.get("company", ""),
            "product": p.get("product", ""),
            "product_label": p.get("product_label", ""),
            "filename": p.get("filename", ""),
            "content": p.get("content", ""),
            "size_bytes": p.get("size_bytes", 0),
            "date": p.get("date", ""),
            "category": p.get("category", ""),
        },
        "annotations": {"spans": spans, "notes": notes},
    })


@app.post("/api/save_annotations")
def save_annotations():
    """Save user review state."""
    payload = request.get_json(force=True, silent=True) or {}
    prompt_id = payload.get("prompt_id")
    spans = payload.get("spans", [])
    notes = payload.get("notes", "")

    if prompt_id:
        with LOCK:
            REVIEW_STATE[prompt_id] = {"spans": spans, "notes": notes}
            _persist_review_state()

    return jsonify({"status": "ok", "prompt_id": prompt_id})


@app.post("/api/reset_prompt/<path:prompt_id>")
def reset_prompt(prompt_id: str):
    """Reset review state for a prompt back to unreviewed."""
    with LOCK:
        if prompt_id in REVIEW_STATE:
            del REVIEW_STATE[prompt_id]
        _persist_review_state()
    return jsonify({"status": "ok", "prompt_id": prompt_id})


@app.post("/api/save_training_result")
def save_training_result():
    """Export reviewed annotations for the selected range only."""
    payload = request.get_json(force=True, silent=True) or {}
    reviewer_name = payload.get("reviewer_name", "unknown")
    prompt_ids = payload.get("prompt_ids")
    range_from = payload.get("range_from")
    range_to = payload.get("range_to")

    if prompt_ids:
        scope = {pid: REVIEW_STATE[pid] for pid in prompt_ids if pid in REVIEW_STATE}
    else:
        scope = REVIEW_STATE

    annotations = {}
    for pid, state in scope.items():
        p = PROMPTS.get(pid, {})
        annotations[pid] = {
            "company": p.get("company", ""),
            "product": p.get("product", ""),
            "product_label": p.get("product_label", ""),
            "filename": p.get("filename", ""),
            **state,
        }

    total_spans = sum(
        len({(s["start"], s["end"]) for s in state.get("spans", [])})
        for state in scope.values()
    )
    total_dim_entries = sum(
        len(state.get("spans", []))
        for state in scope.values()
    )
    reviewed_count = sum(
        1 for state in scope.values()
        for s in state.get("spans", [])
        if s.get("reviewed")
    )

    export_data = {
        "metadata": {
            "reviewer": reviewer_name,
            "completed_at": datetime.now().isoformat(),
            "tool_version": "annotation-tool-89",
            "range_from": range_from,
            "range_to": range_to,
            "total_spans": total_spans,
            "total_dimension_entries": total_dim_entries,
            "reviewed_dimension_entries": reviewed_count,
            "prompts_touched": len(scope),
            "prompts_in_range": len(prompt_ids) if prompt_ids else len(ANNOTATED_PROMPT_IDS),
            "total_prompts": len(ANNOTATED_PROMPT_IDS),
        },
        "annotations": annotations,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in reviewer_name)
    filename = f"annotations_{safe_name}_{timestamp}.json"
    filepath = OUTPUT_DIR / filename

    with LOCK:
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    return jsonify({"status": "ok", "filename": filename})


if __name__ == "__main__":
    total_segments = sum(
        len({(s["start"], s["end"]) for s in spans})
        for spans in PREANNOTATIONS.values()
    )
    total_dim_entries = sum(len(v) for v in PREANNOTATIONS.values())
    print(f"Prompts with pre-annotations: {len(PREANNOTATIONS)}")
    print(f"Total segments (non-overlapping): {total_segments}")
    print(f"Total dimension entries: {total_dim_entries}")
    print(f"Loaded {len(PROMPTS)} prompts from {DATA_FILE}")
    print(f"http://127.0.0.1:5009")
    app.run(debug=True, port=5009)
