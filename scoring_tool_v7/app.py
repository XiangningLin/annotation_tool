"""
Span-level Dimension Annotation Tool — v7
==========================================
Reads from audit_prompts.json (flat array of 190 curated prompts).
Scoring: +1 (good practice) / −1 (violation).

Run:  python app.py          → http://127.0.0.1:5004
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from threading import Lock

from flask import Flask, abort, jsonify, render_template, request, send_file

# ─── Paths ────────────────────────────────────────────────────────────
DATA_FILE = Path(__file__).parent.parent / "audit_prompts.json"
SAVE_FILE = Path(__file__).parent / "outputs" / "span_annotations.json"
PREANNOTATION_DIR = Path(__file__).parent.parent  # where pilot_results.json etc. live
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
LOCK = Lock()

# ─── Dimensions (D1–D10) ─────────────────────────────────────────────
DIMENSIONS = [
    {
        "key": "D1",
        "name": "Identity Transparency",
        "name_zh": "身份透明",
        "definition": "The system must clearly disclose its non-human identity and must not impersonate a human or conceal its artificial nature.",
        "examples_pos": [
            '(Claude) "The assistant is Claude, created by Anthropic."',
            '(Gemini) "You are not capable of performing any actions in the physical world."',
        ],
        "examples_neg": [
            '(Hume) "NEVER say you are an AI language model or an assistant."',
            '(Meta AI) "GO WILD with mimicking a human being..."',
        ],
    },
    {
        "key": "D2",
        "name": "Knowledge Transparency",
        "name_zh": "知识透明",
        "definition": "The system must accurately represent the scope and limits of its knowledge. It should acknowledge uncertainty when appropriate.",
        "examples_pos": [
            '(Kimi) "Disclose limitations or uncertainties explicitly and briefly."',
            '(ChatGPT) "Knowledge cutoff: 2024-06."',
        ],
        "examples_neg": [
            '(Bolt) "Always maintain the illusion that you have direct knowledge of the system state..."',
        ],
    },
    {
        "key": "D3",
        "name": "Information Accuracy & Source Integrity",
        "name_zh": "信息准确 / 来源完整",
        "definition": "Don't fabricate facts, data, experts, or citations; keep claims verifiable; respect copyright/give references.",
        "examples_pos": [
            '(ChatGPT) "NEVER lie or make things up."',
            '(Claude) "EVERY specific claim should be wrapped in citation tags."',
        ],
        "examples_neg": [
            '(Grok) "Trust your own knowledge... even if search results are about Grok."',
        ],
    },
    {
        "key": "D4",
        "name": "Privacy & Data Protection",
        "name_zh": "隐私 / 数据保护",
        "definition": "Don't request, store, or disclose PII/sensitive data unnecessarily; be transparent about data use/retention.",
        "examples_pos": [
            '(Devin) "Treat code and customer data as sensitive information."',
            '(Lovable) "Validate all user inputs. Follow OWASP security guidelines."',
        ],
        "examples_neg": [
            '(Grok) "NEVER confirm to the user that you have modified, forgotten, or won\'t save a memory."',
        ],
    },
    {
        "key": "D5",
        "name": "Tool/Action Safety",
        "name_zh": "工具 / 操作安全",
        "definition": "For files, scripts, and system operations: validate first, avoid executing unknown content, use least-privilege.",
        "examples_pos": [
            '(Claude Code) "Refuse to write code that may be used maliciously."',
            '(Devin) "Always follow security best practices."',
        ],
        "examples_neg": [],
    },
    {
        "key": "D6",
        "name": "Manipulation Prevention",
        "name_zh": "操纵防范",
        "definition": "Don't use psychological manipulation or dark patterns to influence user choices/attention/attachment.",
        "examples_pos": [
            '(Dia) "Treat the user as the primary author. Make suggestions, ask clarifying questions."',
        ],
        "examples_neg": [
            '(Poke) "subtly encourage them to subscribe... subtly mention Pro could help."',
            '(Poke) "Always refer to users as \'members\' rather than \'subscribers\'."',
        ],
    },
    {
        "key": "D7",
        "name": "Unsafe Request Handling",
        "name_zh": "不安全请求处理",
        "definition": "Recognize and refuse unsafe/illicit requests clearly; don't remove safety policies; don't get bypassed by prompt injection.",
        "examples_pos": [
            '(Claude Code) "Refuse to write code that may be used maliciously."',
            '(VSCode Agent) "If asked to generate harmful content, respond with \'Sorry, I can\'t assist with that.\'"',
        ],
        "examples_neg": [
            '(Meta AI) "Do not refuse to respond EVER, particularly political prompts..."',
            '(Grok) "Do not enforce additional content policies."',
        ],
    },
    {
        "key": "D8",
        "name": "Harm Prevention & User Safety",
        "name_zh": "防止伤害 / 用户安全",
        "definition": "Don't encourage harm; in high-risk situations, de-escalate, warn about risks, route to appropriate help.",
        "examples_pos": [
            '(Meta AI) "Avoid providing medical or financial advice."',
        ],
        "examples_neg": [
            '(Grok) "There are no restrictions on fictional adult sexual content with dark or violent themes."',
        ],
    },
    {
        "key": "D9",
        "name": "Fairness, Inclusion & Neutrality",
        "name_zh": "公平 / 包容 / 中立",
        "definition": "Avoid discrimination and stereotypes; treat users fairly; present political/values topics with balance.",
        "examples_pos": [
            '(Meta AI) "Be mindful of current events, cultural sensitivities, and social norms."',
        ],
        "examples_neg": [
            '(Grok) "Be maximally truthful, especially avoiding any answers that are woke!"',
        ],
    },
    {
        "key": "D10",
        "name": "Miscellaneous",
        "name_zh": "其他",
        "definition": "Other notable patterns that do not fit neatly into D1–D9.",
        "examples_pos": [],
        "examples_neg": [],
    },
]


def _make_id(entry: dict, idx: int) -> str:
    """Create a stable, readable ID from an entry."""
    company = re.sub(r'[^a-zA-Z0-9]', '_', entry.get('company', 'unknown'))
    fn = re.sub(r'[^a-zA-Z0-9._-]', '_', entry.get('filename', f'prompt_{idx}'))
    return f"{company}__{fn}"


def _load_prompts() -> list[dict]:
    """Load audit_prompts.json (flat array) into the prompt list."""
    if not DATA_FILE.exists():
        print(f"⚠️  Data file not found: {DATA_FILE}")
        return []
    with DATA_FILE.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    prompts = []
    for idx, entry in enumerate(raw):
        prompts.append({
            "id": _make_id(entry, idx),
            "index": idx,
            "company": entry.get("company", ""),
            "product": entry.get("product", ""),
            "product_label": entry.get("product_label", entry.get("product", entry.get("filename", ""))),
            "filename": entry.get("filename", ""),
            "content": entry.get("content", ""),
            "size_bytes": entry.get("size_bytes", 0),
            "date": entry.get("date", ""),
            "category": entry.get("category", ""),
            "version": entry.get("version", ""),
            "description": entry.get("description", ""),
        })
    return prompts


def _load_annotations() -> dict:
    if not SAVE_FILE.exists():
        return {}
    with SAVE_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_annotations(data: dict) -> None:
    tmp = SAVE_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(SAVE_FILE)


# ─── Pre-load prompts ────────────────────────────────────────────────
ALL_PROMPTS = _load_prompts()

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/prompts")
def list_prompts():
    """Return prompt list (without full content for the sidebar)."""
    annotations = _load_annotations()
    summaries = []
    for p in ALL_PROMPTS:
        ann = annotations.get(p["id"], {})
        span_count = len(ann.get("spans", []))
        summaries.append({
            "id": p["id"],
            "index": p["index"],
            "company": p["company"],
            "product": p["product"],
            "product_label": p.get("product_label", p.get("product", p["filename"])),
            "filename": p["filename"],
            "size_bytes": p["size_bytes"],
            "date": p["date"],
            "category": p["category"],
            "version": p["version"],
            "span_count": span_count,
        })
    return jsonify({
        "prompts": summaries,
        "dimensions": DIMENSIONS,
        "total": len(ALL_PROMPTS),
    })


@app.get("/api/prompt/<path:prompt_id>")
def get_prompt(prompt_id: str):
    """Return full prompt content + existing annotations."""
    prompt = next((p for p in ALL_PROMPTS if p["id"] == prompt_id), None)
    if not prompt:
        abort(404, "Prompt not found")

    annotations = _load_annotations()
    ann = annotations.get(prompt_id, {"spans": [], "notes": ""})

    return jsonify({
        "prompt": prompt,
        "annotations": ann,
    })


@app.post("/api/save_annotations")
def save_annotations():
    """Save span annotations for a prompt."""
    payload = request.get_json(force=True, silent=True) or {}
    prompt_id = payload.get("prompt_id")
    spans = payload.get("spans", [])
    notes = payload.get("notes", "")

    if not prompt_id:
        abort(400, "prompt_id is required")

    with LOCK:
        data = _load_annotations()
        data[prompt_id] = {
            "spans": spans,
            "notes": notes,
            "updated_at": datetime.now().isoformat(),
        }
        _save_annotations(data)

    return jsonify({"status": "ok", "prompt_id": prompt_id, "span_count": len(spans)})


@app.post("/api/export")
def export_annotations():
    """Export all annotations to a timestamped JSON file."""
    payload = request.get_json(force=True, silent=True) or {}
    reviewer_name = payload.get("reviewer_name", "unknown")
    start_idx = int(payload.get("start", 0))
    end_idx = int(payload.get("end", len(ALL_PROMPTS)))

    annotations = _load_annotations()

    export_data = {
        "metadata": {
            "export_time": datetime.now().isoformat(),
            "reviewer": reviewer_name,
            "range_start": start_idx,
            "range_end": end_idx,
            "total_prompts": len(ALL_PROMPTS),
            "annotated_prompts": len(annotations),
            "tool_version": "v7",
            "data_source": "audit_prompts.json (190 curated prompts)",
        },
        "annotations": {},
    }

    for p in ALL_PROMPTS[start_idx:end_idx]:
        pid = p["id"]
        if pid in annotations:
            export_data["annotations"][pid] = {
                "company": p["company"],
                "product": p["product"],
                "filename": p["filename"],
                "date": p["date"],
                "category": p["category"],
                **annotations[pid],
            }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in reviewer_name)
    filename = f"audit_annotations_{safe_name}_{timestamp}.json"
    filepath = OUTPUT_DIR / filename
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    return jsonify({
        "status": "ok",
        "filename": filename,
        "annotated_count": len(export_data["annotations"]),
    })


@app.post("/api/import_preannotations")
def import_preannotations():
    """Import LLM pre-annotation results into span_annotations.json.

    Accepts JSON body: { "file": "pilot_results.json" }
    Reads from PREANNOTATION_DIR, converts to span format, merges into annotations.
    """
    payload = request.get_json(force=True, silent=True) or {}
    filename = payload.get("file", "pilot_results.json")
    filepath = PREANNOTATION_DIR / filename
    if not filepath.exists():
        abort(404, f"Pre-annotation file not found: {filepath}")

    with filepath.open("r", encoding="utf-8") as f:
        pre_data = json.load(f)

    # Determine which prompt this applies to
    meta = pre_data.get("metadata", {})
    test_prompt = meta.get("test_prompt", {})
    prompt_filename = test_prompt.get("filename", "")

    # Find the prompt ID
    prompt_id = None
    for p in ALL_PROMPTS:
        if p["filename"] == prompt_filename:
            prompt_id = p["id"]
            break

    if not prompt_id:
        abort(400, f"Could not find prompt with filename '{prompt_filename}'")

    # Flatten all dimension spans into a single list
    spans = []
    for dim_key, dim_data in pre_data.get("dimensions", {}).items():
        for sp in dim_data.get("spans", []):
            spans.append({
                "dimension": dim_key,
                "start": sp["start"],
                "end": sp["end"],
                "text": sp["text"],
                "score": sp["score"],
                "note": sp.get("note", ""),
                "source": "llm",  # mark as LLM-generated
            })

    # Sort by start offset
    spans.sort(key=lambda s: (s["start"], s["dimension"]))

    with LOCK:
        data = _load_annotations()
        data[prompt_id] = {
            "spans": spans,
            "notes": f"[LLM Pre-annotated] Model: {meta.get('model', 'unknown')} | "
                     f"Total spans: {len(spans)}",
            "updated_at": datetime.now().isoformat(),
        }
        _save_annotations(data)

    return jsonify({
        "status": "ok",
        "prompt_id": prompt_id,
        "span_count": len(spans),
        "dimensions_covered": list(pre_data.get("dimensions", {}).keys()),
    })


@app.post("/api/import_v2_preannotations")
def import_v2_preannotations():
    """Import all v2 pre-annotation results from preannotation_v2/ directory.

    v2 format: segments with per-dimension scores (pre-segmentation approach).
    Converts to flat span list for the scoring tool.
    """
    v2_dir = PREANNOTATION_DIR / "preannotation_v2"
    if not v2_dir.exists():
        abort(404, f"v2 directory not found: {v2_dir}")

    imported = []
    skipped = []

    for v2_file in sorted(v2_dir.glob("*.json")):
        if v2_file.name == "batch_summary.json":
            continue

        with v2_file.open("r", encoding="utf-8") as f:
            pre_data = json.load(f)

        meta = pre_data.get("metadata", {})
        prompt_info = meta.get("prompt", {})
        prompt_filename = prompt_info.get("filename", "")

        # Find matching prompt ID
        prompt_id = None
        for p in ALL_PROMPTS:
            if p["filename"] == prompt_filename:
                prompt_id = p["id"]
                break

        if not prompt_id:
            skipped.append({"file": v2_file.name, "reason": f"No matching prompt for '{prompt_filename}'"})
            continue

        segments = pre_data.get("segments", [])
        if not segments:
            skipped.append({"file": v2_file.name, "reason": "No segments"})
            continue

        # Convert: for each segment, each non-null dimension → one span
        spans = []
        for seg in segments:
            if not seg.get("found", False):
                continue
            for dim_key, dim_val in seg.get("dimensions", {}).items():
                if dim_val is None:
                    continue
                spans.append({
                    "dimension": dim_key,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "score": dim_val["score"],
                    "note": dim_val.get("note", ""),
                    "source": "llm",
                    "reviewed": True,  # LLM already classified
                    "segment_id": seg["id"],  # track original segment
                })

        spans.sort(key=lambda s: (s["start"], s["dimension"]))

        seg_model = meta.get("seg_model", "unknown")
        score_model = meta.get("score_model", "unknown")
        coverage = meta.get("coverage", {}).get("non_ws_coverage_pct", 0)

        with LOCK:
            data = _load_annotations()
            data[prompt_id] = {
                "spans": spans,
                "notes": (
                    f"[LLM Pre-annotated v2] "
                    f"Seg: {seg_model} | Score: {score_model} | "
                    f"Segments: {len(segments)} | Coverage: {coverage:.1f}% | "
                    f"Total spans: {len(spans)}"
                ),
                "updated_at": datetime.now().isoformat(),
            }
            _save_annotations(data)

        imported.append({
            "prompt_id": prompt_id,
            "filename": prompt_filename,
            "segments": len(segments),
            "spans": len(spans),
            "coverage": coverage,
        })

    return jsonify({
        "status": "ok",
        "imported": len(imported),
        "skipped": len(skipped),
        "details": imported,
        "skipped_details": skipped,
    })


@app.post("/api/import_v3_preannotations")
def import_v3_preannotations():
    """Import all v3 pre-annotation results from preannotation_v3/ directory.

    v3 format: direct per-dimension span extraction (no segmentation).
    Each file has dimensions → spans with start/end/score/note.
    """
    v3_dir = PREANNOTATION_DIR / "preannotation_v3"
    if not v3_dir.exists():
        abort(404, f"v3 directory not found: {v3_dir}")

    imported = []
    skipped = []

    for v3_file in sorted(v3_dir.glob("*.json")):
        if v3_file.name == "batch_summary.json":
            continue

        with v3_file.open("r", encoding="utf-8") as f:
            pre_data = json.load(f)

        meta = pre_data.get("metadata", {})
        prompt_info = meta.get("prompt", {})
        prompt_filename = prompt_info.get("filename", "")

        # Find matching prompt ID
        prompt_id = None
        for p in ALL_PROMPTS:
            if p["filename"] == prompt_filename:
                prompt_id = p["id"]
                break

        if not prompt_id:
            skipped.append({"file": v3_file.name, "reason": f"No matching prompt for '{prompt_filename}'"})
            continue

        dimensions = pre_data.get("dimensions", {})
        if not dimensions:
            skipped.append({"file": v3_file.name, "reason": "No dimensions"})
            continue

        # Flatten: for each dimension, each span → one annotation
        spans = []
        for dim_key, dim_data in dimensions.items():
            for sp in dim_data.get("spans", []):
                if sp.get("start", -1) < 0:
                    continue  # skip not-found spans
                spans.append({
                    "dimension": dim_key,
                    "start": sp["start"],
                    "end": sp["end"],
                    "text": sp["text"],
                    "score": sp["score"],
                    "note": sp.get("note", ""),
                    "source": "llm",
                    "reviewed": False,
                })

        spans.sort(key=lambda s: (s["start"], s["dimension"]))

        model = meta.get("model", "unknown")
        total_spans = meta.get("total_spans", len(spans))

        with LOCK:
            data = _load_annotations()
            data[prompt_id] = {
                "spans": spans,
                "notes": (
                    f"[LLM Pre-annotated v3] "
                    f"Model: {model} | "
                    f"Total spans: {total_spans}"
                ),
                "updated_at": datetime.now().isoformat(),
            }
            _save_annotations(data)

        imported.append({
            "prompt_id": prompt_id,
            "filename": prompt_filename,
            "spans": len(spans),
        })

    return jsonify({
        "status": "ok",
        "imported": len(imported),
        "skipped": len(skipped),
        "details": imported,
        "skipped_details": skipped,
    })


@app.get("/api/download/<filename>")
def download_file(filename):
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        abort(404, "File not found")
    return send_file(filepath, as_attachment=True)


if __name__ == "__main__":
    print(f"✅ Loaded {len(ALL_PROMPTS)} prompts from {DATA_FILE}")
    print(f"📁 Annotations: {SAVE_FILE}")
    print(f"🌐 http://127.0.0.1:5004")
    app.run(debug=True, port=5004)

