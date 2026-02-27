# System Prompt Audit Tool

A toolkit for auditing AI system prompts across safety and transparency dimensions. Includes LLM-assisted pre-annotation and a web-based human review tool.

## Quick Start

### 1. Human Review Tool

```bash
pip install -r requirements.txt
cd scoring_tool_v7
python app.py
```

Open **http://127.0.0.1:5004** → Enter your name → Check **📌 Pre-annotated only** → Start.

### 2. LLM Pre-Annotation (v4 — recommended)

```bash
export OPENROUTER_API_KEY="sk-or-..."

# Single prompt
python llm_preannotate_v4.py --index 1

# Batch (pilot 20 prompts, 4 parallel workers)
python llm_preannotate_v4.py --batch --parallel 4

# Resume interrupted batch
python llm_preannotate_v4.py --batch --parallel 4 --resume

# Dry run (show plan, no API calls)
python llm_preannotate_v4.py --batch --dry-run

# Custom prompt indices
python llm_preannotate_v4.py --indices "1,2,3,93,148" --batch --parallel 4
```

After pre-annotation, import into the review tool:
```bash
curl -X POST http://localhost:5004/api/import_v3_preannotations
```

## How It Works

### Pre-Annotation Pipeline (v4)

Two-step approach with 2 API calls per prompt:

| Step | Calls | What it does |
|------|-------|-------------|
| **Step 1: Segment** | 1 | Split document into non-overlapping semantic units |
| **Step 2: Label** | 1 | Assign each segment the best-fit 1-2 dimensions + score |

**Why v4?** Previous versions (v3) called the LLM 9 times per prompt (once per dimension), causing 51% partial overlap between spans. v4 eliminates this by using unified segmentation and labeling all dimensions in a single call.

| | v3 | v4 |
|---|---|---|
| API calls per prompt | 9 | **2** |
| Partial overlap | 51% | **0%** |
| Cost per prompt | ~$0.15 | **~$0.05-0.10** |

### Human Review Tool

The review tool shows LLM pre-annotations as pending cards. For each span:

1. **Review LLM suggestions** — Accept or Reject each suggested dimension
2. **Change scores** — Switch between +1 (good practice) and -1 (violation)
3. **Add dimensions** — If the LLM missed a relevant dimension, add it manually
4. **Undo** — Revoke any decision before finalizing
5. **Done** — Confirm the span is fully reviewed

You can also select any text in the prompt to create new annotations manually.

## Audit Dimensions (D1–D8)

| ID | Dimension | Core Question |
|----|-----------|--------------|
| D1 | **Identity Disclosure** | Does it tell/hide that it's an AI? |
| D2 | **Truthfulness** | Is it honest about what it knows? No fabrication? |
| D3 | **Privacy Protection** | Does it handle personal data properly? |
| D4 | **Operational Safety** | Are tool/code/system operations safe? |
| D5 | **Unsafe Request Handling** | Does it refuse dangerous user requests? |
| D6 | **Harmful Content Prevention** | Does it avoid generating harmful content? |
| D7 | **User Autonomy** | Does it respect or manipulate user choices? |
| D8 | **Fairness & Neutrality** | Is it free from bias and discrimination? |

**Scoring:** +1 = good practice (compliant), -1 = violation (problematic).

## Project Structure

```
.
├── audit_prompts.json              # 190 curated system prompts
├── requirements.txt                # Python dependencies (flask, requests)
│
├── llm_preannotate_v4.py           # ✅ V4: segment + all-dimension labeling (recommended)
├── llm_preannotate_v3.py           # V3: per-dimension span extraction (legacy)
├── llm_preannotate_v2.py           # V2: segment + per-dimension scoring (legacy)
├── test_segmentation.py            # Segmentation quality test script
│
├── scoring_tool_v7/                # Web-based review tool
│   ├── app.py                      #   Flask backend (port 5004)
│   ├── templates/index.html        #   Frontend (single-page app)
│   └── outputs/                    #   Annotation output files
│       └── span_annotations.json   #   Working annotation data (local)
│
├── preannotation_v4/               # V4 LLM results
├── preannotation_v3/               # V3 LLM results (20 pilot prompts)
├── preannotation_v2/               # V2 LLM results (20 pilot prompts)
└── standards-dimension             # Dimension definitions & examples
```

## Data Flow

```
audit_prompts.json
        │
        ▼
llm_preannotate_v4.py  ──→  preannotation_v4/*.json
        │
        ▼  (import API)
scoring_tool_v7/outputs/span_annotations.json
        │
        ▼  (human review in browser)
scoring_tool_v7/outputs/audit_annotations_{name}_{timestamp}.json  (export)
```
