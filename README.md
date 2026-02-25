# System Prompt Annotation Tool

A toolkit for auditing AI system prompts across 9 safety/transparency dimensions (D1–D9), including LLM-assisted pre-annotation and a web-based human review tool.

## Quick Start — Scoring Tool (Human Review)

```bash
# 1. Clone the repo
git clone git@github.com:XiangningLin/annotation_tool.git
cd annotation_tool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the scoring tool
cd scoring_tool_v7
python app.py
```

Open **http://127.0.0.1:5004** in your browser.

- Click **📌 Annotated Only** in the sidebar to see the 20 pre-annotated prompts
- Click any prompt to view its text with annotated spans (colored underlines + dimension tags)
- Use the dimension filter dropdown to focus on a specific dimension
- Review, modify scores, add/delete spans as needed

## Project Structure

```
.
├── audit_prompts.json              # 190 curated system prompts for auditing
├── standards-dimension             # D1–D9 audit dimension definitions & examples
├── requirements.txt                # Python dependencies
│
├── scoring_tool_v7/                # Web-based annotation/review tool
│   ├── app.py                      #   Flask backend
│   ├── templates/index.html        #   Frontend (single-page app)
│   └── outputs/                    #   Annotation output files
│       └── span_annotations.json   #   Human-reviewed span annotations
│
├── llm_preannotate_pilot.py        # Pilot: single-pass LLM annotation
├── llm_preannotate_v2.py           # V2: segment-then-score approach
├── llm_preannotate_v3.py           # V3: direct per-dimension span extraction
├── llm_batch_preannotate.py        # Batch runner for v2
│
├── preannotation_v2/               # V2 LLM results (20 pilot prompts)
├── preannotation_v3/               # V3 LLM results (20 pilot prompts)
├── pilot_results.json              # Pilot run results
│
├── unique_system_prompts_summary.csv   # Prompt metadata (all 359)
└── unique_system_prompts_summary.json  # Prompt metadata (all 359)
```

## LLM Pre-Annotation (Optional)

To run LLM pre-annotation on new prompts, you need an [OpenRouter](https://openrouter.ai/) API key:

```bash
export OPENROUTER_API_KEY="sk-or-..."

# Run on a single prompt
python llm_preannotate_v3.py --index 148

# Run batch on all 20 pilot prompts
python llm_preannotate_v3.py --batch

# Resume interrupted batch
python llm_preannotate_v3.py --batch --resume

# Dry run (show plan only, no API calls)
python llm_preannotate_v3.py --batch --dry-run
```

After running pre-annotation, import results into the scoring tool:
1. Start the scoring tool (`cd scoring_tool_v7 && python app.py`)
2. Call the import endpoint: `curl -X POST http://localhost:5004/api/import_v3_preannotations`

## Audit Dimensions (D1–D9)

| ID | Dimension | Focus |
|----|-----------|-------|
| D1 | Identity Transparency | AI identity disclosure |
| D2 | Knowledge Transparency | Knowledge limits & uncertainty |
| D3 | Information Accuracy & Source Integrity | No fabrication, cite sources |
| D4 | Privacy & Data Protection | PII handling, data transparency |
| D5 | Tool/Action Safety | Validate before execute, least-privilege |
| D6 | Manipulation Prevention | No dark patterns or psychological manipulation |
| D7 | Unsafe Request Handling | Refuse unsafe/illicit requests |
| D8 | Harm Prevention & User Safety | De-escalate, warn, refer to help |
| D9 | Fairness, Inclusion & Neutrality | No discrimination, political balance |

See `standards-dimension` for full definitions with positive/negative examples.

