# System Prompt Human Audit Paper — TODO List

> Generated: 2026-03-02

---

## A. Taxonomy & Framework

- [ ] Write clear definitions, positive/negative examples, and boundary conditions for D1–D8
- [ ] Map our dimensions to EU AI Act / NIST AI RMF / Universal Declaration of Human Rights requirements
- [ ] Differentiate from existing safety benchmarks (TrustLLM, SafetyBench, DecodingTrust) — they evaluate model outputs, we audit system prompts
- [ ] Clarify our motivation

## B. Dataset

- [ ] Describe data collection pipeline: selection criteria from TheBigPromptLibrary / system prompts GitHub, etc.
- [ ] Dataset statistics table: company distribution, category distribution, time span, size distribution
- [ ] Discuss ethical justification for using leaked prompts?

## C. Annotation Methodology

- [ ] Describe LLM pre-annotation prompt design (per-dimension extraction prompts)
- [ ] Justify why span-level rather than sentence-level or document-level
- [ ] Describe training phase workflow: joint annotation → IAA calibration → formal annotation
- [ ] Report training phase IAA scores
- [ ] Describe annotation tool design (screenshots + feature overview)?
- [ ] Describe review tool second-pass workflow

## D. LLM-as-Annotator Analysis

> *May not be included*

- [ ] Overall quality: per-dimension breakdown of 85.9% acceptance rate / 14.1% rejection rate
- [ ] False positive analysis: categorize the 815 rejected spans — what types of errors did the LLM make
- [ ] Miss analysis: categorize the 121 human-added spans — where did the LLM fail to detect
- [ ] Cross-model validation: per-dimension breakdown of Gemini 3.1 Pro results (exact match 36.7%, partial 54.4%)
- [ ] Add more models for cross-validation (GPT-5, Claude Sonnet, Llama, etc.), compare LLM reliability across dimensions

## E. Statistical Analysis

- [ ] Add significance tests to company rankings (bootstrap CI or permutation test)?
- [ ] Statistical tests for product category differences (chatbot vs coding agent vs specialized agent)
- [ ] Regression analysis of prompt size vs safety coverage (controlling for company, category, etc.)
- [ ] Statistical test for time trends (is the industry improving overall?)
- [ ] Dimension correlation matrix (not just co-occurrence, but Pearson/Spearman correlation)

## F. In-Depth Qualitative Analysis

- [ ] **Failure mode taxonomy**: systematically classify all ~796 negative spans into failure types (e.g., over-anthropomorphization, privacy opacity, safety boundary erosion, commercial manipulation, autonomy override, selective compliance)?
- [ ] **Case study: xAI** — what specifically changed from Grok 1 (58.3% neg) to Grok 4 (12.5% neg)
- [ ] **Case study: Meta** — why the regression from Llama 3 (1.9% neg) to Llama 4 WhatsApp (75% neg)
- [ ] **Case study: Anthropic** — why they rank best overall; analyze their prompt design philosophy
- [ ] **Case study: Poke/Venice** — extreme negative cases; how commercial interests erode safety
- [ ] **Design tension analysis**: trade-off cases of safety vs functionality, safety vs UX, safety vs business goals

## G. Automated Audit Model / Tool / Website

> *Technical contribution — website exists but no fine-tuned model yet*

- [ ] Task definition: given a span → predict (dimension, polarity), multi-label classification
- [ ] Zero-shot baseline: multiple LLMs (GPT-5, Claude, Gemini, Llama) zero-shot performance
- [ ] Few-shot baseline: in-context learning with a small set of labeled data
- [ ] Fine-tuned model: fine-tune a classifier on 5,072 labeled spans
- [ ] Evaluation metrics: per-dimension F1, macro F1, agreement with human annotations
- [ ] Error analysis: where does the automated model fail

## H. Discussion & Implications

- [ ] Argue that system prompts should be included in AI audit frameworks
- [ ] Transparency paradox: publishing prompts helps auditing but also helps jailbreaking
- [ ] Concrete recommendations for AI companies (per-dimension best practices)
- [ ] Recommendations for regulators: standardized audit framework
- [ ] Limitations: representativeness of leaked data, annotation subjectivity, sample bias, temporal snapshot, single source

## I. Releasable Artifacts

- [ ] Open-source annotated dataset (PromptSafetyCorpus)
- [ ] Open-source annotation tool
- [ ] Open-source taxonomy + annotation guideline
- [ ] Open-source automated audit model / website
