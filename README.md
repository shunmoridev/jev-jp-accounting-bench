# jev-jp-accounting-bench

Japanese accounting / financial-reasoning benchmarks adapted for Jev-compatible System One decision models.

This repository currently evaluates:

- **JMMLU `professional_accounting`** — direct multiple-choice evaluation with Jev `choice`.
- **jfinqa** — adapted **answer-selection** evaluation. jfinqa is originally a free-form financial numerical reasoning benchmark; because Jev does not generate arbitrary free text, this repository converts each item into a deterministic candidate-selection task and evaluates it with Jev `choice`.

> This is **not** an official JMMLU or jfinqa evaluation harness. In particular, the jfinqa score produced here is not directly comparable with the canonical jfinqa exact/numerical-match score.

## Why

Jev is designed for typed decisions rather than prose generation. Accounting is a useful test domain because many tasks have objectively checkable answers while still requiring domain knowledge and numerical reasoning.

The benchmark intentionally keeps the two protocols separate:

| Benchmark | Original task | Jev protocol | Primary metric |
|---|---|---|---|
| JMMLU / professional_accounting | 4-way multiple choice | direct `choice` | accuracy |
| jfinqa | free-form financial QA | deterministic answer selection | accuracy |

## Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .
```

Configure a Jev-compatible System One endpoint:

```bash
# TypeSafe Jev
export JEV_API_KEY="..."
export JEV_BASE_URL="https://api.typesafe.ai"
export JEV_MODEL="jev-latest"
```

PowerShell:

```powershell
$env:JEV_API_KEY="..."
$env:JEV_BASE_URL="https://api.typesafe.ai"
$env:JEV_MODEL="jev-latest"
```

Any compatible implementation exposing `POST /v1/systemone` can also be used by changing `JEV_BASE_URL` and `JEV_MODEL`.

## Run

Smoke test with a small sample:

```bash
python bench.py jmmlu --limit 10
python bench.py jfinqa --limit 10
```

Full runs:

```bash
python bench.py jmmlu
python bench.py jfinqa
python bench.py all
```

Useful options:

```bash
python bench.py jfinqa --subtask numerical_reasoning --limit 50 --seed 42
python bench.py all --output results/run.json
python bench.py jmmlu --dry-run --limit 3
python bench.py all --base-url http://127.0.0.1:8000 --model openjev-latest
python bench.py jmmlu --log logs/my-run.jsonl   # custom request/response trace path
python bench.py jmmlu --no-log                  # disable request/response logging
python bench.py jmmlu --batch-size 4            # pack 4 items into one request (q1..q4)
```

`--batch-size N` concatenates N items into a single `state` (delimited by `【項目k】`) and asks `q1`..`qN` in one call, cutting the number of HTTP requests by N. This changes the model input, so batched accuracy is not directly comparable with `--batch-size 1` runs; `batch_size` is recorded per result in `meta` and in the top-level result JSON.

`--group-context` (jfinqa) is semantic aggregation instead of concatenation: jfinqa questions sharing the same evidence (same rendered context / `source_doc_id`) are grouped, the shared `資料` is sent once as `state`, and each question text is embedded in its own question's `instructions` as `質問: ...`. `--group-max` caps questions per request (default 10). JMMLU items have no shared context and remain singleton requests.

Use `python bench.py --help` for the complete CLI.

## jfinqa adaptation

Canonical jfinqa expects a model to produce an arbitrary answer string. Jev instead selects among typed alternatives, so this harness creates answer candidates deterministically:

1. The canonical gold answer is always included.
2. For numeric answers, plausible numeric distractors are generated while preserving the answer's surrounding text/unit where possible.
3. If additional distractors are required, answers from the same jfinqa subtask are used.
4. Candidate order is shuffled deterministically from `--seed` and the question ID.
5. The candidate text is placed in Jev `choice.criteria`; the model returns only the candidate label.

This protocol measures **financial answer selection**, not canonical free-form jfinqa generation. Results therefore report a separate protocol name.

## Results

By default results are written under `results/`. Result JSON contains benchmark metadata and per-item labels/probabilities, but intentionally does **not** copy the upstream question text or tables.

Raw request/response bodies (including the full `state` sent to the endpoint) are written to a JSONL trace under `logs/` for local debugging. Both `results/` and `logs/` are git-ignored; do not publish trace logs, since they contain dataset text.

Example summary:

```text
JMMLU professional_accounting: 112/150 = 74.67%
jfinqa answer-selection:       731/1000 = 73.10%
```

## Dataset provenance and licenses

This repository does **not** vendor or redistribute the benchmark datasets. They are fetched from their upstream sources at runtime.

### JMMLU

- Project: **JMMLU: Japanese Massive Multitask Language Understanding Benchmark**
- Upstream: https://github.com/nlp-waseda/JMMLU
- File used: `JMMLU/professional_accounting.csv`
- Upstream license for the `JMMLU/` tasks, including `professional_accounting`: **CC BY-SA 4.0**
- This harness does not use the separately licensed `JMMLU_NC_ND/` tasks.

### jfinqa

- Project: **jfinqa: Japanese Financial Numerical Reasoning QA Benchmark**
- Upstream: https://github.com/ajtgjmdjp/jfinqa
- Dataset: https://huggingface.co/datasets/ajtgjmdjp/jfinqa
- Upstream license: **Apache License 2.0**
- jfinqa documents its underlying financial source data as EDINET data under Japan's **Public Data License 1.0**.

See [THIRD_PARTY.md](THIRD_PARTY.md) for details.

## Repository license

The benchmark runner, adapters, documentation, and other original code in **this repository** are licensed under the **Apache License 2.0**. Upstream datasets remain under their own licenses; the repository license does not relicense them.

## Reproducibility notes

- Record `--model`, `--seed`, benchmark version, and result JSON together.
- For publication-quality comparisons, pin the upstream dataset revisions as well as this repository commit.
- jfinqa's canonical benchmark and this answer-selection adaptation answer different research questions; do not put their accuracy values in the same leaderboard column.
