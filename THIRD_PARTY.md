# Third-party datasets and attribution

This repository contains an evaluation harness. It does **not** vendor or redistribute the benchmark datasets described below. Dataset content is obtained from the upstream sources at runtime and remains subject to the upstream license and terms.

## JMMLU — professional_accounting

- Name: **JMMLU: Japanese Massive Multitask Language Understanding Benchmark**
- Upstream repository: https://github.com/nlp-waseda/JMMLU
- File used by this harness: `JMMLU/professional_accounting.csv`
- Upstream license for the `JMMLU/` dataset directory: **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**
- License text: https://creativecommons.org/licenses/by-sa/4.0/
- Upstream repository license file: https://github.com/nlp-waseda/JMMLU/blob/main/JMMLU/LICENSE

JMMLU also contains a separate `JMMLU_NC_ND/` directory with different licensing. This harness does not load those tasks.

The JMMLU `professional_accounting` questions are used without being committed to this repository. The runner downloads the upstream CSV into a local ignored cache.

## jfinqa

- Name: **jfinqa: Japanese Financial Numerical Reasoning QA Benchmark**
- Upstream repository: https://github.com/ajtgjmdjp/jfinqa
- Hugging Face dataset: https://huggingface.co/datasets/ajtgjmdjp/jfinqa
- Upstream license: **Apache License 2.0**
- Upstream license file: https://github.com/ajtgjmdjp/jfinqa/blob/main/LICENSE
- Upstream NOTICE: https://github.com/ajtgjmdjp/jfinqa/blob/main/NOTICE

The jfinqa project states that its underlying financial source data comes from EDINET, operated by the Financial Services Agency of Japan, and is provided under Japan's **Public Data License 1.0**:

- EDINET: https://disclosure.edinet-fsa.go.jp/
- Public Data License 1.0: https://www.digital.go.jp/resources/open_data/

This harness depends on the published `jfinqa` Python package / Hugging Face dataset rather than copying the dataset into this repository.

## Adaptation notice

jfinqa is canonically a free-form QA benchmark. Jev is a typed decision model and cannot generate arbitrary answer strings. This repository therefore creates a deterministic multiple-candidate answer-selection task from jfinqa examples.

That transformed protocol is specific to this repository and must not be represented as the canonical jfinqa benchmark score.

## Repository code license

Original code and documentation in this repository are licensed under Apache-2.0. That license applies only to this repository's original material. It does not replace, weaken, or relicense the licenses of the upstream datasets.
