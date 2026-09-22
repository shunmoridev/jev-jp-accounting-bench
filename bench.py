#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import httpx

JMMLU_URL = (
    "https://raw.githubusercontent.com/nlp-waseda/JMMLU/main/"
    "JMMLU/professional_accounting.csv"
)
DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_NUM_RE = re.compile(
    r"(?P<num>[+-]?(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?)"
)


@dataclass(frozen=True)
class Case:
    benchmark: str
    protocol: str
    item_id: str
    state: str
    instructions: str
    criteria: dict[str, str]
    gold_label: str
    meta: dict[str, Any]


def stable_rng(seed: int, item_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def download_cached(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "jev-jp-accounting-bench/0.1"})
    with urlopen(req, timeout=60) as response:
        data = response.read()
    path.write_bytes(data)
    return path


def load_jmmlu(cache_dir: Path, limit: int | None) -> list[Case]:
    csv_path = download_cached(
        JMMLU_URL, cache_dir / "jmmlu" / "professional_accounting.csv"
    )
    cases: list[Case] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for index, row in enumerate(csv.reader(f), start=1):
            if not row:
                continue
            if len(row) != 6:
                raise ValueError(
                    f"Unexpected JMMLU row shape at row {index}: expected 6 columns, got {len(row)}"
                )
            question, a, b, c, d, gold = row
            gold = gold.strip().upper()
            if gold not in {"A", "B", "C", "D"}:
                raise ValueError(f"Unexpected JMMLU answer at row {index}: {gold!r}")
            cases.append(
                Case(
                    benchmark="jmmlu_professional_accounting",
                    protocol="direct_choice",
                    item_id=f"professional_accounting:{index:03d}",
                    state=question,
                    instructions=(
                        "次の専門会計の問題について、正しい選択肢を1つ選んでください。"
                        "criteria に示された A〜D のうち、問題への正答に対応するラベルを選択してください。"
                    ),
                    criteria={"A": a, "B": b, "C": c, "D": d},
                    gold_label=gold,
                    meta={"row": index},
                )
            )
            if limit is not None and len(cases) >= limit:
                break
    return cases


def _answer_signature(text: str) -> str:
    text = text.strip()
    if "%" in text or "％" in text:
        return "percent"
    if any(unit in text for unit in ("円", "ドル", "千", "百万", "億", "兆")):
        return "money_or_scale"
    if _NUM_RE.search(text):
        return "numeric"
    return "text"


def _render_number_like(original: str, value: float) -> str:
    match = _NUM_RE.search(original)
    if not match:
        return original

    token = match.group("num")
    decimals = len(token.split(".", 1)[1]) if "." in token else 0
    use_commas = "," in token

    if decimals == 0:
        rendered = str(int(round(value)))
    else:
        rendered = f"{value:.{decimals}f}"

    if use_commas:
        if "." in rendered:
            integer, frac = rendered.split(".", 1)
            rendered = f"{int(integer):,}.{frac}"
        else:
            rendered = f"{int(rendered):,}"

    return original[: match.start()] + rendered + original[match.end() :]


def numeric_distractors(gold: str) -> list[str]:
    match = _NUM_RE.search(gold)
    if not match:
        return []

    raw = match.group("num").replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return []

    decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
    if value == 0:
        candidate_values = [1.0, -1.0, 10.0]
    elif decimals == 0 and abs(value) < 10:
        candidate_values = [value + 1, value - 1, value * 2]
    else:
        candidate_values = [value * 0.9, value * 1.1, value * 1.25]

    out: list[str] = []
    for candidate_value in candidate_values:
        candidate = _render_number_like(gold, candidate_value)
        if candidate != gold and candidate not in out:
            out.append(candidate)
    return out


def build_jfinqa_candidates(
    *,
    gold: str,
    answer_pool: list[str],
    seed: int,
    item_id: str,
    max_candidates: int = 4,
) -> tuple[dict[str, str], str]:
    rng = stable_rng(seed, item_id)
    candidates = [gold]

    for candidate in numeric_distractors(gold):
        if candidate not in candidates:
            candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break

    signature = _answer_signature(gold)
    same_type = [
        answer
        for answer in answer_pool
        if answer != gold
        and answer not in candidates
        and _answer_signature(answer) == signature
    ]
    rng.shuffle(same_type)
    for answer in same_type:
        candidates.append(answer)
        if len(candidates) >= max_candidates:
            break

    if len(candidates) < min(2, max_candidates):
        fallback = [
            answer
            for answer in answer_pool
            if answer != gold and answer not in candidates
        ]
        rng.shuffle(fallback)
        for answer in fallback:
            candidates.append(answer)
            if len(candidates) >= max_candidates:
                break

    if len(candidates) < 2:
        raise ValueError(
            f"Could not construct at least two candidates for jfinqa item {item_id}"
        )

    candidates = candidates[:max_candidates]
    rng.shuffle(candidates)

    criteria = {LABELS[i]: value for i, value in enumerate(candidates)}
    gold_label = next(label for label, value in criteria.items() if value == gold)
    return criteria, gold_label


def load_jfinqa_cases(
    *,
    subtask: str | None,
    limit: int | None,
    seed: int,
) -> list[Case]:
    try:
        from jfinqa import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "jfinqa is not installed. Run: pip install -e ."
        ) from exc

    questions = load_dataset(subtask) if subtask else load_dataset()

    pools: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        st = q.subtask.value
        answer = q.qa.answer.strip()
        if answer not in pools[st]:
            pools[st].append(answer)

    cases: list[Case] = []
    for q in questions:
        st = q.subtask.value
        gold = q.qa.answer.strip()
        criteria, gold_label = build_jfinqa_candidates(
            gold=gold,
            answer_pool=pools[st],
            seed=seed,
            item_id=q.id,
        )

        context = q.format_context()
        state_parts = []
        if context:
            state_parts.append("資料:\n" + context)
        state_parts.append("質問:\n" + q.qa.question)
        state = "\n\n".join(state_parts)

        cases.append(
            Case(
                benchmark="jfinqa",
                protocol="answer_selection_v1",
                item_id=q.id,
                state=state,
                instructions=(
                    "資料と質問に基づいて正しい回答候補を1つ選んでください。"
                    "criteria の各ラベルには回答候補が入っています。"
                    "計算が必要な場合は資料の数値関係を考慮し、正答に対応するラベルを選択してください。"
                ),
                criteria=criteria,
                gold_label=gold_label,
                meta={"subtask": st},
            )
        )
        if limit is not None and len(cases) >= limit:
            break

    return cases


class JevClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: float,
    ) -> None:
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def evaluate(self, http: httpx.Client, case: Case) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "state": case.state,
            "questions": {
                "answer": {
                    "type": "choice",
                    "instructions": case.instructions,
                    "criteria": case.criteria,
                }
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start = time.perf_counter()
        response = http.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        response.raise_for_status()
        body = response.json()

        answer = body.get("answers", {}).get("answer")
        if not isinstance(answer, dict):
            raise ValueError(
                f"Malformed System One response for {case.item_id}: missing answers.answer"
            )

        predicted = answer.get("choice")
        if predicted not in case.criteria:
            raise ValueError(
                f"Malformed choice for {case.item_id}: {predicted!r}; "
                f"expected one of {list(case.criteria)}"
            )

        probabilities = answer.get("probabilities")
        confidence = answer.get("confidence")
        if confidence is None and isinstance(probabilities, dict):
            try:
                confidence = float(probabilities[predicted])
            except (KeyError, TypeError, ValueError):
                confidence = None

        return {
            "item_id": case.item_id,
            "benchmark": case.benchmark,
            "protocol": case.protocol,
            "predicted": predicted,
            "gold": case.gold_label,
            "correct": predicted == case.gold_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "latency_ms": round(latency_ms, 2),
            "meta": case.meta,
        }


def evaluate_cases(
    *,
    cases: list[Case],
    client: JevClient,
    workers: int,
) -> list[dict[str, Any]]:
    if not cases:
        return []

    results: list[dict[str, Any] | None] = [None] * len(cases)

    def run_one(index: int, case: Case) -> tuple[int, dict[str, Any]]:
        with httpx.Client() as http:
            try:
                return index, client.evaluate(http, case)
            except Exception as exc:
                return index, {
                    "item_id": case.item_id,
                    "benchmark": case.benchmark,
                    "protocol": case.protocol,
                    "predicted": None,
                    "gold": case.gold_label,
                    "correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "meta": case.meta,
                }

    if workers <= 1:
        for i, case in enumerate(cases):
            _, result = run_one(i, case)
            results[i] = result
            status = "OK" if "error" not in result else "ERR"
            print(
                f"[{i + 1:>4}/{len(cases)}] {status} "
                f"{case.item_id}: {result.get('predicted')} / {case.gold_label}"
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_one, i, case): (i, case)
                for i, case in enumerate(cases)
            }
            completed = 0
            for future in as_completed(futures):
                index, case = futures[future]
                _, result = future.result()
                results[index] = result
                completed += 1
                status = "OK" if "error" not in result else "ERR"
                print(
                    f"[{completed:>4}/{len(cases)}] {status} "
                    f"{case.item_id}: {result.get('predicted')} / {case.gold_label}"
                )

    return [result for result in results if result is not None]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(result["benchmark"], result["protocol"])].append(result)

    summary: dict[str, Any] = {}
    for (benchmark, protocol), rows in groups.items():
        successful = [row for row in rows if "error" not in row]
        correct = sum(bool(row["correct"]) for row in successful)
        errors = len(rows) - len(successful)
        accuracy = correct / len(successful) if successful else 0.0
        latencies = [
            float(row["latency_ms"])
            for row in successful
            if row.get("latency_ms") is not None
        ]
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        key = f"{benchmark}:{protocol}"
        summary[key] = {
            "benchmark": benchmark,
            "protocol": protocol,
            "requested": len(rows),
            "evaluated": len(successful),
            "correct": correct,
            "errors": errors,
            "accuracy": accuracy,
            "avg_latency_ms": round(avg_latency, 2) if avg_latency is not None else None,
        }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\nResults")
    print("=" * 72)
    for row in summary.values():
        print(
            f"{row['benchmark']} [{row['protocol']}]: "
            f"{row['correct']}/{row['evaluated']} = {row['accuracy']:.2%} "
            f"(errors={row['errors']}, avg={row['avg_latency_ms']} ms)"
        )


def dry_run(cases: list[Case]) -> None:
    for case in cases:
        preview = {
            "item_id": case.item_id,
            "benchmark": case.benchmark,
            "protocol": case.protocol,
            "state": case.state,
            "question": {
                "type": "choice",
                "instructions": case.instructions,
                "criteria": case.criteria,
            },
            "gold_label": case.gold_label,
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JMMLU professional_accounting and jfinqa against a Jev-compatible System One endpoint."
    )
    parser.add_argument(
        "benchmark",
        choices=("jmmlu", "jfinqa", "all"),
        help="Benchmark to run.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum items per benchmark.")
    parser.add_argument(
        "--subtask",
        choices=("numerical_reasoning", "consistency_checking", "temporal_reasoning"),
        default=None,
        help="jfinqa subtask filter.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Candidate-generation seed.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("JEV_BASE_URL", DEFAULT_BASE_URL),
        help="System One API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("JEV_API_KEY"),
        help="API key. Defaults to JEV_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("JEV_MODEL", DEFAULT_MODEL),
        help="Model ID. Defaults to JEV_MODEL or jev-latest.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent requests. Start with 1; increase if your endpoint permits it.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
        help="Local upstream dataset cache.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Result JSON path. Default: results/<timestamp>.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print benchmark requests without calling Jev.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be > 0")
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")
    if args.subtask and args.benchmark == "jmmlu":
        raise SystemExit("--subtask only applies to jfinqa")

    cases: list[Case] = []
    if args.benchmark in ("jmmlu", "all"):
        cases.extend(load_jmmlu(args.cache_dir, args.limit))
    if args.benchmark in ("jfinqa", "all"):
        cases.extend(
            load_jfinqa_cases(
                subtask=args.subtask,
                limit=args.limit,
                seed=args.seed,
            )
        )

    if args.dry_run:
        dry_run(cases)
        return 0

    client = JevClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
    )
    results = evaluate_cases(cases=cases, client=client, workers=args.workers)
    summary = summarize(results)
    print_summary(summary)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("results") / f"jev-accounting-{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "benchmark": args.benchmark,
        "subtask": args.subtask,
        "summary": summary,
        "results": results,
        "notes": {
            "jmmlu": "Direct multiple-choice protocol on JMMLU professional_accounting.",
            "jfinqa": (
                "Adapted answer-selection protocol; not directly comparable "
                "to canonical free-form jfinqa scores."
            ),
        },
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {output}")

    return 1 if any(row["errors"] for row in summary.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
