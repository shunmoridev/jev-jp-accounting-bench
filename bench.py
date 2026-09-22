#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import sys
import threading
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
    r"(?P<num>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
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
    context: str = ""
    question: str | None = None


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
                    context=question,
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
                context="資料:\n" + context if context else "",
                question=q.qa.question,
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
        independent_items: bool = False,
        group_context: bool = False,
    ) -> None:
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.independent_items = independent_items
        self.group_context = group_context

    def evaluate_batch(
        self, http: httpx.Client, cases: list[Case], trace: dict[str, Any]
    ) -> list[dict[str, Any]]:
        single = len(cases) == 1
        independence_note = (
            "各項目は互いに独立した問題です。他の項目の内容・選択を参照しないでください。"
            if self.independent_items and not single
            else ""
        )

        if self.group_context:
            # Shared evidence goes to `state` once; each question text lives in
            # its own question's instructions (semantic aggregation).
            state = cases[0].context or cases[0].state
        elif single:
            state = cases[0].state
        else:
            state = "\n\n".join(
                f"【項目{k}】\n{case.state}" for k, case in enumerate(cases, start=1)
            )

        questions: dict[str, dict[str, Any]] = {}
        qkeys: list[str] = []
        for k, case in enumerate(cases, start=1):
            qkey = "answer" if single else f"q{k}"
            instructions = independence_note
            if not single and not self.group_context:
                instructions += f"この設問は state 内の【項目{k}】に対応します。"
            instructions += case.instructions
            if self.group_context and case.question:
                instructions += f"\n質問: {case.question}"
            questions[qkey] = {
                "type": "choice",
                "instructions": instructions,
                "criteria": case.criteria,
            }
            qkeys.append(qkey)

        payload = {"model": self.model, "state": state, "questions": questions}
        trace["request"] = {"url": self.url, "payload": payload}
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
        trace["status"] = response.status_code
        trace["latency_ms"] = round(latency_ms, 2)
        if response.is_error:
            trace["response_text"] = response.text[:4000]
        response.raise_for_status()
        body = response.json()
        trace["response"] = body

        answers = body.get("answers", {})
        results: list[dict[str, Any]] = []
        for case, qkey in zip(cases, qkeys):
            answer = answers.get(qkey)
            if not isinstance(answer, dict):
                raise ValueError(
                    f"Malformed System One response for {case.item_id}: "
                    f"missing answers.{qkey}"
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

            results.append(
                {
                    "item_id": case.item_id,
                    "benchmark": case.benchmark,
                    "protocol": case.protocol,
                    "predicted": predicted,
                    "gold": case.gold_label,
                    "correct": predicted == case.gold_label,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "latency_ms": round(latency_ms, 2),
                    "meta": {**case.meta, "batch_size": len(cases)},
                }
            )
        return results


def evaluate_cases(
    *,
    cases: list[Case],
    client: JevClient,
    workers: int,
    batch_size: int = 1,
    group_context: bool = False,
    group_max: int = 10,
    log_file: Any | None = None,
) -> list[dict[str, Any]]:
    if not cases:
        return []

    indexed: list[tuple[int, Case]] = list(enumerate(cases))
    if group_context:
        groups: dict[str, list[tuple[int, Case]]] = {}
        for pair in indexed:
            groups.setdefault(pair[1].context or pair[1].state, []).append(pair)
        batches: list[list[tuple[int, Case]]] = []
        for members in groups.values():
            for i in range(0, len(members), group_max):
                batches.append(members[i : i + group_max])
    else:
        batches = [
            indexed[start : start + batch_size]
            for start in range(0, len(cases), batch_size)
        ]

    results: list[dict[str, Any] | None] = [None] * len(cases)
    log_lock = threading.Lock()

    def write_trace(trace: dict[str, Any]) -> None:
        if log_file is None:
            return
        with log_lock:
            log_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
            log_file.flush()

    def run_batch(
        http: httpx.Client, batch: list[tuple[int, Case]]
    ) -> tuple[list[tuple[int, Case]], list[dict[str, Any]], dict[str, Any]]:
        batch_cases = [case for _, case in batch]
        trace: dict[str, Any] = {"item_ids": [case.item_id for case in batch_cases]}
        try:
            return batch, client.evaluate_batch(http, batch_cases, trace), trace
        except Exception as exc:
            trace["error"] = f"{type(exc).__name__}: {exc}"
            return batch, [
                {
                    "item_id": case.item_id,
                    "benchmark": case.benchmark,
                    "protocol": case.protocol,
                    "predicted": None,
                    "gold": case.gold_label,
                    "correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "meta": {**case.meta, "batch_size": len(batch_cases)},
                }
                for _, case in batch
            ], trace

    completed = 0

    def report(batch: list[tuple[int, Case]], rows: list[dict[str, Any]]) -> None:
        nonlocal completed
        for (index, _), result in zip(batch, rows):
            results[index] = result
            completed += 1
            status = "OK" if "error" not in result else "ERR"
            print(
                f"[{completed:>4}/{len(cases)}] {status} "
                f"{result['item_id']}: {result.get('predicted')} / {result['gold']}"
            )

    with httpx.Client() as http:
        if workers <= 1:
            for batch in batches:
                _, rows, trace = run_batch(http, batch)
                write_trace(trace)
                report(batch, rows)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(run_batch, http, batch): batch for batch in batches
                }
                for future in as_completed(futures):
                    batch = futures[future]
                    _, rows, trace = future.result()
                    write_trace(trace)
                    report(batch, rows)

    return [result for result in results if result is not None]


def _prob_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    probs = [
        float(row["confidence"])
        for row in rows
        if row.get("confidence") is not None
    ]
    if not probs:
        return {"median": None, "std": None, "n": 0}
    return {
        "median": round(statistics.median(probs), 4),
        "std": round(statistics.pstdev(probs), 4),
        "n": len(probs),
    }


def _latency_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(
        float(row["latency_ms"])
        for row in rows
        if row.get("latency_ms") is not None
    )
    stats: dict[str, Any] = {
        "mean": None,
        "median": None,
        "p95": None,
        "min": None,
        "max": None,
    }
    if not values:
        return stats
    stats["mean"] = round(statistics.fmean(values), 2)
    stats["median"] = round(statistics.median(values), 2)
    stats["p95"] = round(
        statistics.quantiles(values, n=100, method="inclusive")[94]
        if len(values) >= 2
        else values[0],
        2,
    )
    stats["min"] = round(values[0], 2)
    stats["max"] = round(values[-1], 2)
    return stats


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(result["benchmark"], result["protocol"])].append(result)

    summary: dict[str, Any] = {}
    for (benchmark, protocol), rows in groups.items():
        successful = [row for row in rows if "error" not in row]
        correct_rows = [row for row in successful if row["correct"]]
        incorrect_rows = [row for row in successful if not row["correct"]]
        correct = len(correct_rows)
        errors = len(rows) - len(successful)
        accuracy = correct / len(successful) if successful else 0.0
        key = f"{benchmark}:{protocol}"
        summary[key] = {
            "benchmark": benchmark,
            "protocol": protocol,
            "requested": len(rows),
            "evaluated": len(successful),
            "correct": correct,
            "errors": errors,
            "accuracy": accuracy,
            "latency_ms": _latency_stats(successful),
            "selected_prob_when_correct": _prob_stats(correct_rows),
            "selected_prob_when_incorrect": _prob_stats(incorrect_rows),
        }
    return summary


def print_summary(
    summary: dict[str, Any], *, total_seconds: float, workers: int
) -> None:
    print("\nResults")
    print("=" * 72)
    for row in summary.values():
        latency = row["latency_ms"]
        latency_str = (
            f"mean={latency['mean']} p95={latency['p95']} ms"
            if latency["mean"] is not None
            else "n/a"
        )
        print(
            f"{row['benchmark']} [{row['protocol']}]: "
            f"{row['correct']}/{row['evaluated']} = {row['accuracy']:.2%} "
            f"(errors={row['errors']}, latency {latency_str})"
        )
        for key, label in (
            ("selected_prob_when_correct", "correct"),
            ("selected_prob_when_incorrect", "incorrect"),
        ):
            stats = row[key]
            if stats["n"]:
                print(
                    f"    chosen-prob {label}: median={stats['median']:.4f} "
                    f"std={stats['std']:.4f} (n={stats['n']})"
                )
    print(f"total={total_seconds:.2f}s workers={workers}")


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
        "--batch-size",
        type=int,
        default=1,
        help="Items per request. >1 packs N items into one state with questions q1..qN.",
    )
    parser.add_argument(
        "--independent-items",
        action="store_true",
        help="Tell the model batched items are independent (only affects --batch-size > 1).",
    )
    parser.add_argument(
        "--group-context",
        action="store_true",
        help=(
            "Semantic aggregation: share each jfinqa context once in state and "
            "put each question text into its own question's instructions. "
            "Overrides --batch-size."
        ),
    )
    parser.add_argument(
        "--group-max",
        type=int,
        default=10,
        help="Max questions per shared-context request (with --group-context).",
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
        help="Result JSON path. Default: results/<timestamp>-w<workers>-b<batch>.json",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Request/response JSONL log path. Default: logs/<timestamp>-w<workers>-b<batch>.jsonl",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable request/response logging.",
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
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.group_max <= 0:
        raise SystemExit("--group-max must be > 0")
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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    log_file = None
    log_path = None
    if not args.no_log:
        log_path = args.log or Path("logs") / (
            f"jev-trace-{timestamp}-w{args.workers}-b{args.batch_size}.jsonl"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")

    client = JevClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        independent_items=args.independent_items,
        group_context=args.group_context,
    )
    eval_start = time.perf_counter()
    try:
        results = evaluate_cases(
            cases=cases,
            client=client,
            workers=args.workers,
            batch_size=args.batch_size,
            group_context=args.group_context,
            group_max=args.group_max,
            log_file=log_file,
        )
    finally:
        total_seconds = time.perf_counter() - eval_start
        if log_file is not None:
            log_file.close()
    if log_path is not None:
        print(f"Wrote {log_path}")

    summary = summarize(results)
    print_summary(summary, total_seconds=total_seconds, workers=args.workers)

    output = args.output or Path("results") / (
        f"jev-accounting-{timestamp}-w{args.workers}-b{args.batch_size}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "benchmark": args.benchmark,
        "subtask": args.subtask,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "group_context": args.group_context,
        "group_max": args.group_max,
        "independent_items": args.independent_items,
        "total_time_s": round(total_seconds, 2),
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
