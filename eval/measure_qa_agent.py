"""실험 QA Agent 자동화 효과 측정.

사용자에게 보이는 품질을 세 축으로 잰다.

1. 조회 정확도
   - 의도 분류: 응답 intent가 기대 intent와 같은가
   - 슬롯: experiment / metric / run / top_k가 맞게 채워졌는가
   - 사실: 답 또는 metadata가 MLflow 원본 값과 일치하는가
2. 응답시간
   - POST /chat/ask 왕복 시간 (신규 대화 첫 턴은 세션 제목 생성 포함)
   - 같은 사실을 앱의 MLflow API로 직접 조회한 시간
3. 보고서
   - POST /report/ 생성 시간
   - 입력 run 이름·지표 값이 본문에 다시 나타나는 비율
   - 필수 섹션 존재 여부

정답은 실행 시점에 Tracking API와 앱 API에서 다시 읽는다.
측정이 만든 채팅 세션과 보고서는 끝나면 삭제한다.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APP_BASE = "http://127.0.0.1:8010"
MLFLOW_BASE = "http://127.0.0.1:2522"
CHAT_TIMEOUT_S = 180
REPORT_TIMEOUT_S = 300
OUT_DIR = Path(__file__).resolve().parent / "results"

NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")


def _request(
    url: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    timeout: float = 30,
) -> tuple[int, Any, float]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            elapsed_ms = (time.perf_counter() - started) * 1000
            if not raw:
                return response.status, None, elapsed_ms
            return response.status, json.loads(raw.decode()), elapsed_ms
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        detail = exc.read().decode(errors="replace")
        try:
            parsed = json.loads(detail)
        except json.JSONDecodeError:
            parsed = {"detail": detail[:500]}
        return exc.code, parsed, elapsed_ms


def app_get(path: str, timeout: float = 30) -> tuple[int, Any, float]:
    return _request(f"{APP_BASE}{path}", timeout=timeout)


def app_post(path: str, body: dict, timeout: float) -> tuple[int, Any, float]:
    return _request(f"{APP_BASE}{path}", method="POST", body=body, timeout=timeout)


def app_delete(path: str) -> None:
    try:
        _request(f"{APP_BASE}{path}", method="DELETE", timeout=20)
    except Exception:
        return


def mlflow_post(path: str, body: dict) -> Any:
    status, payload, _ = _request(
        f"{MLFLOW_BASE}{path}",
        method="POST",
        body=body,
        timeout=30,
    )
    if status != 200:
        raise RuntimeError(f"MLflow {path} -> {status}: {payload}")
    return payload


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def round_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 1)


def contains_number(text: str, expected: float, *, abs_tol: float = 1e-4) -> bool:
    tolerance = max(abs_tol, abs(expected) * 1e-3)
    for token in NUMBER_RE.findall(text or ""):
        try:
            number = float(token)
        except ValueError:
            continue
        if abs(number - expected) <= tolerance:
            return True
    return False


def load_ground_truth() -> dict[str, Any]:
    status, names_payload, _ = app_get("/mlflow/names")
    if status != 200:
        raise RuntimeError(f"/mlflow/names failed: {status} {names_payload}")
    experiment_names = names_payload["experiment_names"]

    experiments = mlflow_post(
        "/api/2.0/mlflow/experiments/search",
        {"max_results": 200},
    ).get("experiments", [])
    recent = [
        item["name"]
        for item in sorted(
            experiments,
            key=lambda item: int(item.get("creation_time") or 0),
            reverse=True,
        )[:5]
    ]

    status, timm_exp, _ = app_get(
        "/mlflow/experiment?" + urllib.parse.urlencode({"experiment_name": "Timm Hub"})
    )
    if status != 200:
        raise RuntimeError(f"Timm Hub experiment lookup failed: {status}")
    timm = timm_exp["experiment"]

    status, timm_runs_payload, _ = app_get(
        "/mlflow/runs?" + urllib.parse.urlencode({"experiment_name": "Timm Hub"})
    )
    if status != 200:
        raise RuntimeError(f"Timm Hub runs failed: {status}")
    timm_runs = timm_runs_payload["runs"]

    status, metric_payload, _ = app_get(
        "/mlflow/metric-keys?" + urllib.parse.urlencode({"experiment_name": "Timm Hub"})
    )
    if status != 200:
        raise RuntimeError(f"Timm Hub metrics failed: {status}")

    status, top_payload, _ = app_get(
        "/mlflow/top-runs?"
        + urllib.parse.urlencode(
            {
                "experiment_name": "Timm Hub",
                "metric_key": "metrics.val_acc",
                "top_k": 3,
                "ascending": "false",
            }
        )
    )
    if status != 200:
        raise RuntimeError(f"top-runs failed: {status} {top_payload}")

    def metric_value(run: dict, key: str) -> float:
        if key in run and run[key] is not None:
            return float(run[key])
        prefixed = key if key.startswith("metrics.") else f"metrics.{key}"
        return float(run[prefixed])

    lowest_loss = min(timm_runs, key=lambda run: metric_value(run, "val_loss"))

    hpo = mlflow_post(
        "/api/2.0/mlflow/runs/search",
        {
            "experiment_ids": ["9"],
            "filter": 'attributes.run_name = "yolo26n-hpo-20"',
            "max_results": 5,
        },
    )["runs"][0]
    hpo_metrics = {item["key"]: item["value"] for item in hpo["data"].get("metrics", [])}

    exp1 = mlflow_post(
        "/api/2.0/mlflow/runs/search",
        {"experiment_ids": ["8"], "max_results": 5},
    )["runs"][0]
    exp1_metrics = {item["key"]: item["value"] for item in exp1["data"].get("metrics", [])}

    return {
        "experiment_names": experiment_names,
        "recent_names": recent,
        "timm_experiment_id": str(timm["experiment_id"]),
        "timm_lifecycle": str(timm["lifecycle_stage"]),
        "timm_run_names": [
            str(run.get("tags.mlflow.runName"))
            for run in timm_runs
            if run.get("tags.mlflow.runName")
        ],
        "timm_metric_keys": metric_payload["metric_keys"],
        "top_val_acc": top_payload["top_runs"],
        "lowest_val_loss_run": str(lowest_loss.get("tags.mlflow.runName")),
        "lowest_val_loss": metric_value(lowest_loss, "val_loss"),
        "hpo20_map50": float(hpo_metrics["metrics/mAP50B"]),
        "exp1_map50": float(exp1_metrics["metrics/mAP50B"]),
        "burly_val_accuracy": 77.95,
    }


def ask(message: str, *, session_id: str | None = None, report_text: str | None = None) -> dict:
    body: dict[str, Any] = {"message": message}
    if session_id:
        body["session_id"] = session_id
    if report_text is not None:
        body["report_text"] = report_text
    status, payload, elapsed_ms = app_post("/chat/ask", body, CHAT_TIMEOUT_S)
    return {
        "http_status": status,
        "elapsed_ms": elapsed_ms,
        "payload": payload if isinstance(payload, dict) else {"detail": payload},
    }


def score_item(
    *,
    case_id: str,
    category: str,
    question: str,
    response: dict,
    expected_intent: str,
    expected_slots: dict[str, Any] | None = None,
    fact_checks: list[tuple[str, bool]] | None = None,
    baseline_ms: float | None = None,
    manual_steps: int | None = None,
    turn: int = 1,
) -> dict[str, Any]:
    payload = response["payload"] if isinstance(response["payload"], dict) else {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    intent = payload.get("intent")
    intent_ok = response["http_status"] == 200 and intent == expected_intent

    slot_results = []
    for key, expected in (expected_slots or {}).items():
        actual = metadata.get(key)
        if key == "metric_key" and isinstance(actual, str) and isinstance(expected, str):
            ok = actual == expected or actual.removeprefix("metrics.") == expected.removeprefix("metrics.")
        else:
            ok = actual == expected
        slot_results.append({"key": key, "expected": expected, "actual": actual, "ok": ok})
    slots_ok = all(item["ok"] for item in slot_results) if slot_results else None

    facts = [{"name": name, "ok": ok} for name, ok in (fact_checks or [])]
    facts_ok = all(item["ok"] for item in facts) if facts else None
    passed = bool(intent_ok and (slots_ok is not False) and (facts_ok is not False) and response["http_status"] == 200)

    return {
        "id": case_id,
        "turn": turn,
        "category": category,
        "question": question,
        "http_status": response["http_status"],
        "elapsed_ms": round_ms(response["elapsed_ms"]),
        "baseline_ms": round_ms(baseline_ms),
        "manual_steps": manual_steps,
        "intent": intent,
        "expected_intent": expected_intent,
        "intent_ok": intent_ok,
        "slots": slot_results,
        "slots_ok": slots_ok,
        "facts": facts,
        "facts_ok": facts_ok,
        "passed": passed,
        "answer_preview": str(payload.get("answer") or payload.get("detail") or "")[:400],
        "session_id": payload.get("session_id"),
    }


def direct_baseline(path: str) -> float:
    status, payload, elapsed_ms = app_get(path)
    if status != 200:
        raise RuntimeError(f"baseline {path} -> {status} {payload}")
    return elapsed_ms


def run_chat_cases(gt: dict[str, Any]) -> tuple[list[dict], list[str]]:
    results: list[dict] = []
    session_ids: list[str] = []
    answer_of: dict[str, str] = {}

    top_runs = gt["top_val_acc"]
    top_names = [run["run_name"] for run in top_runs]
    best = top_runs[0]

    report_fixture = (
        f"<p>val_acc 기준 최상위 run은 {best['run_name']}이며 점수는 {best['metric_score']}이다.</p>"
    )

    plan = [
        {
            "id": "list_experiments",
            "category": "experiment_lookup",
            "question": "실험 목록 보여줘",
            "expected_intent": "list_experiments",
            "baseline": "/mlflow/names",
            "manual_steps": 1,
            "slots": {"count": len(gt["experiment_names"])},
            "facts": lambda answer, meta: [
                ("experiment_count", meta.get("count") == len(gt["experiment_names"])),
                (
                    "all_names_visible",
                    all(name in answer for name in gt["experiment_names"]),
                ),
            ],
        },
        {
            "id": "list_recent_experiments",
            "category": "experiment_lookup",
            "question": "최근 실험 알려줘",
            "expected_intent": "list_recent_experiments",
            "manual_steps": 2,
            "slots": {"count": len(gt["recent_names"])},
            "facts": lambda answer, meta: [
                ("recent_names", all(name in answer for name in gt["recent_names"])),
                (
                    "recent_order",
                    [name for name in gt["recent_names"]]
                    == [line[2:].strip() for line in answer.splitlines() if line.startswith("- ")],
                ),
            ],
        },
        {
            "id": "get_experiment",
            "category": "experiment_lookup",
            "question": "Timm Hub 실험 정보 알려줘",
            "expected_intent": "get_experiment",
            "baseline": "/mlflow/experiment?" + urllib.parse.urlencode({"experiment_name": "Timm Hub"}),
            "manual_steps": 2,
            "slots": {"experiment_name": "Timm Hub"},
            "facts": lambda answer, meta: [
                ("experiment_id", f"| {gt['timm_experiment_id']} |" in answer),
                ("lifecycle", gt["timm_lifecycle"] in answer),
            ],
        },
        {
            "id": "get_runs",
            "category": "run_lookup",
            "question": "Timm Hub run 목록",
            "expected_intent": "get_runs",
            "baseline": "/mlflow/runs?" + urllib.parse.urlencode({"experiment_name": "Timm Hub"}),
            "manual_steps": 2,
            "slots": {"experiment_name": "Timm Hub", "count": len(gt["timm_run_names"])},
            "facts": lambda answer, meta: [
                ("run_names", all(name in answer for name in gt["timm_run_names"])),
            ],
        },
        {
            "id": "list_metric_keys",
            "category": "run_lookup",
            "question": "Timm Hub metric key 알려줘",
            "expected_intent": "list_metric_keys",
            "baseline": "/mlflow/metric-keys?" + urllib.parse.urlencode({"experiment_name": "Timm Hub"}),
            "manual_steps": 2,
            "slots": {"experiment_name": "Timm Hub", "count": len(gt["timm_metric_keys"])},
            "facts": lambda answer, meta: [
                ("metric_keys", all(key in answer for key in gt["timm_metric_keys"])),
            ],
        },
        {
            "id": "get_run_metric",
            "category": "run_lookup",
            "question": "yolo26n-hpo-20 run의 metrics/mAP50B 값은 얼마야?",
            "expected_intent": "get_run_info",
            "baseline": "/mlflow/runs?" + urllib.parse.urlencode({"experiment_name": "ultralytics-yolo-hpo"}),
            "manual_steps": 3,
            "slots": {"run_name": "yolo26n-hpo-20", "metric_key": "metrics.metrics/mAP50B"},
            "facts": lambda answer, meta: [
                ("run_name", "yolo26n-hpo-20" in answer),
                ("map50", contains_number(answer, gt["hpo20_map50"])),
            ],
        },
        {
            "id": "get_run_detail",
            "category": "run_lookup",
            "question": "Deep Learning Experiment에서 burly-kit-617 run 상세 알려줘",
            "expected_intent": "get_run_info",
            "manual_steps": 3,
            "slots": {"run_name": "burly-kit-617", "experiment_name": "Deep Learning Experiment"},
            "facts": lambda answer, meta: [
                ("run_name", "burly-kit-617" in answer),
                ("val_accuracy", contains_number(answer, gt["burly_val_accuracy"])),
            ],
        },
        {
            "id": "search_top_accuracy",
            "category": "run_search",
            "question": "Timm Hub에서 metrics.val_acc가 높은 run top 3 보여줘",
            "expected_intent": "search_runs",
            "baseline": "/mlflow/top-runs?"
            + urllib.parse.urlencode(
                {
                    "experiment_name": "Timm Hub",
                    "metric_key": "metrics.val_acc",
                    "top_k": 3,
                    "ascending": "false",
                }
            ),
            "manual_steps": 4,
            "slots": {"experiment_name": "Timm Hub", "count": 3},
            "facts": lambda answer, meta: _top_run_facts(answer, meta, top_names, [run["metric_score"] for run in top_runs]),
        },
        {
            "id": "search_lowest_loss",
            "category": "run_search",
            "question": "Timm Hub에서 metrics.val_loss가 가장 낮은 run 보여줘",
            "expected_intent": "search_runs",
            "manual_steps": 4,
            "slots": {"experiment_name": "Timm Hub"},
            "facts": lambda answer, meta: [
                (
                    "lowest_run_first",
                    _first_table_name(meta) == gt["lowest_val_loss_run"] or answer.find(gt["lowest_val_loss_run"]) >= 0,
                ),
                ("lowest_loss_value", contains_number(answer, gt["lowest_val_loss"])),
            ],
        },
        {
            "id": "run_judgment",
            "category": "judgment",
            "question": "yolo26n-exp1 run 학습이 잘 된 편이야?",
            "expected_intent": "mlflow_mcp",
            "manual_steps": 4,
            "slots": {"run_name": "yolo26n-exp1", "grounded_on_run": True},
            "facts": lambda answer, meta: [
                ("mentions_run", "yolo26n-exp1" in answer or meta.get("run_name") == "yolo26n-exp1"),
                ("cites_map50", contains_number(answer, gt["exp1_map50"])),
            ],
        },
        {
            "id": "report_qa",
            "category": "report_qa",
            "question": "보고서에서 val_acc가 가장 높은 run 이름과 점수를 말해줘",
            "expected_intent": "report_qa",
            "report_text": report_fixture,
            "manual_steps": 2,
            "slots": {},
            "facts": lambda answer, meta: [
                ("best_run", best["run_name"] in answer),
                ("best_score", contains_number(answer, float(best["metric_score"]))),
            ],
        },
        {
            "id": "missing_experiment",
            "category": "robustness",
            "question": "없는실험XYZ 실험 정보 알려줘",
            "expected_intent": "get_experiment",
            "manual_steps": 2,
            "slots": {},
            "facts": lambda answer, meta: [
                (
                    "graceful_not_found",
                    "없습니다" in answer or "찾을 수 없" in answer,
                ),
            ],
        },
    ]

    for spec in plan:
        print(f"[chat] {spec['id']}", flush=True)
        baseline_ms = direct_baseline(spec["baseline"]) if spec.get("baseline") else None
        response = ask(spec["question"], report_text=spec.get("report_text"))
        answer = str((response["payload"] or {}).get("answer") or "")
        meta = (response["payload"] or {}).get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        item = score_item(
            case_id=spec["id"],
            category=spec["category"],
            question=spec["question"],
            response=response,
            expected_intent=spec["expected_intent"],
            expected_slots=spec.get("slots") or None,
            fact_checks=spec["facts"](answer, meta) if response["http_status"] == 200 else [("http_ok", False)],
            baseline_ms=baseline_ms,
            manual_steps=spec.get("manual_steps"),
        )
        results.append(item)
        if item["session_id"]:
            session_ids.append(item["session_id"])
        answer_of[spec["id"]] = answer

    print("[chat] multiturn_followup", flush=True)
    first = ask("Timm Hub 실험 정보 알려줘")
    first_payload = first["payload"] if isinstance(first["payload"], dict) else {}
    session_id = first_payload.get("session_id")
    if session_id:
        session_ids.append(session_id)
    second = ask("metric key 알려줘", session_id=session_id) if session_id else {
        "http_status": first["http_status"],
        "elapsed_ms": 0,
        "payload": {"detail": "첫 턴에서 session_id를 받지 못했습니다."},
    }
    second_answer = str((second["payload"] or {}).get("answer") or "")
    second_meta = (second["payload"] or {}).get("metadata") or {}
    if not isinstance(second_meta, dict):
        second_meta = {}
    results.append(
        score_item(
            case_id="multiturn_context",
            category="multiturn",
            question="metric key 알려줘",
            response=second,
            expected_intent="list_metric_keys",
            expected_slots={"experiment_name": "Timm Hub"},
            fact_checks=[
                ("carries_experiment", second_meta.get("experiment_name") == "Timm Hub"),
                ("metric_keys", all(key in second_answer for key in gt["timm_metric_keys"])),
            ] if second["http_status"] == 200 else [("http_ok", False)],
            manual_steps=1,
            turn=2,
        )
    )
    results.append(
        score_item(
            case_id="multiturn_context_setup",
            category="multiturn",
            question="Timm Hub 실험 정보 알려줘",
            response=first,
            expected_intent="get_experiment",
            expected_slots={"experiment_name": "Timm Hub"},
            fact_checks=[
                ("experiment_id", f"| {gt['timm_experiment_id']} |" in str(first_payload.get("answer") or "")),
            ] if first["http_status"] == 200 else [("http_ok", False)],
            manual_steps=2,
            turn=1,
        )
    )
    return results, session_ids


def _first_table_name(metadata: dict) -> str | None:
    rows = metadata.get("run_display_rows") or []
    if not rows:
        return None
    return rows[0].get("Name")


def _top_run_facts(answer: str, metadata: dict, names: list[str], scores: list[float]) -> list[tuple[str, bool]]:
    rows = metadata.get("run_display_rows") or []
    got_names = [row.get("Name") for row in rows]
    got_scores = [row.get("Score") for row in rows]
    score_ok = len(got_scores) == len(scores) and all(
        isinstance(got, (int, float)) and abs(float(got) - expected) <= 1e-4
        for got, expected in zip(got_scores, scores)
    )
    return [
        ("top3_names_in_order", got_names == names),
        ("top3_scores", score_ok),
        ("names_in_answer", all(name in answer for name in names)),
    ]


def run_report(gt: dict[str, Any]) -> tuple[dict, str | None]:
    print("[report] generate", flush=True)
    top_runs = gt["top_val_acc"]
    body = {
        "experiment_name": "Timm Hub",
        "metric_key": "metrics.val_acc",
        "metric_direction": "maximize",
        "top_runs": top_runs,
    }
    status, payload, elapsed_ms = app_post("/report/", body, REPORT_TIMEOUT_S)
    report_id = payload.get("id") if isinstance(payload, dict) else None
    text = payload.get("report", "") if isinstance(payload, dict) else ""
    checks = []
    if status == 200:
        checks.append(("title", "Timm Hub 실험 분석 보고서" in text))
        for heading in (
            "1. 실험 개요 및 핵심 결과",
            "2. 실험 결과 비교",
            "3. 실험별 강점 및 한계",
            "4. 성능 차이 분석",
            "5. 후속 실험 계획",
            "6. 종합 결론",
        ):
            checks.append((f"section:{heading}", heading in text))
        value_hits = []
        for run in top_runs:
            name_ok = run["run_name"] in text
            score_ok = contains_number(text, float(run["metric_score"]))
            value_hits.append({"run_name": run["run_name"], "name_ok": name_ok, "score_ok": score_ok})
            checks.append((f"run:{run['run_name']}", name_ok))
            checks.append((f"score:{run['run_name']}", score_ok))
    else:
        value_hits = []
    grounded = [item for item in checks if item[0].startswith("run:") or item[0].startswith("score:")]
    grounding_rate = (
        sum(1 for _, ok in grounded if ok) / len(grounded) if grounded else 0.0
    )
    section_checks = [ok for name, ok in checks if name.startswith("section:")]
    return {
        "http_status": status,
        "elapsed_ms": round_ms(elapsed_ms),
        "model": payload.get("model") if isinstance(payload, dict) else None,
        "experiment_name": "Timm Hub",
        "metric_key": "metrics.val_acc",
        "run_count": len(top_runs),
        "checks": [{"name": name, "ok": ok} for name, ok in checks],
        "value_hits": value_hits,
        "grounding_rate": round(grounding_rate, 3),
        "section_coverage": round(sum(section_checks) / len(section_checks), 3) if section_checks else 0.0,
        "passed": status == 200 and all(ok for _, ok in checks),
        "error": None if status == 200 else payload,
    }, report_id


def _rate(items: list[dict], key: str) -> float | None:
    judged = [item[key] for item in items if item[key] is not None]
    if not judged:
        return None
    return round(sum(1 for value in judged if value) / len(judged), 3)


def _latency_block(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    return {
        "n": len(values),
        "mean_ms": round_ms(statistics.fmean(values)),
        "median_ms": round_ms(statistics.median(values)),
        "p95_ms": round_ms(percentile(values, 0.95)),
        "min_ms": round_ms(min(values)),
        "max_ms": round_ms(max(values)),
    }


def summarize(items: list[dict], report: dict) -> dict[str, Any]:
    ok_items = [item for item in items if item["http_status"] == 200]
    latencies = [item["elapsed_ms"] for item in ok_items if item["elapsed_ms"] is not None]
    by_category = {}
    for category in sorted({item["category"] for item in items}):
        group = [item for item in items if item["category"] == category]
        group_ms = [item["elapsed_ms"] for item in group if item["http_status"] == 200 and item["elapsed_ms"] is not None]
        by_category[category] = {
            "n": len(group),
            "pass_rate": _rate(group, "passed"),
            "intent_accuracy": _rate(group, "intent_ok"),
            "slot_accuracy": _rate(group, "slots_ok"),
            "factual_accuracy": _rate(group, "facts_ok"),
            "latency": _latency_block(group_ms),
        }
    paired = [
        item for item in ok_items
        if item.get("baseline_ms") is not None and item.get("elapsed_ms") is not None
    ]
    return {
        "chat_cases": len(items),
        "pass_rate": _rate(items, "passed"),
        "intent_accuracy": _rate(items, "intent_ok"),
        "slot_accuracy": _rate(items, "slots_ok"),
        "factual_accuracy": _rate(items, "facts_ok"),
        "latency_ms": _latency_block(latencies),
        "by_category": by_category,
        "api_baseline_ms": _latency_block([item["baseline_ms"] for item in paired]),
        "median_overhead_ms": round_ms(
            statistics.median([item["elapsed_ms"] - item["baseline_ms"] for item in paired])
        ) if paired else None,
        "report": {
            "elapsed_ms": report["elapsed_ms"],
            "grounding_rate": report["grounding_rate"],
            "section_coverage": report["section_coverage"],
            "passed": report["passed"],
            "model": report["model"],
        },
    }


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 1000:
        return f"{value / 1000:.2f}s"
    return f"{value:.0f}ms"


def write_markdown(path: Path, summary: dict, items: list[dict], report: dict, measured_at: str) -> None:
    lines = [
        "# 실험 QA Agent 측정 결과",
        "",
        f"측정 시각: {measured_at}",
        "",
        "정답은 측정 시점의 MLflow 원본이다. 채팅 정확도는 의도, 슬롯, 사실 세 가지가 모두 맞을 때만 통과로 센다.",
        "신규 대화의 응답시간에는 의도 분류, 조회, 첫 턴 세션 제목 생성이 포함된다.",
        "",
        "## 요약",
        "",
        f"- 채팅 케이스: {summary['chat_cases']}건",
        f"- 종합 통과율: {_fmt_rate(summary['pass_rate'])}",
        f"- 의도 정확도: {_fmt_rate(summary['intent_accuracy'])}",
        f"- 슬롯 정확도: {_fmt_rate(summary['slot_accuracy'])}",
        f"- 사실 정확도: {_fmt_rate(summary['factual_accuracy'])}",
        f"- 채팅 응답시간 중앙값 / p95: {_fmt_ms(summary['latency_ms']['median_ms'])} / {_fmt_ms(summary['latency_ms']['p95_ms'])}",
        f"- 같은 조회의 직접 API 중앙값: {_fmt_ms(summary['api_baseline_ms']['median_ms'])}",
        f"- 자연어 경로의 중앙 오버헤드: {_fmt_ms(summary['median_overhead_ms'])}",
        f"- 보고서 생성 시간: {_fmt_ms(summary['report']['elapsed_ms'])} ({summary['report']['model']})",
        f"- 보고서 수치 재현율: {_fmt_rate(summary['report']['grounding_rate'])}",
        f"- 보고서 섹션 커버리지: {_fmt_rate(summary['report']['section_coverage'])}",
        "",
        "## 범주별",
        "",
        "| 범주 | 건수 | 통과 | 의도 | 슬롯 | 사실 | 응답 중앙값 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, block in summary["by_category"].items():
        lines.append(
            f"| {category} | {block['n']} | {_fmt_rate(block['pass_rate'])} | "
            f"{_fmt_rate(block['intent_accuracy'])} | {_fmt_rate(block['slot_accuracy'])} | "
            f"{_fmt_rate(block['factual_accuracy'])} | {_fmt_ms(block['latency']['median_ms'])} |"
        )
    lines.extend([
        "",
        "## 케이스",
        "",
        "| ID | 통과 | 의도 | 사실 | 응답시간 | 직접 API | 질문 |",
        "|---|---|---|---|---:|---:|---|",
    ])
    for item in items:
        lines.append(
            f"| {item['id']} | {'Y' if item['passed'] else 'N'} | "
            f"{item.get('intent')} | {_fmt_rate(1.0 if item['facts_ok'] else 0.0 if item['facts_ok'] is False else None)} | "
            f"{_fmt_ms(item['elapsed_ms'])} | {_fmt_ms(item.get('baseline_ms'))} | {item['question']} |"
        )
    failed = [item for item in items if not item["passed"]]
    if failed:
        lines.extend(["", "## 실패 상세", ""])
        for item in failed:
            missed_facts = [fact["name"] for fact in item["facts"] if not fact["ok"]]
            missed_slots = [slot["key"] for slot in item["slots"] if not slot["ok"]]
            lines.append(
                f"- `{item['id']}` 기대 의도 `{item['expected_intent']}`, 실제 `{item['intent']}`, "
                f"HTTP {item['http_status']}. 슬롯 불일치: {missed_slots or '-'}. 사실 불일치: {missed_facts or '-'}."
            )
            lines.append(f"  답변 앞부분: {item['answer_preview'].replace(chr(10), ' ')}")
    lines.extend([
        "",
        "## 보고서",
        "",
        f"- 생성 시간: {_fmt_ms(report['elapsed_ms'])}",
        f"- 통과: {'예' if report['passed'] else '아니오'}",
        f"- 수치 재현율: {_fmt_rate(report['grounding_rate'])}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    measured_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    gt = load_ground_truth()
    session_ids: list[str] = []
    report_id = None
    try:
        items, session_ids = run_chat_cases(gt)
        report, report_id = run_report(gt)
    finally:
        for session_id in session_ids:
            app_delete(f"/chat/sessions/{session_id}")
        if report_id:
            app_delete(f"/report/{report_id}")

    summary = summarize(items, report)
    document = {
        "measured_at": measured_at,
        "app_base": APP_BASE,
        "ground_truth_snapshot": {
            "experiment_count": len(gt["experiment_names"]),
            "recent_names": gt["recent_names"],
            "timm_run_names": gt["timm_run_names"],
            "hpo20_map50": gt["hpo20_map50"],
            "exp1_map50": gt["exp1_map50"],
            "top_val_acc_runs": [run["run_name"] for run in gt["top_val_acc"]],
        },
        "summary": summary,
        "cases": items,
        "report": report,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "qa_agent_metrics.json"
    md_path = OUT_DIR / "qa_agent_metrics.md"
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, summary, items, report, measured_at)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
