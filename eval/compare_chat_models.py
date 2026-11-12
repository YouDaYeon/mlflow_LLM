"""Qwen과 Gemma의 채팅 응답시간을 같은 질문으로 비교한다.

Qwen은 현재 백엔드(CHAT_MODEL=Qwen3.6-35B-A3B)다.
Gemma는 같은 이미지로 띄운 별도 백엔드(CHAT_MODEL=gemma-4-26B-A4B-it-UD-Q8_K_XL)다.
보고서 생성은 원래 Gemma라 이 비교에 넣지 않는다.
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

QWEN_BASE = "http://127.0.0.1:8010"
GEMMA_BASE = "http://127.0.0.1:8012"
TIMEOUT_S = 180
OUT_PATH = Path(__file__).resolve().parent / "results" / "chat_model_compare.json"

QUESTIONS = [
    {"id": "list_experiments", "message": "실험 목록 보여줘"},
    {"id": "list_recent_experiments", "message": "최근 실험 알려줘"},
    {"id": "get_experiment", "message": "Timm Hub 실험 정보 알려줘"},
    {"id": "get_runs", "message": "Timm Hub run 목록"},
    {"id": "list_metric_keys", "message": "Timm Hub metric key 알려줘"},
    {"id": "get_run_metric", "message": "yolo26n-hpo-20 run의 metrics/mAP50B 값은 얼마야?"},
    {"id": "get_run_detail", "message": "Deep Learning Experiment에서 burly-kit-617 run 상세 알려줘"},
    {"id": "search_top_accuracy", "message": "Timm Hub에서 metrics.val_acc가 높은 run top 3 보여줘"},
    {"id": "search_lowest_loss", "message": "Timm Hub에서 metrics.val_loss가 가장 낮은 run 보여줘"},
    {"id": "run_judgment", "message": "yolo26n-exp1 run 학습이 잘 된 편이야?"},
    {
        "id": "report_qa",
        "message": "보고서에서 val_acc가 가장 높은 run 이름과 점수를 말해줘",
        "report_text": (
            "<p>val_acc 기준 최상위 run은 capricious-moth-486이며 점수는 0.9569이다.</p>"
        ),
    },
    {"id": "missing_experiment", "message": "없는실험XYZ 실험 정보 알려줘"},
    {
        "id": "multiturn",
        "turns": [
            "Timm Hub 실험 정보 알려줘",
            "metric key 알려줘",
        ],
    },
]


def _request(url: str, *, method: str = "GET", body: dict | None = None) -> tuple[int, dict | None, float]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as response:
            raw = response.read()
            elapsed_ms = (time.perf_counter() - started) * 1000
            payload = json.loads(raw.decode()) if raw else None
            return response.status, payload, elapsed_ms
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        detail = exc.read().decode(errors="replace")
        try:
            payload = json.loads(detail)
        except json.JSONDecodeError:
            payload = {"detail": detail[:300]}
        return exc.code, payload, elapsed_ms


def ask(base: str, message: str, *, session_id: str | None = None, report_text: str | None = None) -> dict:
    body: dict = {"message": message}
    if session_id:
        body["session_id"] = session_id
    if report_text is not None:
        body["report_text"] = report_text
    status, payload, elapsed_ms = _request(f"{base}/chat/ask", method="POST", body=body)
    payload = payload or {}
    answer = str(payload.get("answer") or payload.get("detail") or "")
    return {
        "http_status": status,
        "elapsed_ms": round(elapsed_ms, 1),
        "route": (payload.get("metadata") or {}).get("route") or payload.get("intent"),
        "answer_chars": len(answer),
        "answer_preview": answer.replace("\n", " ")[:180],
        "session_id": payload.get("session_id"),
    }


def delete_session(base: str, session_id: str | None) -> None:
    if not session_id:
        return
    try:
        _request(f"{base}/chat/sessions/{session_id}", method="DELETE")
    except Exception:
        return


def run_model(base: str, label: str) -> list[dict]:
    print(f"[{label}] warmup", flush=True)
    warmup = ask(base, "실험 목록 보여줘")
    delete_session(base, warmup.get("session_id"))

    rows: list[dict] = []
    for spec in QUESTIONS:
        print(f"[{label}] {spec['id']}", flush=True)
        if "turns" in spec:
            session_id = None
            turns = []
            for index, message in enumerate(spec["turns"], start=1):
                result = ask(base, message, session_id=session_id)
                session_id = result.get("session_id") or session_id
                result["turn"] = index
                result["message"] = message
                turns.append(result)
            delete_session(base, session_id)
            rows.append({"id": spec["id"], "turns": turns})
            continue
        result = ask(base, spec["message"], report_text=spec.get("report_text"))
        delete_session(base, result.get("session_id"))
        result["message"] = spec["message"]
        rows.append({"id": spec["id"], **{k: v for k, v in result.items() if k != "session_id"}})
    return rows


def flat_times(rows: list[dict]) -> list[float]:
    times: list[float] = []
    for row in rows:
        if "turns" in row:
            times.extend(turn["elapsed_ms"] for turn in row["turns"] if turn["http_status"] == 200)
        elif row.get("http_status") == 200:
            times.append(row["elapsed_ms"])
    return times


def latency_summary(times: list[float]) -> dict:
    if not times:
        return {"n": 0}
    ordered = sorted(times)
    p95_index = max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))
    return {
        "n": len(times),
        "mean_ms": round(statistics.fmean(times), 1),
        "median_ms": round(statistics.median(times), 1),
        "p95_ms": round(ordered[p95_index], 1),
        "min_ms": round(min(times), 1),
        "max_ms": round(max(times), 1),
        "total_ms": round(sum(times), 1),
    }


def main() -> None:
    qwen = run_model(QWEN_BASE, "qwen")
    gemma = run_model(GEMMA_BASE, "gemma")
    document = {
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "qwen": {
            "base": QWEN_BASE,
            "model": "Qwen3.6-35B-A3B",
            "endpoint": "http://127.0.0.1:9983/v1",
            "summary": latency_summary(flat_times(qwen)),
            "cases": qwen,
        },
        "gemma": {
            "base": GEMMA_BASE,
            "model": "gemma-4-26B-A4B-it-UD-Q8_K_XL",
            "endpoint": "http://127.0.0.1:9980/v1",
            "summary": latency_summary(flat_times(gemma)),
            "cases": gemma,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"qwen": document["qwen"]["summary"], "gemma": document["gemma"]["summary"]}, ensure_ascii=False, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
