"""키워드 규칙(_heuristic_intent)과 JSON 메타프롬프팅의 채팅 14건 비교.

통과는 의도, 슬롯, 사실이 모두 맞을 때다.

키워드: 아래 규칙이 intent만 반환한다. 규칙에 없으면 mlflow_mcp, 인자는 비운다.
메타: INTENT_PARSING_PROMPT로 intent와 arguments를 함께 추출한다.

질문, 라우터, 채점 함수는 같다.
MCP는 양쪽 모두 조회 도구만 연다.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

import backend.app.services.chat_service as chat_service
import backend.app.services.mlflow_service as mlflow_service
from backend.app.main import app

MEASURE_PATH = Path(__file__).resolve().parent / "measure_qa_agent.py"
OUT_PATH = Path(__file__).resolve().parent / "results" / "heuristic_vs_meta.json"

READ_ONLY_MCP_TOOLS = [
    "search_experiments",
    "get_experiment",
    "list_runs",
    "describe_run",
    "search_traces",
    "get_trace",
    "get_trace_assessment",
    "list_scorers",
]


def _load_measure():
    spec = importlib.util.spec_from_file_location("measure_qa_agent", MEASURE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_read_only_mcp() -> None:
    def create_read_only_mcp_server():
        from agents.mcp import MCPServerStdio

        return MCPServerStdio(
            params=mlflow_service.mlflow_mcp_stdio_params(),
            name="mlflow-mcp-readonly",
            cache_tools_list=True,
            client_session_timeout_seconds=mlflow_service.settings.mcp_client_session_timeout_seconds,
            tool_filter={"allowed_tool_names": READ_ONLY_MCP_TOOLS},
        )

    chat_service.create_mlflow_mcp_server = create_read_only_mcp_server


# JSON 스키마와 JSON 예시만 뺀 의도 분류 프롬프트. 의도 정의는 현재 프롬프트와 같다.
PROSE_INTENT_PROMPT = (
    "당신은 MLflow 실험 QA 시스템의 의도 분류기다.\n"
    "사용자 메시지를 읽고 의도(intent)와 필요한 인자를 파악하라.\n"
    "JSON으로 출력하지 마라. 아래 줄 형식으로만 답하라.\n\n"
    "출력 형식:\n"
    "intent: <intent 하나>\n"
    "experiment_name: <있으면 적고, 없으면 이 줄을 생략>\n"
    "run_name: <있으면 적고, 없으면 생략>\n"
    "run_id: <있으면 적고, 없으면 생략>\n"
    "metric_key: <있으면 적고, 없으면 생략>\n"
    "top_k: <있으면 적고, 없으면 생략>\n"
    "ascending: <true 또는 false. 해당 없으면 생략>\n\n"
    "intent 후보:\n"
    "list_experiments, list_recent_experiments, list_metric_keys, "
    "get_experiment, get_runs, get_run_info, search_runs, report_qa, mlflow_mcp\n\n"
    "=== intent 정의 ===\n"
    "- list_experiments: 전체 실험 목록 또는 특정 이름 기반 실험 나열\n"
    "- list_recent_experiments: 최근/최신 실험 조회\n"
    "- list_metric_keys: metric/지표 이름 목록 요청\n"
    "- get_experiment: 특정 실험의 상세 정보 요청. 식별자는 실험 이름이다.\n"
    "- get_runs: 특정 실험의 run 목록 조회 (조건 없음, run 여러 개)\n"
    "- get_run_info: 특정 run 하나의 상세 조회, 또는 그 run의 특정 metric 값 조회.\n"
    "  식별자는 run 이름 또는 32자리 hex run id.\n"
    "  MLflow 기본 run 이름은 adjective-noun-숫자 형태다 (예: carefree-moth-973).\n"
    "  사용자가 이것을 run ID라고 불러도, 정수 experiment id가 아니면 get_run_info이다.\n"
    "  '실험 정보를 알려줘'라도 식별자가 run 이름/run id이면 get_experiment가 아니라 get_run_info이다.\n"
    "  특정 metric 값만 물으면 metric_key를 포함한다. run 전체 상세/정보를 물으면 metric_key를 넣지 않는다.\n"
    "  학습이 잘 됐는지, 평가, 해석, 원인처럼 판단이 필요한 질문은 get_run_info가 아니다.\n"
    "- search_runs: 기간/metric/top_k/상태/tag 등 조건 기반 run 검색\n"
    "- report_qa: 보고서 내용 기반 질의응답\n"
    "- mlflow_mcp: 위 intent로 처리 불가능한 모든 질문.\n"
    "  run 성과 평가, 해석, 원인처럼 판단이 필요하면 mlflow_mcp로 두고 run_name 또는 run_id를 남긴다.\n\n"
    "인자 규칙:\n"
    "1. intent는 반드시 하나만 고른다.\n"
    "2. 추출 가능한 인자만 적는다.\n"
    "3. 실험 이름은 사용자가 말한 그대로 유지한다.\n"
    "4. 현재 질문에 인자가 없어도, 이전 대화에 명시된 값이 있으면 채운다. 없으면 추론하지 않는다.\n"
    "5. get_run_info의 metric_key는 현재 질문이 그 metric 값을 물을 때만 적는다.\n\n"
    "예시:\n"
    "사용자: 실험 목록 보여줘\n"
    "intent: list_experiments\n\n"
    "사용자: 최근 실험 알려줘\n"
    "intent: list_recent_experiments\n\n"
    "사용자: yolo26_gaze tuning 실험 정보 알려줘\n"
    "intent: get_experiment\n"
    "experiment_name: yolo26_gaze tuning\n\n"
    "사용자: yolo26n-hpo-20 run의 metrics/mAP50B 값은 얼마야?\n"
    "intent: get_run_info\n"
    "run_name: yolo26n-hpo-20\n"
    "metric_key: metrics/mAP50B\n\n"
    "사용자: yolo26n-hpo-20 run 학습이 잘 된 편이야?\n"
    "intent: mlflow_mcp\n"
    "run_name: yolo26n-hpo-20\n\n"
    "사용자: 보고서 기반으로 다음 실험을 제안해줘\n"
    "intent: report_qa\n"
)

_EMPTY_ARGUMENT = {"", "없음", "없다", "null", "none", "-", "생략"}


def _parse_prose_intent(raw: str) -> tuple[str, dict]:
    """줄 형식의 의도 분류 결과를 intent와 arguments로 읽는다."""
    text = chat_service.strip_think_tags(raw)
    arguments: dict = {}
    intent = None
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().strip("-*`").lower().replace(" ", "_")
        value = value.strip().strip("`").strip('"').strip("'")
        if key == "intent":
            token = value.split()[0].strip(",") if value else ""
            if token in chat_service.ALLOWED_INTENTS:
                intent = token
            continue
        if key not in chat_service.ALLOWED_INTENT_ARGUMENT_KEYS:
            continue
        if not value or value.lower() in _EMPTY_ARGUMENT:
            continue
        if key == "top_k" and value.isdigit():
            arguments[key] = int(value)
        elif key == "ascending":
            arguments[key] = value.lower() in {"true", "1", "yes", "오름차순"}
        else:
            arguments[key] = value
    if intent is None:
        for name in chat_service.ALLOWED_INTENTS:
            if name in text:
                intent = name
                break
    if intent is None:
        raise ValueError(f"intent를 찾지 못했습니다: {text[:240]}")
    return intent, arguments


def _heuristic_intent(message: str) -> str | None:
    """모델 없이도 자주 쓰는 표현은 규칙으로 고정해 분류 흔들림을 줄인다."""
    m = message.strip()
    if not m:
        return None
    lower = m.lower()

    if re.search(
        r"(실험\s*목록|실험\s*리스트|실험\s*들\s*뭐|어떤\s*실험|전체\s*실험)",
        m,
    ) and "run" not in lower:
        return "list_experiments"

    if re.search(r"(최근\s*실험|최신\s*실험)", m):
        return "list_recent_experiments"

    # 검색 신호(기간/실험자/태그/상태/top_k)가 있으면 get_runs보다 search_runs를 우선한다.
    has_run_term = re.search(r"(?:\bruns?\b|런)", m, re.IGNORECASE) is not None
    has_search_signal = re.search(
        r"(최근\s*\d+\s*일|실험자|사용자|user|유저|태그|running|finished|failed|killed|완료|실패|중단|top[\s_-]*k\s*[:=]?\s*\d+|상위\s*\d+)",
        m,
        re.IGNORECASE,
    ) is not None
    if has_run_term and has_search_signal:
        return "search_runs"

    if has_run_term:
        return "get_runs"

    if re.search(
        r"(실험\s*정보|experiment\s*정보|실험\s*개요|실험\s*상세|아티팩트\s*경로|artifact)",
        m,
        re.IGNORECASE,
    ) and not re.search(r"목록|리스트", m):
        return "get_experiment"

    if re.search(r"(metric\s*key|metric\s*키|지표\s*키|메트릭\s*목록)", m, re.IGNORECASE):
        return "list_metric_keys"

    if re.search(r"(보고서|리포트|report|summary|요약)", m, re.IGNORECASE):
        return "report_qa"

    return None


def _classify_with_heuristic(message: str, history=None):
    """키워드 규칙으로 intent만 고른다. 인자는 추출하지 않는다."""
    del history
    intent = _heuristic_intent(message)
    if intent is None:
        return "mlflow_mcp", {}
    return intent, {}


def _classify_without_json_format(message: str, history=None):
    """의도는 파악하되 JSON 형식으로 받지 않는다."""
    completion = chat_service._client().chat.completions.create(
        model=chat_service.settings.chat_model,
        temperature=0,
        messages=[
            {"role": "system", "content": PROSE_INTENT_PROMPT},
            {"role": "user", "content": chat_service._build_intent_user_content(message, history)},
        ],
    )
    raw = completion.choices[0].message.content or ""
    return _parse_prose_intent(raw)


def main() -> None:
    measure = _load_measure()
    _install_read_only_mcp()
    original_classify = chat_service.classify_intent
    measured_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    with TestClient(app, raise_server_exceptions=False) as client:
        def ask(message: str, *, session_id: str | None = None, report_text: str | None = None) -> dict:
            body: dict = {"message": message}
            if session_id:
                body["session_id"] = session_id
            if report_text is not None:
                body["report_text"] = report_text
            started = time.perf_counter()
            response = client.post("/chat/ask", json=body)
            elapsed_ms = (time.perf_counter() - started) * 1000
            try:
                payload = response.json()
            except Exception:
                payload = {"detail": response.text[:300]}
            return {
                "http_status": response.status_code,
                "elapsed_ms": elapsed_ms,
                "payload": payload if isinstance(payload, dict) else {"detail": payload},
            }

        measure.ask = ask

        def run(label: str, classify) -> tuple[list[dict], dict]:
            print(f"[{label}] start", flush=True)
            chat_service.classify_intent = classify
            items, session_ids = measure.run_chat_cases(measure.load_ground_truth())
            for session_id in session_ids:
                client.delete(f"/chat/sessions/{session_id}")
            summary = measure.summarize(items, {
                "elapsed_ms": None,
                "grounding_rate": None,
                "section_coverage": None,
                "passed": None,
                "model": None,
            })
            print(
                f"[{label}] pass={summary['pass_rate']} intent={summary['intent_accuracy']} "
                f"slots={summary['slot_accuracy']} facts={summary['factual_accuracy']}",
                flush=True,
            )
            return items, summary

        heuristic_items, heuristic_summary = run("heuristic", _classify_with_heuristic)
        meta_items, meta_summary = run("meta", original_classify)

    def compact(items: list[dict]) -> list[dict]:
        return [
            {
                "id": item["id"],
                "turn": item["turn"],
                "passed": item["passed"],
                "intent_ok": item["intent_ok"],
                "slots_ok": item["slots_ok"],
                "facts_ok": item["facts_ok"],
                "intent": item["intent"],
                "expected_intent": item["expected_intent"],
                "http_status": item["http_status"],
                "elapsed_ms": item["elapsed_ms"],
                "answer_preview": item["answer_preview"],
            }
            for item in items
        ]

    document = {
        "measured_at": measured_at,
        "definition": {
            "heuristic": "키워드 규칙으로 intent만 선택. 규칙에 없으면 mlflow_mcp. arguments는 비움",
            "meta": "INTENT_PARSING_PROMPT로 JSON intent와 arguments를 추출한 뒤 분기",
            "pass_rule": "의도, 슬롯, 사실이 모두 맞으면 통과",
            "cases": 14,
        },
        "heuristic": {"summary": heuristic_summary, "cases": compact(heuristic_items)},
        "meta": {"summary": meta_summary, "cases": compact(meta_items)},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "heuristic_pass_rate": heuristic_summary["pass_rate"],
        "meta_pass_rate": meta_summary["pass_rate"],
        "heuristic_intent": heuristic_summary["intent_accuracy"],
        "meta_intent": meta_summary["intent_accuracy"],
        "heuristic_slots": heuristic_summary["slot_accuracy"],
        "meta_slots": meta_summary["slot_accuracy"],
        "heuristic_facts": heuristic_summary["factual_accuracy"],
        "meta_facts": meta_summary["factual_accuracy"],
    }, ensure_ascii=False, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
