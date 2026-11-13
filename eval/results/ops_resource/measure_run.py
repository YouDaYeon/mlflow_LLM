"""운영 리소스 측정용: eval/compare_request_parsing.py에 LLM 호출별 시간·토큰과 조회 함수 시간을 더 기록한 사본.

사용:
  python eval/results/ops_resource/measure_run.py --label current
  python eval/results/ops_resource/measure_run.py --label head --code-root /path/to/head-worktree

--code-root의 backend 코드를 불러와 TestClient로 실행한다(.env는 실행 위치 기준).
케이스마다 다음을 남긴다.
- parser: 개선 전은 classify_intent 결과(intent, arguments)와 검색 필터 파서 결과,
  개선 후는 parse_request가 만든 요청 구조
- calls: 실행된 MLflow 조회 함수와 인자, LLM 답변 호출, MCP 도구 호출
- answer: 최종 답변 전문

MCP는 조회 도구만 연다(삭제 요청 케이스가 실제 삭제하지 않도록).
측정이 만든 채팅 세션은 끝나면 삭제한다.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

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

# chat_service가 import한 조회 함수 중 기록할 것. 버전마다 있는 것만 감싼다.
TRACED_LOOKUPS = [
    "list_experiment_names",
    "list_recent_experiment_names",
    "list_recent_experiments",
    "list_run_metrics",
    "find_runs",
    "get_experiment",
    "get_runs",
]


def _routed(payload):
    if not isinstance(payload, dict):
        return None
    route = (payload.get("metadata") or {}).get("route")
    return route if route else payload.get("intent")


def _short(value, limit=200):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--code-root", default=str(EVAL_DIR.parent))
    parser.add_argument("--only", nargs="*", help="실행할 케이스 id(세션 단위로 실행됨)")
    parser.add_argument("--cases", default="request_parsing_cases", help="케이스 모듈 이름(eval/ 안)")
    args = parser.parse_args()

    code_root = Path(args.code_root).resolve()
    sys.path.insert(0, str(code_root))
    sys.path.insert(1, str(EVAL_DIR))

    from fastapi.testclient import TestClient

    import backend.app.services.chat_service as chat_service
    import backend.app.services.mlflow_service as mlflow_service
    from backend.app.main import app
    from session_cleanup import cleanup_empty_sessions
    import importlib

    CASES = importlib.import_module(args.cases).CASES

    assert Path(chat_service.__file__).resolve().is_relative_to(code_root), chat_service.__file__

    trace: dict[str, list] = {"parser": [], "calls": []}

    # ── 계측 ──
    try:  # 개선 후 버전: 요청 구조 해석기
        import backend.app.services.chat_request as chat_request
    except ImportError:
        chat_request = None

    if hasattr(chat_service, "classify_intent"):  # 개선 전 버전
        original_classify = chat_service.classify_intent

        def traced_classify(message, history=None):
            intent, arguments = original_classify(message, history=history)
            trace["parser"].append({"intent": intent, "arguments": arguments})
            return intent, arguments

        chat_service.classify_intent = traced_classify

        original_filter = chat_service._parse_search_runs_query

        def traced_filter(message):
            filters = original_filter(message)
            trace["parser"].append({"search_filters": filters})
            return filters

        chat_service._parse_search_runs_query = traced_filter

    if chat_request is not None:
        original_parse = chat_request.parse_request

        def traced_parse(message, history=None, previous=None):
            request = original_parse(message, history=history, previous=previous)
            compact = {k: v for k, v in request.items() if v not in (None, [], {}, False)}
            trace["parser"].append({"request": compact})
            return request

        chat_request.parse_request = traced_parse

    modules = [chat_service] + ([chat_request] if chat_request is not None else [])
    for module, name in [(m, n) for m in modules for n in TRACED_LOOKUPS]:
        fn = getattr(module, name, None)
        if fn is None:
            continue

        def make(fn=fn, name=name):
            @functools.wraps(fn)
            def wrapper(*a, **kw):
                t0 = time.perf_counter()
                result = fn(*a, **kw)
                ms = round((time.perf_counter() - t0) * 1000)
                size = len(result) if isinstance(result, (list, dict)) else None
                trace["calls"].append({"tool": name, "args": list(a), "kwargs": kw, "result_size": size, "ms": ms})
                return result

            return wrapper

        setattr(module, name, make())

    # 정리 후 버전은 llm_client, 이전 버전은 _client다.
    client_attr = "llm_client" if hasattr(chat_service, "llm_client") else "_client"
    original_client = getattr(chat_service, client_attr)

    def traced_client():
        client = original_client()
        create = client.chat.completions.create

        def wrapped_create(*a, **kw):
            system = next(
                (m["content"] for m in kw.get("messages", []) if m.get("role") == "system"), ""
            )
            if system.startswith("당신은 MLflow 실험 QA 시스템의 Query Parser"):
                kind = "llm:intent_parser"
            elif system.startswith("당신은 MLflow run 검색 필터"):
                kind = "llm:search_filter_parser"
            elif system.startswith("당신은 실험 리포트"):
                kind = "llm:report_answer"
            elif system.startswith("당신은 MLflow run을 해석"):
                kind = "llm:run_answer"
            elif system.startswith("당신은 MLflow 실험 QA 챗봇의 요청 해석기"):
                kind = "llm:request_parser"
            elif system.startswith("당신은 MLflow run을 요약"):
                kind = "llm:run_summary"
            elif system.startswith("당신은 MLflow run을 평가"):
                kind = "llm:run_evaluation"
            elif system.startswith("대화 세션 제목"):
                kind = "llm:session_title"
            else:
                kind = "llm:other"
            t0 = time.perf_counter()
            resp = create(*a, **kw)
            ms = round((time.perf_counter() - t0) * 1000)
            u = getattr(resp, "usage", None)
            trace["calls"].append({"tool": kind, "llm_ms": ms,
                "prompt_tokens": getattr(u, "prompt_tokens", None),
                "completion_tokens": getattr(u, "completion_tokens", None)})
            return resp

        client.chat.completions.create = wrapped_create
        return client

    setattr(chat_service, client_attr, traced_client)

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

    cases = CASES
    if args.only:
        wanted_sessions = {c.get("session") for c in CASES if c["id"] in args.only and c.get("session")}
        cases = [c for c in CASES if c["id"] in args.only or (c.get("session") in wanted_sessions)]

    results = []
    session_ids: dict[str, str] = {}
    created_sessions: list[str] = []
    measured_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    with cleanup_empty_sessions(), TestClient(app, raise_server_exceptions=False) as client:
        for case in cases:
            trace["parser"] = []
            trace["calls"] = []
            body: dict = {"message": case["message"]}
            group = case.get("session")
            if group and group in session_ids:
                body["session_id"] = session_ids[group]
            if case.get("report_text"):
                body["report_text"] = case["report_text"]

            started = time.perf_counter()
            response = client.post("/chat/ask", json=body)
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            try:
                payload = response.json()
            except Exception:
                payload = {"detail": response.text[:500]}

            sid = payload.get("session_id") if isinstance(payload, dict) else None
            if sid:
                if sid not in created_sessions:
                    created_sessions.append(sid)
                if group:
                    session_ids[group] = sid

            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            mcp_calls = [
                tc for tc in (metadata.get("tool_calls") or [])
                if isinstance(tc, dict) and metadata.get("mcp_connected")
            ]
            calls = list(trace["calls"])
            for tc in mcp_calls:
                calls.append({"tool": f"mcp:{tc.get('name') or tc.get('tool')}", "args": tc.get("arguments")})

            item = {
                "id": case["id"],
                "group": case["group"],
                "problem": case["problem"],
                "session": group,
                "message": case["message"],
                "expected": case["expected"],
                "http_status": response.status_code,
                "elapsed_ms": elapsed_ms,
                "parser": list(trace["parser"]),
                # 개선 전 응답은 intent, 개선 후 응답은 metadata.route(파서·함수)로 경로를 알린다.
                "routed": _routed(payload),
                "calls": calls,
                "answer": payload.get("answer") if isinstance(payload, dict) else None,
                "error": None if response.status_code == 200 else payload,
            }
            results.append(item)
            print(
                f"[{args.label}] {case['id']} {response.status_code} {elapsed_ms}ms "
                f"route={item['routed']} parser={_short(item['parser'], 160)}",
                flush=True,
            )

        for sid in created_sessions:
            client.delete(f"/chat/sessions/{sid}")

    out = {
        "label": args.label,
        "code_root": str(code_root),
        "measured_at": measured_at,
        "model": chat_service.settings.chat_model,
        "cases": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"request_parsing_{args.label}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
