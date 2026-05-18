import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple, Union

from openai import OpenAI
from agents import Agent, MultiProvider, Runner, RunConfig
from agents.mcp import MCPServerManager

from backend.app.config import settings
from backend.app.schemas import ExpDisplayRow, RunDisplayRow
from backend.app.services.mlflow_service import (
    list_experiment_names,
    list_recent_experiment_names,
    list_run_metrics,
    get_experiment,
    get_runs,
    create_mlflow_mcp_server,
    extract_tool_calls,
)

ALLOWED_INTENTS = {
    "list_experiments",
    "list_recent_experiments",
    "list_metric_keys",
    "get_experiment",
    "get_runs",
    "search_runs",
    "report_qa",
    "mlflow_mcp",
}
ALLOWED_INTENT_ARGUMENT_KEYS = frozenset(
    {"experiment_name", "metric_key", "top_k", "ascending"}
)
DEFAULT_INTENT = "mlflow_mcp"
MAX_EXPERIMENT_PREVIEW = 20
MAX_RUNS_PREVIEW = 20
MAX_RECENT_EXPERIMENTS = 5
MAX_METRIC_KEYS_PREVIEW = 30
MAX_FILTER_RUNS_PREVIEW = 20

# INTENT_CLASSIFICATION_PROMPT = (
#     "사용자 메시지를 아래 intent 중 하나로만 분류하라. "
#     "반드시 JSON 한 줄로만 답하라. 형식: {\"intent\":\"...\"}\n"
#     "intent 후보: list_experiments, list_recent_experiments, list_metric_keys, get_experiment, get_runs, "
#     "search_runs, report_qa, mlflow_mcp\n"
#     "규칙:\n"
#     "- '어떤 실험(들) 있/있어/보여'처럼 실험 이름 나열·전체 목록: list_experiments\n"
#     "- 최근/최신 실험만: list_recent_experiments\n"
#     "- metric 키/지표 이름 목록: list_metric_keys\n"
#     "- 선택한 실험의 메타(이름, id, artifact 경로, 생성 시각, 태그 등)·개요·상세 정보: get_experiment\n"
#     "- 선택한 실험의 run 전체/목록/리스트(숫자 조건 없이 나열만): get_runs\n"
#     "- 기간/상태/태그/실험자/top_k 등 복합 조건 검색: search_runs\n"
#     "- 보고서 기반 질문에 답변: report_qa\n"
#     "- 위 모든 질문에 답변할 수 없으면 mlflow_mcp를 사용해 답변: mlflow_mcp\n"
#     "예시:\n"
#     "- '실험 목록 보여줘' -> list_experiments\n"
#     "- '이 실험 정보 알려줘' -> get_experiment\n"
#     "- 'metric key 뭐 있어' -> list_metric_keys\n"
#     "- 'run 목록', 'runs 보여줘' -> get_runs\n"
#     "- 'YOLO26 실험 목록 보여줘' -> list_experiments(yolo26) \n"
# )

INTENT_PARSING_PROMPT = (
    "당신은 MLflow 실험 QA 시스템의 Query Parser이다.\n"
    "사용자 메시지를 분석하여 '의도(intent)'와 '필요한 인자(arguments)'를 추출하라.\n"
    "반드시 JSON 한 줄로만 응답하라.\n\n"

    "출력 형식:\n"
    "{\"intent\":\"...\", \"arguments\":{...}}\n\n"

    "intent 후보:\n"
    "list_experiments, list_recent_experiments, list_metric_keys, "
    "get_experiment, get_runs, search_runs, report_qa, mlflow_mcp\n\n"

    "=== intent 정의 ===\n"
    "- list_experiments: 전체 실험 목록 또는 특정 이름 기반 실험 나열\n"
    "- list_recent_experiments: 최근/최신 실험 조회\n"
    "- list_metric_keys: metric/지표 이름 목록 요청\n"
    "- get_experiment: 특정 실험의 상세 정보 요청\n"
    "- get_runs: 특정 실험의 run 목록 조회 (조건 없음)\n"
    "- search_runs: 기간/metric/top_k/상태/tag 등 조건 기반 run 검색\n"
    "- report_qa: 보고서 내용 기반 질의응답\n"
    "- mlflow_mcp: 위 intent로 처리 불가능한 모든 질문\n\n"

    "=== arguments 규칙 ===\n"
    "가능한 인자 키:\n"
    "- experiment_name: 실험 이름\n"
    "- metric_key: metric 이름\n"
    "- top_k: 상위 k개\n"
    "- ascending: 오름차순 정렬 여부\n"

    "규칙:\n"
    "1. intent는 반드시 하나만 선택한다.\n"
    "2. arguments는 추출 가능한 값만 포함한다.\n"
    "3. 인자가 없으면 빈 객체 {} 를 반환한다.\n"
    "4. 자연어 조사, 조사 변형, 조사 제거 후 핵심 값만 추출한다.\n"
    "5. 실험 이름은 사용자가 말한 그대로 유지한다.\n"
    "6. 추론하지 말고 명시된 정보만 추출한다.\n"
    "7. 절대 설명하지 말고 JSON만 출력한다.\n\n"

    "=== 예시 ===\n"

    "사용자: 실험 목록 보여줘\n"
    "{\"intent\":\"list_experiments\",\"arguments\":{}}\n\n"

    "사용자: 최근 실험 알려줘\n"
    "{\"intent\":\"list_recent_experiments\",\"arguments\":{}}\n\n"

    "사용자: yolo26_gaze tuning 실험 정보 알려줘\n"
    "{\"intent\":\"get_experiment\",\"arguments\":{\"experiment_name\":\"yolo26_gaze tuning\"}}\n\n"

    "사용자: yolo26_gaze run 목록\n"
    "{\"intent\":\"get_runs\",\"arguments\":{\"experiment_name\":\"yolo26_gaze\",\"ascending\":false}}\n\n"

    "사용자: metrics.metrics/mAP50B가 높은 run top 5 보여줘\n"
    "{\"intent\":\"search_runs\",\"arguments\":{\"metric_key\":\"metrics.metrics/mAP50B\",\"top_k\":5}}\n\n"

    "사용자: 보고서 기반으로 다음 실험을 제안해줘\n"
    "{\"intent\":\"report_qa\",\"arguments\":{}}\n\n"
)

REPORT_QA_SYSTEM_PROMPT = (
    "당신은 실험 리포트 Q&A 어시스턴트다.\n"
    "아래 규칙을 반드시 지켜라:\n"
    "1) 제공된 report_text만 근거로 답한다.\n"
    "2) 근거가 없으면 반드시 '리포트에 근거 없음'이라고 답한다.\n"
    "3) 사용자가 '핵심 결론', '요약', '한눈에', '정리' 등을 요청하면 "
    "리포트의 핵심 결론을 3~5줄로 요약해 답한다.\n"
    "4) 요약 시에는 핵심 성능 결과, 주요 차이/원인, 다음 액션(튜닝 제안)을 우선 포함한다.\n"
    "5) 불필요한 장문 설명은 피하고 한국어로 간결하게 답한다."
)


def _client() -> OpenAI:
    return OpenAI(base_url=settings.chat_base_url, api_key=settings.chat_api_key)
    # return OpenAI(base_url=settings.llama_base_url, api_key=settings.llama_api_key)


# def _intent_from_llm_raw(raw: str) -> str:
#     text = (raw or "").strip()
#     if not text:
#         return DEFAULT_INTENT
#     if text.startswith("```"):
#         lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("```")]
#         text = "\n".join(lines).strip()
#     try:
#         parsed = json.loads(text)
#         intent = parsed.get("intent", DEFAULT_INTENT)
#     except json.JSONDecodeError:
#         m = re.search(r'"intent"\s*:\s*"([^"]+)"', text)
#         intent = m.group(1) if m else DEFAULT_INTENT
#     return intent if intent in ALLOWED_INTENTS else DEFAULT_INTENT


# def _heuristic_intent(message: str) -> str | None:
#     """모델 없이도 자주 쓰는 표현은 규칙으로 고정해 분류 흔들림을 줄인다."""
#     m = message.strip()
#     if not m:
#         return None
#     lower = m.lower()

#     if re.search(
#         r"(실험\s*목록|실험\s*리스트|실험\s*들\s*뭐|어떤\s*실험|전체\s*실험)",
#         m,
#     ) and "run" not in lower:
#         return "list_experiments"

#     if re.search(r"(최근\s*실험|최신\s*실험)", m):
#         return "list_recent_experiments"

#     # 검색 신호(기간/실험자/태그/상태/top_k)가 있으면 get_runs보다 search_runs를 우선한다.
#     has_run_term = re.search(r"(?:\bruns?\b|런)", m, re.IGNORECASE) is not None
#     has_search_signal = re.search(
#         r"(최근\s*\d+\s*일|실험자|사용자|user|유저|태그|running|finished|failed|killed|완료|실패|중단|top[\s_-]*k\s*[:=]?\s*\d+|상위\s*\d+)",
#         m,
#         re.IGNORECASE,
#     ) is not None
#     if has_run_term and has_search_signal:
#         return "search_runs"

#     if has_run_term:
#         return "get_runs"

#     if re.search(
#         r"(실험\s*정보|experiment\s*정보|실험\s*개요|실험\s*상세|아티팩트\s*경로|artifact)",
#         m,
#         re.IGNORECASE,
#     ) and not re.search(r"목록|리스트", m):
#         return "get_experiment"

#     if re.search(r"(metric\s*key|metric\s*키|지표\s*키|메트릭\s*목록)", m, re.IGNORECASE):
#         return "list_metric_keys"

#     if re.search(r"(보고서|리포트|report|summary|요약)", m, re.IGNORECASE):
#         return "report_qa"

#     return None

def classify_intent(message: str) -> Tuple[str, Dict[str, Any]]:
    # hinted = _heuristic_intent(message)
    # if hinted is not None:
    #     return hinted

    completion = _client().chat.completions.create(
        model=settings.llama_model,
        temperature=0,
        messages=[
            {"role": "system", "content": INTENT_PARSING_PROMPT},
            {"role": "user", "content": message},
        ],
    )
    raw = (completion.choices[0].message.content or "").strip()

    # gemma4 raw 형식: {"intent":"...", "arguments":{...}}
    # qwen3.6 raw 형식: '<think>\n\n</think>\n\n{"intent":"list_experiments","arguments":{}}'
    match = re.search(r"\{.*\}", raw, re.DOTALL)

    if not match:
        raise ValueError("No JSON found in LLM output")

    json_str = match.group(0)

    data = json.loads(json_str)
    intent = data.get("intent")
    arguments = data.get("arguments", {})

    return intent, arguments

def _format_preview_list(items: list[str], limit: int) -> tuple[str, str]:
    preview = "\n".join(f"- {item}" for item in items[:limit])
    suffix = f"\n(너무 길어 {limit}개만 표시)" if len(items) > limit else ""
    return preview, suffix

def _format_run_display_markdown(experiment_name: str, total: int, rows: list[RunDisplayRow]) -> str:
    """챗봇 답변용 마크다운 표(Parameters 열은 생략, 상세는 metadata.run_display_rows)."""
    if not rows:
        return f"`{experiment_name}`에 조회된 run이 없습니다."
    lines = [
        f"`{experiment_name}` run **{total}개** (미리보기 **{len(rows)}개**)",
        "",
        "| Name | Score | Status | Date | Time |",
        "|------|-------|--------|------|------|",
    ]
    for row in rows:
        d = row.model_dump(by_alias=True)
        score = d["Score"]
        score_s = f"{score:.4f}" if score is not None else "-"
        name = str(d["Name"]).replace("|", "\\|")
        lines.append(f"| {name} | {score_s} | {d['Status']} | {d['Date']} | {d['Time']} |")
    return "\n".join(lines)


def _parse_search_runs_query(message: str) -> dict[str, object]:
    """기간/상태/실험자/태그/top_k 조건을 사용자 문장에서 추출한다."""
    msg = (message or "").strip()
    lower_msg = msg.lower()

    recent_days: int | None = None
    m_days = re.search(r"최근\s*(\d+)\s*일", msg)
    if m_days:
        recent_days = int(m_days.group(1))

    status: str | None = None
    if "running" in lower_msg:
        status = "RUNNING"
    elif "finished" in lower_msg or "완료" in msg:
        status = "FINISHED"
    elif "failed" in lower_msg or "실패" in msg:
        status = "FAILED"
    elif "killed" in lower_msg or "중단" in msg:
        status = "KILLED"

    user: str | None = None
    m_user = re.search(r"(?:실험자|사용자|유저)\s*[:=]?\s*([A-Za-z0-9._-]+)", msg)
    if m_user:
        user = m_user.group(1).strip()

    tag_key: str | None = None
    tag_value: str | None = None
    m_tag = re.search(r"태그\s*[:=]?\s*([A-Za-z0-9._/-]+)\s*[:=]\s*([A-Za-z0-9._/-]+)", msg)
    if m_tag:
        tag_key = m_tag.group(1).strip()
        tag_value = m_tag.group(2).strip()

    return {
        "recent_days": recent_days,
        "status": status,
        "user": user,
        "tag_key": tag_key,
        "tag_value": tag_value
    }


def _to_epoch_seconds(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        return ts
    s = str(value).strip()
    if not s:
        return None
    try:
        ts = float(s)
        if ts > 1e12:
            ts /= 1000.0
        return ts
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.timestamp()
    except ValueError:
        return None


def _apply_search_runs_filters(runs: list[dict], filters: dict[str, object]) -> list[dict]:
    """조건별로 독립 적용: 기간 -> 상태 -> 실험자 -> 태그."""
    filtered = list(runs)

    recent_days = filters.get("recent_days")
    if isinstance(recent_days, int) and recent_days > 0:
        cutoff = datetime.now().timestamp() - timedelta(days=recent_days).total_seconds()
        filtered = [
            run
            for run in filtered
            if (_to_epoch_seconds(run.get("start_time")) is not None and _to_epoch_seconds(run.get("start_time")) >= cutoff)
        ]

    status = filters.get("status")
    if isinstance(status, str) and status:
        target = status.upper()
        filtered = [run for run in filtered if str(run.get("status", "")).upper() == target]

    user = filters.get("user")
    if isinstance(user, str) and user:
        filtered = [run for run in filtered if str(run.get("tags.mlflow.user", "")).strip() == user]

    tag_key = filters.get("tag_key")
    tag_value = filters.get("tag_value")
    if isinstance(tag_key, str) and tag_key and isinstance(tag_value, str) and tag_value:
        tag_col = f"tags.{tag_key}"
        filtered = [run for run in filtered if str(run.get(tag_col, "")).strip() == tag_value]

    return filtered


def _format_search_filters(filters: dict[str, object], metric_key: str | None) -> str:
    items: list[str] = []
    if isinstance(filters.get("recent_days"), int):
        items.append(f"기간:{filters['recent_days']}일")
    if isinstance(filters.get("status"), str) and filters["status"]:
        items.append(f"상태:{filters['status']}")
    if isinstance(filters.get("user"), str) and filters["user"]:
        items.append(f"실험자:{filters['user']}")
    tag_key = filters.get("tag_key")
    tag_value = filters.get("tag_value")
    if isinstance(tag_key, str) and tag_key and isinstance(tag_value, str) and tag_value:
        items.append(f"태그:{tag_key}:{tag_value}")
    if isinstance(filters.get("top_k"), int):
        top_desc = f"top_k:{filters['top_k']}"
        if metric_key:
            top_desc += f" ({metric_key} 기준)"
        items.append(top_desc)
    if isinstance(filters.get("ascending"), bool):
        if filters['ascending'] == True:
            items.append("오름차순")
        else:
            items.append("내림차순")
    if not items:
        return "적용 필터: 없음(전체 run)"
    return "적용 필터: " + " | ".join(items)

def answer_experiment_question(
    intent: str,
    experiment_name: str | None,
    message: str | None,
    metric_key: str | None,
    top_k: int | None,
    ascending: bool | None,
) -> tuple[str, str, dict]:
    if intent == "list_experiments":   
        names = list_experiment_names()
        if not names:
            return "등록된 실험이 없습니다.", intent, {"count": 0}
        preview, suffix = _format_preview_list(names, MAX_EXPERIMENT_PREVIEW)
        return f"실험 목록({len(names)}개)\n{preview}{suffix}", intent, {"count": len(names)}

    if intent == "list_recent_experiments":
        names = list_recent_experiment_names(limit=MAX_RECENT_EXPERIMENTS)
        if not names:
            return "최근 실험이 없습니다.", intent, {"count": 0}
        return f"최근 실험({len(names)}개)\n" + "\n".join(f"- {name}" for name in names), intent, {"count": len(names)}

    target_experiment = experiment_name
    if not target_experiment:
        return "조회할 실험이 없습니다.", intent, {}

    # list_metric_keys
    if intent == "list_metric_keys":
        keys = list_run_metrics(target_experiment)
        if not keys:
            return f"{target_experiment}의 metric 키가 없습니다.", intent, {"experiment_name": target_experiment, "count": 0}
        preview, suffix = _format_preview_list(keys, MAX_METRIC_KEYS_PREVIEW)
        return (
            f"{target_experiment} metric 키 목록({len(keys)}개)\n{preview}{suffix}",
            intent,
            {"experiment_name": target_experiment, "count": len(keys)},
        )

    # get_experiment
    if intent == "get_experiment":
        experiment = get_experiment(target_experiment)
        if not experiment:
            return f"{target_experiment} 실험이 없습니다.", intent, {"experiment_name": target_experiment, "count": 0}
        exp_row = ExpDisplayRow.from_experiment_dict(target_experiment, experiment)
        return (
            exp_row.to_markdown(),
            intent,
            {
                "experiment_name": target_experiment,
                "experiment_display_row": exp_row.to_metadata(),
            },
        )

    # get_runs
    if intent == "get_runs":
        runs = get_runs(target_experiment)
        if not runs:
            return f"{target_experiment} run 목록이 없습니다.", intent, {"experiment_name": target_experiment, "count": 0}
        rows = [
            RunDisplayRow.from_mlflow_raw(r, metric_key)
            for r in runs
        ]
        # # Score(선택 metric 기준) 내림차순; 없으면 맨 뒤로
        # rows = sorted(
        #     rows,
        #     key=lambda r: r.score if r.score is not None else float("-inf"),
        #     reverse=True,
        # )
        preview_rows = rows[:MAX_RUNS_PREVIEW]
        answer = _format_run_display_markdown(target_experiment, len(runs), preview_rows)
        return (
            answer,
            intent,
            {
                "experiment_name": target_experiment,
                "count": len(runs),
                "run_display_rows": [row.model_dump(mode="json", by_alias=True) for row in preview_rows],
            },
        )

    # search_runs
    if intent == "search_runs":
        runs = get_runs(target_experiment)
        if not runs:
            return "조회 가능한 run이 없습니다.", intent, {"experiment_name": target_experiment, "count": 0}

        filters = _parse_search_runs_query(message or "")
        filters = {**filters, "top_k": top_k, "ascending": ascending}
        if isinstance(top_k, int) and top_k > 0 and not metric_key:
            return (
                "top_k 검색에는 metric_key 선택이 필요합니다. metric을 선택한 뒤 다시 시도해 주세요.",
                intent,
                {"experiment_name": target_experiment, "filters": filters},
            )
        
        filtered_runs = _apply_search_runs_filters(runs, filters)
        if not filtered_runs:
            return (
                f"조건에 맞는 run이 없습니다.\n{_format_search_filters(filters, metric_key)}",
                intent,
                {"experiment_name": target_experiment, "count": 0, "filters": filters},
            )

        rows = [RunDisplayRow.from_mlflow_raw(r, metric_key) for r in filtered_runs]
        if metric_key and ascending==True:
            rows = sorted(rows, key=lambda r: r.score if r.score is not None else float("-inf"), reverse=False)
        elif metric_key and ascending==False:
            rows = sorted(rows, key=lambda r: r.score if r.score is not None else float("-inf"), reverse=True)
        else:
            rows = sorted(rows, key=lambda r: _to_epoch_seconds(r.model_dump(by_alias=True).get("Date")) or 0.0, reverse=True)

        if isinstance(top_k, int) and top_k > 0:
            rows = rows[:top_k]

        preview_rows = rows[:MAX_RUNS_PREVIEW]
        filter_text = _format_search_filters(filters, metric_key)
        table_text = _format_run_display_markdown(target_experiment, len(rows), preview_rows)
        answer = f"{filter_text}\n\n{table_text}"
        return (
            answer,
            intent,
            {
                "experiment_name": target_experiment,
                "count": len(rows),
                "filters": filters,
                "run_display_rows": [row.model_dump(mode="json", by_alias=True) for row in preview_rows],
            },
        )

    return f"지원하지 않는 실험 요청입니다: {intent}", intent, {"experiment_name": target_experiment}

def answer_report_question(
    intent: str,
    message: str,
    report_text: str | None,
    experiment_name: str | None,
    metric_key: str | None,
) -> tuple[str, str, dict]:

    if not report_text:
        return "생성된 report_text가 없어 리포트 질문에 답할 수 없습니다.", intent, {}

    payload = {
        "question": message,
        "report_text": report_text,
        "experiment_name": experiment_name,
        "metric_key": metric_key,
    }
    completion = _client().chat.completions.create(
        model=settings.llama_model,
        temperature=settings.llama_temperature,
        messages=[
            {"role": "system", "content": REPORT_QA_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    answer = (completion.choices[0].message.content or "").strip()
    return answer, intent, {"experiment_name": experiment_name, "metric_key": metric_key}


async def run_with_mcp(
    message: str,
) -> tuple[str, str, dict]:
    server = create_mlflow_mcp_server()
    async with MCPServerManager([server], strict=False) as manager:
        if not manager.active_servers:
            err = next(iter(manager.errors.values()), None)
            detail = repr(err) if err else "연결 실패(상세 없음)"
            return (
                f"MLflow MCP에 연결하지 못했습니다: {detail}",
                DEFAULT_INTENT,
                {"mcp_connected": False, "error": detail},
            )

        agent = Agent(
            name="MCP Helper",
            instructions=(
                "제공된 MCP 도구를 사용해 사용자 질문에 답하세요.\n"
                "도구 호출이 필요하면 적극적으로 사용하세요."
            ),
            model=settings.llama_model,
            mcp_servers=manager.active_servers,
        )
        result = await Runner.run(
            starting_agent=agent,
            input=message,
            run_config=RunConfig(
                tracing_disabled=True,         # openapi 트레이싱 끔
                model_provider=MultiProvider(
                    openai_api_key=settings.llama_api_key,
                    openai_base_url=settings.llama_base_url,
                    openai_use_responses=False,
                ),
            ),
        )
        answer = result.final_output
        tool_calls = extract_tool_calls(result)
        answer = f"Tool Calls: {tool_calls}\n\n {answer}"

    return answer, DEFAULT_INTENT, {"mcp_connected": True, "tool_calls": tool_calls}

