import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple, Union

from openai import OpenAI
from agents import Agent, MultiProvider, Runner, RunConfig
from agents.mcp import MCPServerManager

from backend.app.config import settings
from backend.app.schemas import ChatMessage, ExpDisplayRow, RunDisplayRow
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

# Qwen 등 reasoning 모델이 붙이는 사고 과정 블록
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def strip_think_tags(text: str | None) -> str:
    """LLM 응답에서 <think>...</think> 블록을 제거한다."""
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    return cleaned.strip()

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
    "6. 현재 질문에 인자가 없어도, 이전 대화에 명시된 experiment_name/metric_key/top_k/"
    "ascending이 있으면 arguments에 채운다. 이전 대화에도 없으면 추론하지 않는다.\n"
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

    "이전 대화에 실험명 yolo26_gaze가 있고 현재 질문만 'metric key 알려줘'인 경우:\n"
    "{\"intent\":\"list_metric_keys\",\"arguments\":{\"experiment_name\":\"yolo26_gaze\"}}\n"
)

SEARCH_RUNS_FILTER_PARSING_PROMPT = (
    "당신은 MLflow run 검색 필터 추출기(Query Parser)이다.\n"
    "사용자 메시지에서 run 검색에 쓰일 '기간/상태/실험자/태그' 조건만 추출하라.\n"
    "반드시 JSON 한 줄로만 응답하라.\n\n"

    "출력 형식:\n"
    "{\"recent_days\":null,\"status\":null,\"user\":null,\"tag_key\":null,\"tag_value\":null}\n\n"

    "=== 필드 정의 ===\n"
    "- recent_days: 정수. '최근 N일', '지난 N일', 'last N days' 등 기간 조건의 N.\n"
    "  기간 조건이 없으면 null.\n"
    "- status: MLflow run 상태. 아래 중 하나 또는 null.\n"
    "  RUNNING | FINISHED | FAILED | KILLED\n"
    "  한국어/영어 예: running/실행중, finished/완료, failed/실패, killed/중단\n"
    "- user: 실험자(MLflow user) 식별자 문자열. '실험자', '사용자', '유저', 'user' 뒤 값.\n"
    "  없으면 null.\n"
    "- tag_key, tag_value: MLflow 태그 key/value 쌍. '태그 key=value', 'tag key: value' 등.\n"
    "  둘 다 있을 때만 채우고, 하나만 있으면 둘 다 null.\n\n"

    "=== 규칙 ===\n"
    "1. metric_key, top_k, experiment_name, 정렬(ascending)은 추출하지 않는다.\n"
    "2. 메시지에 명시된 정보만 추출하고, 추론·가정하지 않는다.\n"
    "3. 조건이 없는 필드는 반드시 null이다 (빈 문자열 \"\" 사용 금지).\n"
    "4. status는 반드시 대문자 enum(RUNNING, FINISHED, FAILED, KILLED)만 사용한다.\n"
    "5. user, tag_key, tag_value는 사용자가 말한 값을 그대로 유지한다.\n"
    "6. 절대 설명하지 말고 JSON만 출력한다.\n\n"

    "=== 예시 ===\n"

    "사용자: 최근 7일 finished run 보여줘\n"
    "{\"recent_days\":7,\"status\":\"FINISHED\",\"user\":null,\"tag_key\":null,\"tag_value\":null}\n\n"

    "사용자: 실험자 alice가 실행한 failed runs\n"
    "{\"recent_days\":null,\"status\":\"FAILED\",\"user\":\"alice\",\"tag_key\":null,\"tag_value\":null}\n\n"

    "사용자: 태그 env=prod 인 run 목록\n"
    "{\"recent_days\":null,\"status\":null,\"user\":null,\"tag_key\":\"env\",\"tag_value\":\"prod\"}\n\n"

    "사용자: 최근 3일 동안 사용자 bob이 돌린 running run, 태그 stage=train\n"
    "{\"recent_days\":3,\"status\":\"RUNNING\",\"user\":\"bob\",\"tag_key\":\"stage\",\"tag_value\":\"train\"}\n\n"

    "사용자: mAP top 5\n"
    "{\"recent_days\":null,\"status\":null,\"user\":null,\"tag_key\":null,\"tag_value\":null}\n\n"
)

REPORT_QA_SYSTEM_PROMPT = (
    "당신은 실험 리포트 Q&A 어시스턴트다.\n"
    "아래 규칙을 반드시 지켜라:\n"
    "1) 제공된 report_text만 근거로 답한다.\n"
    "2) 근거가 없으면 반드시 '리포트에 근거 없음'이라고 답한다.\n"
    "3) 사용자가 '핵심 결론', '요약', '한눈에', '정리' 등을 요청하면 "
    "리포트의 핵심 결론을 3~5줄로 요약해 답한다.\n"
    "4) 요약 시에는 핵심 성능 결과, 주요 차이/원인, 다음 액션(튜닝 제안)을 우선 포함한다.\n"
    "5) 이전 대화가 있으면 후속 질문의 지시어(그거/약점도/다시 등)를 맥락에 맞게 해석하되, "
    "여전히 report_text만 근거로 답한다.\n"
    "6) 불필요한 장문 설명은 피하고 한국어로 간결하게 답한다."
)


def _client() -> OpenAI:
    return OpenAI(base_url=settings.chat_base_url, api_key=settings.chat_api_key)
    # return OpenAI(base_url=settings.llama_base_url, api_key=settings.llama_api_key)


def generate_session_title(user_message: str, assistant_answer: str | None = None) -> str:
    """첫 턴 기준 짧은 세션 제목을 LLM으로 생성. 실패 시 메시지 truncate."""
    fallback = (user_message or "").strip().replace("\n", " ")
    if len(fallback) > 80:
        fallback = fallback[:80].rstrip() + "…"
    fallback = fallback or "새 대화"

    snippet = (assistant_answer or "").strip().replace("\n", " ")
    if len(snippet) > 200:
        snippet = snippet[:200].rstrip() + "…"

    try:
        completion = _client().chat.completions.create(
            model=settings.chat_model,
            temperature=0,
            max_tokens=40,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "대화 세션 제목을 한 줄로 만들어라. "
                        "한국어 명사구 위주, 최대 20자, 따옴표/설명 없이 제목만 출력."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"사용자: {user_message.strip()}\n"
                        f"답변 요약: {snippet or '(없음)'}"
                    ),
                },
            ],
        )
        title = strip_think_tags(completion.choices[0].message.content or "")
        title = title.strip(" \"'`")
        if title:
            return title[:80]
    except Exception:
        pass
    return fallback


def _format_history_for_intent(
    history: list[ChatMessage] | None,
    *,
    max_assistant_chars: int = 400,
) -> str:
    """intent 분류용으로 이전 대화를 짧게 직렬화한다. 비어 있으면 빈 문자열."""
    if not history:
        return ""
    lines: list[str] = []
    for msg in history:
        role = (msg.role or "").strip().lower()
        content = (msg.content or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        if role == "assistant" and len(content) > max_assistant_chars:
            content = content[:max_assistant_chars].rstrip() + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_intent_user_content(message: str, history: list[ChatMessage] | None) -> str:
    history_text = _format_history_for_intent(history)
    if not history_text:
        return message
    return (
        "이전 대화(참고용):\n"
        f"{history_text}\n\n"
        "현재 질문:\n"
        f"{message}"
    )


def _history_to_openai_messages(
    history: list[ChatMessage] | None,
) -> list[dict[str, str]]:
    """ChatMessage 이력을 OpenAI chat messages 형식의 user/assistant 턴으로 변환."""
    messages: list[dict[str, str]] = []
    if not history:
        return messages
    for msg in history:
        role = (msg.role or "").strip().lower()
        content = (msg.content or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    return messages


def classify_intent(
    message: str,
    history: list[ChatMessage] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    # hinted = _heuristic_intent(message)
    # if hinted is not None:
    #     return hinted

    completion = _client().chat.completions.create(
        model=settings.chat_model,
        temperature=0,
        messages=[
            {"role": "system", "content": INTENT_PARSING_PROMPT},
            {"role": "user", "content": _build_intent_user_content(message, history)},
        ],
    )
    raw = strip_think_tags(completion.choices[0].message.content or "")

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


_ALLOWED_RUN_STATUSES = frozenset({"RUNNING", "FINISHED", "FAILED", "KILLED"})


def _empty_search_runs_filters() -> dict[str, object]:
    return {
        "recent_days": None,
        "status": None,
        "user": None,
        "tag_key": None,
        "tag_value": None,
    }

def _parse_search_runs_filters_from_llm_raw(raw: str) -> dict[str, object]:
    text = (raw or "").strip()
    if not text:
        return _empty_search_runs_filters()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return _empty_search_runs_filters()

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        return _empty_search_runs_filters()
        
    # LLM JSON을 dict로 정규화
    out = _empty_search_runs_filters()

    recent_days = parsed.get("recent_days")
    if isinstance(recent_days, int) and recent_days > 0:
        out["recent_days"] = recent_days
    elif isinstance(recent_days, str) and recent_days.strip().isdigit():
        days = int(recent_days.strip())
        if days > 0:
            out["recent_days"] = days

    status = parsed.get("status")
    if isinstance(status, str) and status.strip():
        normalized = status.strip().upper()
        if normalized in _ALLOWED_RUN_STATUSES:
            out["status"] = normalized

    user = parsed.get("user")
    if isinstance(user, str) and user.strip():
        out["user"] = user.strip()

    tag_key = parsed.get("tag_key")
    tag_value = parsed.get("tag_value")
    if (
        isinstance(tag_key, str)
        and tag_key.strip()
        and isinstance(tag_value, str)
        and tag_value.strip()
    ):
        out["tag_key"] = tag_key.strip()
        out["tag_value"] = tag_value.strip()

    return out


def _parse_search_runs_query(message: str) -> dict[str, object]:
    """기간/상태/실험자/태그 조건을 사용자 문장에서 LLM으로 추출한다."""
    msg = (message or "").strip()
    if not msg:
        return _empty_search_runs_filters()

    completion = _client().chat.completions.create(
        model=settings.chat_model,
        temperature=0,
        messages=[
            {"role": "system", "content": SEARCH_RUNS_FILTER_PARSING_PROMPT},
            {"role": "user", "content": msg},
        ],
    )
    raw = strip_think_tags(completion.choices[0].message.content or "")

    return _parse_search_runs_filters_from_llm_raw(raw)


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
        
        # 없는 metric_key를 입력했을 경우
        if metric_key not in list_run_metrics(target_experiment):
            return(
                f"{metric_key}가 없습니다. {target_experiment}의 metric 키 목록을 조회해 주세요.",
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
    history: list[ChatMessage] | None = None,
) -> tuple[str, str, dict]:
    if not report_text:
        return "생성된 report_text가 없어 리포트 질문에 답할 수 없습니다.", intent, {}

    report_context = {
        "report_text": report_text,
        "experiment_name": experiment_name,
        "metric_key": metric_key,
    }
    messages: list[dict[str, str]] = [
        {"role": "system", "content": REPORT_QA_SYSTEM_PROMPT},
        {
            "role": "system",
            "content": "리포트 컨텍스트(JSON):\n"
            + json.dumps(report_context, ensure_ascii=False),
        },
        *_history_to_openai_messages(history),
        {"role": "user", "content": message},
    ]
    completion = _client().chat.completions.create(
        model=settings.chat_model,
        temperature=settings.chat_temperature,
        messages=messages,
    )
    answer = strip_think_tags(completion.choices[0].message.content or "")
    return answer, intent, {"experiment_name": experiment_name, "metric_key": metric_key}


async def run_with_mcp(
    message: str,
    history: list[ChatMessage] | None = None,
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
                "이전 대화가 있으면 지시어·후속 질문을 맥락에 맞게 해석하세요.\n"
                "도구 호출이 필요하면 적극적으로 사용하세요."
            ),
            model=settings.chat_model,
            mcp_servers=manager.active_servers,
        )
        mcp_input: list[dict[str, str]] = [
            *_history_to_openai_messages(history),
            {"role": "user", "content": message},
        ]
        result = await Runner.run(
            starting_agent=agent,
            input=mcp_input,
            run_config=RunConfig(
                tracing_disabled=True,         # openapi 트레이싱 끔
                model_provider=MultiProvider(
                    openai_api_key=settings.chat_api_key,
                    openai_base_url=settings.chat_base_url,
                    openai_use_responses=False,
                ),
            ),
        )
        answer = strip_think_tags(result.final_output)
        tool_calls = extract_tool_calls(result)
        answer = f"Tool Calls: {tool_calls}\n\n {answer}"

    return answer, DEFAULT_INTENT, {"mcp_connected": True, "tool_calls": tool_calls}

