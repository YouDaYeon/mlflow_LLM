import json
import re

from fastapi import HTTPException
from openai import OpenAI

from backend.app.config import settings
from backend.app.services.mlflow_service import (
    filter_runs_by_metric_threshold,
    get_metric_keys,
    list_experiment_names,
    list_recent_experiment_names,
    search_runs_with_filters,
)

ALLOWED_INTENTS = {
    "list_experiments",
    "list_recent_experiments",
    "list_metric_keys",
    "filter_runs_by_metric",
    "search_runs",
    "report_qa",
}
DEFAULT_INTENT = "report_qa"
MAX_EXPERIMENT_PREVIEW = 20
MAX_RECENT_EXPERIMENTS = 5
MAX_METRIC_KEYS_PREVIEW = 30
MAX_FILTER_RUNS_PREVIEW = 20

INTENT_CLASSIFICATION_PROMPT = (
    "사용자 메시지를 아래 intent 중 하나로만 분류하라. "
    "반드시 JSON 한 줄로만 답하라. 형식: {\"intent\":\"...\"}\n"
    "intent 후보: list_experiments, list_recent_experiments, list_metric_keys, filter_runs_by_metric, report_qa\n"
    "규칙:\n"
    "- 실험 목록/리스트/전체 보기: list_experiments\n"
    "- 최근 실험/최신 실험: list_recent_experiments\n"
    "- metric 키 목록/지표 키 목록: list_metric_keys\n"
    "- 특정 metric 기준으로 run 필터링(예: mAP 0.98 이상 run): filter_runs_by_metric\n"
    "- run 다중 조건 검색(기간/실험자/상태/태그/top_k/전체): search_runs\n"
    "- 그 외 질문: report_qa"
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
    return OpenAI(base_url=settings.llama_base_url, api_key=settings.llama_api_key)


def _format_preview_list(items: list[str], limit: int) -> tuple[str, str]:
    preview = "\n".join(f"- {item}" for item in items[:limit])
    suffix = f"\n(너무 길어 {limit}개만 표시)" if len(items) > limit else ""
    return preview, suffix


def _format_conditions(conditions: list[tuple[str, float]]) -> str:
    return " and ".join(f"{op} {val:.4f}" for op, val in conditions)


def classify_intent(message: str) -> str:
    completion = _client().chat.completions.create(
        model=settings.llama_model,
        temperature=0,
        messages=[
            {"role": "system", "content": INTENT_CLASSIFICATION_PROMPT},
            {"role": "user", "content": message},
        ],
    )
    raw = (completion.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
        intent = parsed.get("intent", DEFAULT_INTENT)
    except json.JSONDecodeError:
        intent = DEFAULT_INTENT
    return intent if intent in ALLOWED_INTENTS else DEFAULT_INTENT


def _normalize_operator(op_text: str) -> str:
    if op_text in {"이상", ">= ", ">=", "at least"}:
        return ">="
    if op_text in {"초과", ">", "greater than"}:
        return ">"
    if op_text in {"이하", "<=", "at most"}:
        return "<="
    if op_text in {"미만", "<", "less than"}:
        return "<"
    if op_text in {"같음", "같다", "동일", "=", "==", "equal"}:
        return "=="
    return ">="


def _parse_metric_filter_query(message: str, fallback_metric_key: str | None) -> tuple[str | None, list[tuple[str, float]]]:
    metric_query = fallback_metric_key.strip() if fallback_metric_key else None
    if not metric_query:
        metric_match = re.search(
            r"(mAP\d*(?:[-_]\d+)?|precision|recall|loss|[A-Za-z][A-Za-z0-9_./-]{1,})",
            message,
            flags=re.IGNORECASE,
        )
        metric_query = metric_match.group(1) if metric_match else None

    conditions: list[tuple[str, float]] = []
    pattern_after = re.compile(r"(\d+(?:\.\d+)?)\s*(%)?\s*(이상|초과|이하|미만|같음|같다|동일)")
    pattern_before = re.compile(r"(이상|초과|이하|미만|같음|같다|동일)\s*(\d+(?:\.\d+)?)\s*(%)?")

    for match in pattern_after.finditer(message):
        value = float(match.group(1))
        if match.group(2) or value > 1.0:
            value = value / 100.0
        conditions.append((_normalize_operator(match.group(3)), value))

    for match in pattern_before.finditer(message):
        value = float(match.group(2))
        if match.group(3) or value > 1.0:
            value = value / 100.0
        conditions.append((_normalize_operator(match.group(1)), value))

    if not conditions:
        number_match = re.search(r"(\d+(?:\.\d+)?)\s*(%)?", message)
        if number_match:
            value = float(number_match.group(1))
            if number_match.group(2) or value > 1.0:
                value = value / 100.0
            conditions.append((">=", value))

    return metric_query, conditions


def _parse_run_search_query(message: str) -> dict:
    lower_msg = message.lower()
    recent_days = None
    m_days = re.search(r"최근\s*(\d+)\s*일", message)
    if m_days:
        recent_days = int(m_days.group(1))

    status = None
    if "running" in lower_msg:
        status = "RUNNING"
    elif "finished" in lower_msg or "완료" in message:
        status = "FINISHED"
    elif "failed" in lower_msg or "실패" in message:
        status = "FAILED"
    elif "killed" in lower_msg or "중단" in message:
        status = "KILLED"

    user = None
    m_user = re.search(r"(?:실험자|사용자|유저)\s*[:=]?\s*([A-Za-z0-9._-]+)", message)
    if m_user:
        user = m_user.group(1).strip()

    tag_key = None
    tag_value = None
    m_tag = re.search(r"태그\s*[:=]?\s*([A-Za-z0-9._/-]+)\s*[:=]\s*([A-Za-z0-9._/-]+)", message)
    if m_tag:
        tag_key = m_tag.group(1).strip()
        tag_value = m_tag.group(2).strip()

    top_k = None
    if "모든 run" not in message and "전체 run" not in message:
        m_top = re.search(r"(?:top[_\s-]*k|상위)\s*(\d+)", lower_msg)
        if m_top:
            top_k = int(m_top.group(1))

    return {
        "recent_days": recent_days,
        "status": status,
        "user": user,
        "tag_key": tag_key,
        "tag_value": tag_value,
        "top_k": top_k,
    }

def answer_experiment_question(
    intent: str,
    experiment_name: str | None,
    message: str | None,
    metric_key: str | None,
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

    if intent == "filter_runs_by_metric":
        metric_query, conditions = _parse_metric_filter_query(message=message or "", fallback_metric_key=metric_key)
        if not metric_query:
            return (
                "필터링할 metric을 선택하세요.",
                intent,
                {"experiment_name": target_experiment},
            )
        if not conditions:
            return (
                "임계값 조건을 찾지 못했습니다. 예: 'mAP 0.8 이상이고 0.98 미만 run 보여줘'",
                intent,
                {"experiment_name": target_experiment, "metric_key": metric_query},
            )
        try:
            resolved_metric_key, runs = filter_runs_by_metric_threshold(
                experiment_name=target_experiment,
                metric_query=metric_query,
                conditions=conditions,
                limit=20,
            )
        except HTTPException as exc:
            if exc.status_code == 404:
                return (
                    str(exc.detail),
                    intent,
                    {"experiment_name": target_experiment, "metric_key": metric_query},
                )
            raise

        if not runs:
            condition_text = _format_conditions(conditions)
            return (
                f"{target_experiment}에서 `{resolved_metric_key}` {condition_text} 조건을 만족하는 run이 없습니다.",
                intent,
                {"experiment_name": target_experiment, "metric_key": resolved_metric_key, "count": 0},
            )
        preview = "\n".join(
            f"- {run.get('run_name') or run.get('run_id')} ({run.get('metric_score'):.4f})"
            for run in runs[:MAX_FILTER_RUNS_PREVIEW]
        )
        condition_text = _format_conditions(conditions)
        return (
            f"{target_experiment} 필터 결과({len(runs)}개)\n"
            f"조건: `{resolved_metric_key}` {condition_text}\n{preview}",
            intent,
            {"experiment_name": target_experiment, "metric_key": resolved_metric_key, "count": len(runs)},
        )

    if intent == "search_runs":
        filters = _parse_run_search_query(message or "")
        if filters["top_k"] and not metric_key:
            return (
                "top_k 조회에는 metric_key가 필요합니다. metric을 선택하거나 질문에 metric을 포함해 주세요.",
                intent,
                {"experiment_name": target_experiment, "filters": filters},
            )
        if filters["user"]:
            return(
                f"user는 {filters['user']} 입니다.",
                intent,
                {"experiment_name": target_experiment, "filters": filters},
            )
        runs = search_runs_with_filters(
            target_experiment,
            recent_days=filters["recent_days"],
            user=filters["user"],
            status=filters["status"],
            tag_key=filters["tag_key"],
            tag_value=filters["tag_value"],
            top_k=filters["top_k"],
            metric_key=metric_key,
        )
        if not runs:
            return (
                "조건에 맞는 run이 없습니다.",
                intent,
                {"experiment_name": target_experiment, "count": 0, "filters": filters},
            )
        preview = "\n".join(
            f"- {run.get('run_name') or run.get('run_id')} | {run.get('status')} | {run.get('start_time') or '-'} | {run.get('created_by') or '-'}"
            for run in runs[:MAX_FILTER_RUNS_PREVIEW]
        )
        return (
            f"{target_experiment} run 조회 결과({len(runs)}개)\n{preview}",
            intent,
            {"experiment_name": target_experiment, "count": len(runs), "filters": filters},
        )

    # list_metric_keys
    keys = get_metric_keys(target_experiment)
    if not keys:
        return f"{target_experiment}의 metric 키가 없습니다.", intent, {"experiment_name": target_experiment, "count": 0}
    preview, suffix = _format_preview_list(keys, MAX_METRIC_KEYS_PREVIEW)
    return (
        f"{target_experiment} metric 키 목록({len(keys)}개)\n{preview}{suffix}",
        intent,
        {"experiment_name": target_experiment, "count": len(keys)},
    )


def answer_report_question(
    message: str,
    report_text: str | None,
    experiment_name: str | None,
    metric_key: str | None,
) -> tuple[str, str, dict]:

    if not report_text:
        return "생성된 report_text가 없어 리포트 질문에 답할 수 없습니다.", DEFAULT_INTENT, {}

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
    return answer, DEFAULT_INTENT, {"experiment_name": experiment_name, "metric_key": metric_key}



