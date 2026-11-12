"""챗봇 질문을 구조화된 요청으로 해석하고, 그 구조대로 조회·답변한다.

흐름
1. parse_request: LLM이 질문을 요청 구조(JSON)로 바꾼다.
2. normalize_request: 허용 값·타입만 남긴다. 잘못된 값은 버리되 조건을 몰래 바꾸지 않는다.
3. answer_request: 대상(실험·run·metric)을 확정하고, 작업과 답변 범위에 맞게 조회·답변한다.
   - 값의 출처(질문, 이전 대화, UI 선택, 기본값)를 구분해 답변 첫 줄에 적용 조건으로 보여 준다.
   - 필요한 정보가 없으면 확인 질문을, 지원하지 않는 작업이면 지원 범위를 안내한다.
   - 위 구조로 표현할 수 없는 질문(target=other)은 None을 돌려주고 라우터가 MCP로 보낸다.
"""

from __future__ import annotations

import functools
import json
import math
import re
import statistics
import sys
from datetime import datetime, timedelta
from typing import Any

from backend.app.config import settings
from backend.app.schemas import ChatMessage, ExpDisplayRow, RunDisplayRow, split_run_start_time
from backend.app.services import chat_service as cs
from backend.app.services.mlflow_service import (
    find_runs,
    get_experiment,
    get_runs,
    list_experiment_names,
    list_recent_experiments,
    list_run_metrics,
    record_tool_call,
)

TARGETS = frozenset({"experiment", "run", "metric_key", "report", "other"})
ACTIONS = frozenset({"list", "count", "lookup", "compare", "summarize", "evaluate", "unsupported"})
EXPERIMENT_PICKS = frozenset({"latest", "oldest", "latest_updated"})
SORT_FIELDS = frozenset({"metric", "time", "update_time"})
ORDERS = frozenset({"desc", "asc"})
OUTPUTS = frozenset({"names", "count", "value", "brief", "table", "full"})
FIELDS = frozenset({"params", "metrics", "status", "time", "id"})
RUN_STATUSES = frozenset({"RUNNING", "FINISHED", "FAILED", "KILLED"})
UNRESOLVED_ITEMS = frozenset({"experiment", "run", "metric"})
CONTEXT_FIELDS = frozenset({"experiment", "runs", "metric", "filters", "sort_by", "order", "limit"})
CLEARED_LABELS = {
    "experiment": "실험",
    "runs": "대상 run",
    "metric": "metric",
    "filters": "조건(기간·상태 등)",
    "sort_by": "정렬",
    "limit": "개수",
}

DEFAULT_RECENT_EXPERIMENTS = 5
MAX_EXPERIMENT_ROWS = 20
MAX_RUN_ROWS = 20
MAX_RUN_NAMES = 50
DEFAULT_COMPARE_RUNS = 2
MAX_COMPARE_RUNS = 5
MAX_COMPARE_METRICS = 12
MAX_EVAL_METRICS = 8

_RUN_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")
# 낮을수록 좋은 metric으로 보는 이름 조각
_LOWER_IS_BETTER = ("loss", "error", "err", "mae", "mse", "rmse", "wer", "cer", "perplexity")

REQUEST_PARSING_PROMPT = (
    "당신은 MLflow 실험 QA 챗봇의 요청 해석기다.\n"
    "사용자 질문을 실행할 수 있는 요청 구조로 바꿔 JSON 한 줄로만 출력한다. 설명하지 않는다.\n\n"

    "=== 필드 ===\n"
    "- target: 무엇에 대한 요청인가.\n"
    "  experiment(실험) | run | metric_key(metric 이름 목록) | report(첨부된 보고서 내용) | "
    "other(위 넷으로 표현할 수 없음: trace, 모델 레지스트리, 아티팩트 파일 내용 등)\n"
    "- action: 무엇을 하는가.\n"
    "  list(나열) | count(개수) | lookup(정해진 대상의 정보·값 조회) | compare(둘 이상 비교, 차이) | "
    "summarize(핵심 요약) | evaluate(잘 됐는지 평가·판단·원인 해석) | "
    "unsupported(삭제·수정·생성·학습 실행처럼 조회가 아닌 작업)\n"
    "- experiment: 실험 이름. 사용자가 말한 그대로.\n"
    "- experiment_pick: 실험 이름 없이 시간으로 실험을 고를 때. "
    "latest(가장 최근 생성) | oldest(가장 오래된) | latest_updated(가장 최근 업데이트)\n"
    "- runs: 지목한 run 이름 또는 32자리 run id 목록. 예: brave-owl-12, trial_7, exp3-hpo-05. "
    "실험 이름은 넣지 않는다.\n"
    "- metric: 기준 metric 이름. 사용자가 말한 그대로(예: f1, val_loss, metrics/mAP50-95B).\n"
    "- filters: run 선택 조건 객체. recent_days(최근 N일 정수), status(RUNNING|FINISHED|FAILED|KILLED), "
    "user(실험자), tag_key와 tag_value(둘 다 있을 때만)\n"
    "  실패=FAILED, 완료/성공=FINISHED, 실행중=RUNNING, 중단=KILLED\n"
    "- sort_by: metric(metric 값 순) | time(생성·시작 시각 순) | update_time(최근 업데이트 순, 실험 전용)\n"
    "- order: desc | asc. '높은/좋은/상위/최근/최신'=desc, '낮은/하위/오래된'=asc. "
    "loss·error가 '가장 좋은'이면 asc.\n"
    "- limit: 사용자가 말한 결과 개수. '하나/한 개/제일·가장 ~한 것'=1, 'N개/top N/상위 N'=N, "
    "'전체/모두/다'=\"all\". 말하지 않았으면 생략.\n"
    "- pick: \"both_ends\" — '최고와 최저'처럼 양 끝 두 개를 고를 때만.\n"
    "- output: 답변 범위. names(이름만) | count(개수만) | value(값 하나·몇 개) | brief(짧게, 한 줄) | "
    "table(목록을 표로) | full(상세 정보 전체)\n"
    "  '몇 개'=count, '이름만'=names, '값/얼마/몇이야'=value, '상세/자세히/정보 전체'=full, '요약/핵심/간단히'=brief\n"
    "- fields: 특정 항목만 원할 때. params(파라미터·설정) | metrics | status | time | id\n"
    "- inherited: 현재 질문에는 없고 이전 대화에서 이어받은 필드 이름 목록.\n"
    "- cleared: 이전 대화의 조건 중 현재 질문이 해제한 필드 이름 목록.\n"
    "- unresolved: 답하는 데 꼭 필요한데 질문과 이전 대화 어디에도 없는 항목. experiment | run | metric\n"
    "- unsupported_reason: action=unsupported일 때 짧은 이유.\n"
    "inherited·cleared에 쓰는 필드 이름: experiment, runs, metric, filters, sort_by, order, limit\n\n"

    "=== 규칙 ===\n"
    "1. 질문에 있는 조건(대상, 개수, 기간, 상태, 정렬, 답변 범위)을 하나도 버리지 않는다.\n"
    "2. 질문에 없는 값을 만들지 않는다. limit, metric, sort_by, filters를 추측하지 않는다.\n"
    "3. limit은 개수, sort_by는 기준이다. '최근 run 3개'는 sort_by=time, limit=3이다(metric 순위 아님).\n"
    "4. '가장 최근 실험의 run 중 ~'처럼 시간으로 실험을 고른 뒤 run을 물으면 target=run과 experiment_pick을 함께 쓴다.\n"
    "5. 'metric이 가장 높은 run의 파라미터'처럼 순위로 run을 고른 뒤 정보를 물으면 "
    "action=lookup, sort_by=metric, limit=1, fields를 함께 쓴다.\n"
    "6. 이전 대화 이어받기\n"
    "   - '그 실험/그 run/거기/그 중에'처럼 가리키거나 대상 없이 이어 물으면 직전 대상(experiment, runs)을 "
    "이어받고 inherited에 적는다. 직전 요청 해석과 직전 답변의 대상을 참고한다.\n"
    "   - 조건(filters, metric, sort_by, order, limit)은 '그 중에/거기서'처럼 직전 결과를 좁힐 때만 이어받는다. "
    "같은 종류의 새 조건이면 바꾼다(예: 실패 → 완료).\n"
    "   - '전체/모두/조건 빼고'는 이전 조건을 해제하고 cleared에 적는다.\n"
    "   - 직전 대상이 여러 개라 무엇을 가리키는지 모르면 unresolved에 넣는다.\n"
    "7. 이전 대화가 없는데 '그 run'처럼 가리키는 대상이 없으면 unresolved에 넣는다. "
    "run을 고르는 데 실험이나 기준 metric이 필요한데 없으면 unresolved에 넣는다.\n"
    "8. '보고서/리포트 기준·기반'이면 target=report다. action, limit, output은 그대로 표현한다.\n"
    "9. 값이 없는 키는 생략하고 JSON 한 줄만 출력한다.\n\n"

    "=== 예시 ===\n"
    "사용자: 실험 리스트 줘\n"
    "{\"target\":\"experiment\",\"action\":\"list\",\"output\":\"names\"}\n\n"
    "사용자: 요즘 만든 실험 뭐 있어?\n"
    "{\"target\":\"experiment\",\"action\":\"list\",\"sort_by\":\"time\",\"order\":\"desc\",\"output\":\"names\"}\n\n"
    "사용자: 제일 최근에 만든 실험 딱 하나만\n"
    "{\"target\":\"experiment\",\"action\":\"list\",\"sort_by\":\"time\",\"order\":\"desc\",\"limit\":1,\"output\":\"brief\"}\n\n"
    "사용자: 최근 수정된 실험 2개 자세히\n"
    "{\"target\":\"experiment\",\"action\":\"list\",\"sort_by\":\"update_time\",\"order\":\"desc\",\"limit\":2,\"output\":\"full\"}\n\n"
    "사용자: resnet-ablation 실험 정보\n"
    "{\"target\":\"experiment\",\"action\":\"lookup\",\"experiment\":\"resnet-ablation\",\"output\":\"full\"}\n\n"
    "사용자: resnet-ablation에 run 몇 개 있어?\n"
    "{\"target\":\"run\",\"action\":\"count\",\"experiment\":\"resnet-ablation\",\"output\":\"count\"}\n\n"
    "사용자: resnet-ablation metric 뭐 있어?\n"
    "{\"target\":\"metric_key\",\"action\":\"list\",\"experiment\":\"resnet-ablation\"}\n\n"
    "사용자: resnet-ablation에서 f1 상위 4개\n"
    "{\"target\":\"run\",\"action\":\"list\",\"experiment\":\"resnet-ablation\",\"metric\":\"f1\",\"sort_by\":\"metric\",\"order\":\"desc\",\"limit\":4,\"output\":\"table\"}\n\n"
    "사용자: resnet-ablation에서 val_loss 제일 낮은 run\n"
    "{\"target\":\"run\",\"action\":\"list\",\"experiment\":\"resnet-ablation\",\"metric\":\"val_loss\",\"sort_by\":\"metric\",\"order\":\"asc\",\"limit\":1,\"output\":\"brief\"}\n\n"
    "사용자: resnet-ablation에서 지난 2주 동안 alice가 돌린 중단된 run 이름만\n"
    "{\"target\":\"run\",\"action\":\"list\",\"experiment\":\"resnet-ablation\",\"filters\":{\"recent_days\":14,\"status\":\"KILLED\",\"user\":\"alice\"},\"output\":\"names\"}\n\n"
    "사용자: resnet-ablation 최신 run 다섯 개\n"
    "{\"target\":\"run\",\"action\":\"list\",\"experiment\":\"resnet-ablation\",\"sort_by\":\"time\",\"order\":\"desc\",\"limit\":5,\"output\":\"table\"}\n\n"
    "사용자: 제일 최근 실험에서 f1 가장 좋은 run의 설정값만\n"
    "{\"target\":\"run\",\"action\":\"lookup\",\"experiment_pick\":\"latest\",\"metric\":\"f1\",\"sort_by\":\"metric\",\"order\":\"desc\",\"limit\":1,\"fields\":[\"params\"]}\n\n"
    "사용자: brave-owl-12 run val_loss 얼마야?\n"
    "{\"target\":\"run\",\"action\":\"lookup\",\"runs\":[\"brave-owl-12\"],\"metric\":\"val_loss\",\"output\":\"value\"}\n\n"
    "사용자: brave-owl-12 run 전체 정보\n"
    "{\"target\":\"run\",\"action\":\"lookup\",\"runs\":[\"brave-owl-12\"],\"output\":\"full\"}\n\n"
    "사용자: brave-owl-12 간단히 정리해줘\n"
    "{\"target\":\"run\",\"action\":\"summarize\",\"runs\":[\"brave-owl-12\"],\"output\":\"brief\"}\n\n"
    "사용자: brave-owl-12 결과 괜찮은 거야?\n"
    "{\"target\":\"run\",\"action\":\"evaluate\",\"runs\":[\"brave-owl-12\"]}\n\n"
    "사용자: brave-owl-12랑 calm-fox-3 f1 비교\n"
    "{\"target\":\"run\",\"action\":\"compare\",\"runs\":[\"brave-owl-12\",\"calm-fox-3\"],\"metric\":\"f1\",\"output\":\"value\"}\n\n"
    "사용자: resnet-ablation에서 f1 최고 run이랑 최저 run 차이 보여줘\n"
    "{\"target\":\"run\",\"action\":\"compare\",\"experiment\":\"resnet-ablation\",\"metric\":\"f1\",\"sort_by\":\"metric\",\"order\":\"desc\",\"pick\":\"both_ends\"}\n\n"
    "사용자: 리포트 기준으로 제일 좋은 run 요약해줘\n"
    "{\"target\":\"report\",\"action\":\"summarize\",\"limit\":1,\"output\":\"brief\"}\n\n"
    "사용자(새 대화): 이 run 왜 이렇게 낮아?\n"
    "{\"target\":\"run\",\"action\":\"evaluate\",\"unresolved\":[\"run\"]}\n\n"
    "사용자(새 대화): 정확도 제일 높은 run\n"
    "{\"target\":\"run\",\"action\":\"list\",\"metric\":\"정확도\",\"sort_by\":\"metric\",\"order\":\"desc\",\"limit\":1,\"unresolved\":[\"experiment\"]}\n\n"
    "사용자: brave-owl-12 run 지워줘\n"
    "{\"target\":\"run\",\"action\":\"unsupported\",\"runs\":[\"brave-owl-12\"],\"unsupported_reason\":\"run 삭제\"}\n\n"
    "사용자: 어제 trace 중 에러 난 것\n"
    "{\"target\":\"other\",\"action\":\"list\"}\n\n"
    "직전 요청 해석: {\"target\":\"run\",\"action\":\"list\",\"experiment\":\"resnet-ablation\",\"metric\":\"f1\",\"sort_by\":\"metric\",\"order\":\"desc\",\"limit\":4}\n"
    "사용자: 다 보여줘\n"
    "{\"target\":\"run\",\"action\":\"list\",\"experiment\":\"resnet-ablation\",\"limit\":\"all\",\"output\":\"table\",\"inherited\":[\"experiment\"],\"cleared\":[\"metric\",\"sort_by\",\"order\",\"limit\"]}\n\n"
    "직전 요청 해석: {\"target\":\"run\",\"action\":\"list\",\"experiment\":\"resnet-ablation\",\"filters\":{\"status\":\"KILLED\"}}\n"
    "사용자: 성공한 건?\n"
    "{\"target\":\"run\",\"action\":\"list\",\"experiment\":\"resnet-ablation\",\"filters\":{\"status\":\"FINISHED\"},\"inherited\":[\"experiment\"]}\n\n"
    "직전 답변의 대상: run=[brave-owl-12]\n"
    "사용자: 그거 파라미터는?\n"
    "{\"target\":\"run\",\"action\":\"lookup\",\"runs\":[\"brave-owl-12\"],\"fields\":[\"params\"],\"inherited\":[\"runs\"]}\n"
)

RUN_SUMMARY_PROMPT = (
    "당신은 MLflow run을 요약하는 어시스턴트다.\n"
    "1) 제공된 run 상세만 근거로 한다. 없는 값은 만들지 않는다.\n"
    "2) 3줄 이내로 답한다: 무엇을 어떻게 학습했는지(모델·주요 설정), 결과(주요 metric 값), 눈에 띄는 점.\n"
    "3) 표나 파라미터 전체 목록을 다시 출력하지 않는다. system metric은 질문하지 않으면 언급하지 않는다.\n"
    "4) 한국어로 답한다."
)

RUN_EVALUATION_PROMPT = (
    "당신은 MLflow run을 평가하는 어시스턴트다.\n"
    "1) 제공된 run 상세와 '같은 실험 비교 기준' 표만 근거로 한다. 없는 값은 만들지 않는다.\n"
    "2) 판단은 같은 실험의 다른 run 대비 위치(순위, 최고·중앙값과의 차이)를 기준으로 한다. "
    "절대적인 좋고 나쁨은 근거가 없으면 단정하지 않는다.\n"
    "3) 검증·테스트 metric(val_*, test_*, metrics/*)을 학습 metric(train_*, batch_*)보다 우선한다. "
    "학습 metric만 좋고 검증 metric이 낮으면 그 차이를 밝힌다.\n"
    "4) 최고와 최저의 차이가 작으면(예: 정확도 1%p 미만) 순위 차이가 의미 있게 크지 않다고 밝힌다.\n"
    "5) 첫 문장에 결론을 쓴다. 이어서 근거 metric 2~3개를 이름·값·순위(예: val_accuracy 77.81, 3개 중 2위)로 "
    "정확히 인용한다. metric 이름을 바꿔 부르지 않는다.\n"
    "6) 비교할 run이 1개뿐이면 비교 근거가 없다고 밝히고 train/val 차이 같은 내부 근거만 쓴다.\n"
    "7) 개선 제안은 근거가 있을 때만 한 줄로 덧붙인다.\n"
    "8) run 상세 표나 파라미터 목록을 다시 출력하지 않는다. 한국어로 간결하게 답한다."
)


# ── 1. 해석 ─────────────────────────────────────────────


def _format_previous_context(previous: dict[str, Any] | None) -> str:
    if not previous:
        return ""
    lines: list[str] = []
    request = previous.get("request")
    if isinstance(request, dict) and request:
        compact = {k: v for k, v in request.items() if v not in (None, [], {}, False)}
        lines.append("직전 요청 해석: " + json.dumps(compact, ensure_ascii=False))
    resolved = previous.get("resolved") or {}
    bits: list[str] = []
    if resolved.get("experiment"):
        bits.append(f"실험={resolved['experiment']}")
    if resolved.get("runs"):
        bits.append("run=[" + ", ".join(resolved["runs"]) + "]")
    if resolved.get("metric"):
        bits.append(f"metric={resolved['metric']}")
    if bits:
        lines.append("직전 답변의 대상: " + ", ".join(bits))
    return "\n".join(lines)


def _format_history_for_parser(
    history: list[ChatMessage] | None,
    *,
    max_assistant_chars: int = 400,
) -> str:
    """요청 파서 입력용으로 이전 대화를 짧게 직렬화한다. 비어 있으면 빈 문자열."""
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


def build_parser_input(
    message: str,
    history: list[ChatMessage] | None,
    previous: dict[str, Any] | None,
) -> str:
    parts: list[str] = []
    history_text = _format_history_for_parser(history)
    if history_text:
        parts.append("이전 대화(참고용):\n" + history_text)
    previous_text = _format_previous_context(previous)
    if previous_text:
        parts.append(previous_text)
    if not parts:
        return message
    parts.append("현재 질문:\n" + message)
    return "\n\n".join(parts)


def parse_request(
    message: str,
    history: list[ChatMessage] | None = None,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """질문을 요청 구조로 해석한다. 해석에 실패하면 target=other(MCP 처리)로 둔다."""
    completion = cs.llm_client().chat.completions.create(
        model=settings.chat_model,
        temperature=0,
        messages=[
            {"role": "system", "content": REQUEST_PARSING_PROMPT},
            {"role": "user", "content": build_parser_input(message, history, previous)},
        ],
    )
    raw = cs.strip_think_tags(completion.choices[0].message.content or "")
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    data: Any = None
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
    request = normalize_request(data)
    if data is None:
        request["parse_error"] = raw[:200]
    return request


# 답변 metadata의 route.parser 값. 요청 해석은 이 파서 하나로 한다.
PARSER_NAME = parse_request.__name__


def _clean_text(value: object) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
        return number if number > 0 else None
    return None


def _enum(value: object, allowed: frozenset[str], *, upper: bool = False) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    text = text.upper() if upper else text.lower()
    return text if text in allowed else None


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _clean_text(item)
        if text and text not in out:
            out.append(text)
    return out


def _enum_list(value: object, allowed: frozenset[str]) -> list[str]:
    return [item for item in (s.lower() for s in _string_list(value)) if item in allowed]


def empty_request() -> dict[str, Any]:
    return {
        "target": "other",
        "action": "list",
        "experiment": None,
        "experiment_pick": None,
        "runs": [],
        "metric": None,
        "filters": {},
        "sort_by": None,
        "order": None,
        "limit": None,
        "all": False,
        "pick": None,
        "output": None,
        "fields": [],
        "inherited": [],
        "cleared": [],
        "unresolved": [],
        "unsupported_reason": None,
    }


def normalize_request(raw: object) -> dict[str, Any]:
    """파서 출력을 허용 값과 타입으로 정리한다."""
    req = empty_request()
    if not isinstance(raw, dict):
        return req

    req["experiment"] = _clean_text(raw.get("experiment"))
    req["experiment_pick"] = _enum(raw.get("experiment_pick"), EXPERIMENT_PICKS)
    req["runs"] = _string_list(raw.get("runs"))
    req["metric"] = _clean_text(raw.get("metric"))

    filters: dict[str, Any] = {}
    raw_filters = raw.get("filters") if isinstance(raw.get("filters"), dict) else {}
    days = _positive_int(raw_filters.get("recent_days"))
    if days:
        filters["recent_days"] = days
    status = _enum(raw_filters.get("status"), RUN_STATUSES, upper=True)
    if status:
        filters["status"] = status
    user = _clean_text(raw_filters.get("user"))
    if user:
        filters["user"] = user
    tag_key = _clean_text(raw_filters.get("tag_key"))
    tag_value = _clean_text(raw_filters.get("tag_value"))
    if tag_key and tag_value:
        filters["tag_key"] = tag_key
        filters["tag_value"] = tag_value
    req["filters"] = filters

    req["sort_by"] = _enum(raw.get("sort_by"), SORT_FIELDS)
    req["order"] = _enum(raw.get("order"), ORDERS)
    limit_raw = raw.get("limit")
    if isinstance(limit_raw, str) and limit_raw.strip().lower() in {"all", "전체"}:
        req["all"] = True
    else:
        req["limit"] = _positive_int(limit_raw)
    req["pick"] = "both_ends" if _clean_text(raw.get("pick")) == "both_ends" else None
    req["output"] = _enum(raw.get("output"), OUTPUTS)
    req["fields"] = _enum_list(raw.get("fields"), FIELDS)
    req["inherited"] = _enum_list(raw.get("inherited"), CONTEXT_FIELDS)
    req["cleared"] = _enum_list(raw.get("cleared"), CONTEXT_FIELDS)
    req["unresolved"] = _enum_list(raw.get("unresolved"), UNRESOLVED_ITEMS)
    req["unsupported_reason"] = _clean_text(raw.get("unsupported_reason"))

    target = _enum(raw.get("target"), TARGETS)
    if target is None:
        target = "run" if req["runs"] else "other"
    req["target"] = target
    action = _enum(raw.get("action"), ACTIONS)
    if action is None:
        action = "lookup" if req["runs"] else "list"
    req["action"] = action
    return req


# ── 2. 공통 도우미 ───────────────────────────────────────


def _to_epoch_seconds(value: object) -> float | None:
    if value is None:
        return None
    # get_runs()의 start_time은 pandas Timestamp(datetime 하위 클래스)다. 문자열 변환 없이 바로 쓴다.
    if isinstance(value, datetime):
        try:
            return value.timestamp()
        except (ValueError, OSError):
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


# 뒤에 한국어 조사(인/의/은)가 붙어도 run 이름으로 본다. \b는 한글을 단어 문자로 봐서 경계가 안 생긴다.
_MLFLOW_RUN_NAME_RE = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z]+(?:-[A-Za-z0-9]+)+-\d+(?![A-Za-z0-9_-])"
)


_MLFLOW_RUN_ID_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32}(?![0-9a-fA-F])")


def _fmt_metric_value(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if number != number or number in (float("inf"), float("-inf")):
        return "-"
    return f"{number:.6g}"


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_run_detail_markdown(records: list[dict]) -> str:
    """run 하나(또는 동명 run 여러 개)의 파라미터·메트릭을 마크다운으로 만든다."""
    ordered = sorted(
        records,
        key=lambda r: str(r.get("start_time") or ""),
        reverse=True,
    )
    primary = ordered[0]
    run_name = str(primary.get("tags.mlflow.runName") or "-")
    lines = [f"`{run_name}` run 정보"]
    outside = primary.get("matched_outside_experiment")
    if outside:
        lines.append(
            f"요청한 실험 `{outside}`에는 없고, `{primary.get('experiment_name', '-')}`에서 찾았습니다."
        )
    if len(ordered) > 1:
        lines.append(f"같은 이름의 run이 {len(ordered)}개입니다. 가장 최근 run을 표시합니다.")
    lines.append("")

    date_s, time_s = split_run_start_time(primary.get("start_time"))
    end_date, end_time = split_run_start_time(primary.get("end_time"))
    started = date_s if time_s == "-" else f"{date_s} {time_s}"
    ended = end_date if end_time == "-" else f"{end_date} {end_time}"
    lines.extend(
        [
            f"- Experiment: `{primary.get('experiment_name', '-')}` (ID `{primary.get('experiment_id', '-')}`)",
            f"- Run name: `{run_name}`",
            f"- Run ID: `{primary.get('run_id', '-')}`",
            f"- Status: {primary.get('status', '-')}",
            f"- User: {primary.get('tags.mlflow.user', '-')}",
            f"- Start: {started}",
            f"- End: {ended}",
        ]
    )

    params = sorted(
        (key.removeprefix("params."), value)
        for key, value in primary.items()
        if key.startswith("params.")
    )
    model_metrics = []
    system_metrics = []
    for key, value in primary.items():
        if not key.startswith("metrics."):
            continue
        metric_name = key.removeprefix("metrics.")
        bucket = system_metrics if metric_name.startswith("system/") else model_metrics
        bucket.append((metric_name, value))

    def _table(title: str, rows: list[tuple[str, object]]) -> None:
        lines.extend(["", f"**{title}**", "", "| Key | Value |", "|-----|-------|"])
        if not rows:
            lines.append("| - | - |")
            return
        for key, value in rows:
            shown = _fmt_metric_value(value) if title != "Parameters" else value
            lines.append(f"| {_md_cell(key)} | {_md_cell(shown)} |")

    _table("Parameters", params)
    _table("Metrics", sorted(model_metrics))
    if system_metrics:
        _table("System metrics", sorted(system_metrics))

    if len(ordered) > 1:
        lines.extend(["", "**같은 이름의 다른 run**", ""])
        for record in ordered[1:]:
            lines.append(
                f"- `{record.get('experiment_name', '-')}` / "
                f"`{record.get('run_id', '-')}` / {record.get('status', '-')}"
            )
    return "\n".join(lines)


def _run_detail_metadata(records: list[dict]) -> dict:
    if not records:
        return {"count": 0}
    primary = sorted(records, key=lambda r: str(r.get("start_time") or ""), reverse=True)[0]
    return {
        "count": len(records),
        "experiment_name": primary.get("experiment_name"),
        "experiment_id": str(primary.get("experiment_id", "")),
        "run_name": primary.get("tags.mlflow.runName"),
        "run_id": primary.get("run_id"),
    }


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _epoch(value: object) -> float | None:
    """start_time(ms, ISO 문자열, datetime/Timestamp)을 epoch 초로 바꾼다."""
    return _to_epoch_seconds(value)


def _run_name(record: dict) -> str:
    return str(record.get("tags.mlflow.runName") or record.get("run_id") or "-")


def _start_text(record: dict) -> str:
    date_s, time_s = split_run_start_time(record.get("start_time"))
    return date_s if time_s == "-" else f"{date_s} {time_s}"


def _metric_name(column: str) -> str:
    return column.removeprefix("metrics.")


def _model_metric_columns(record: dict) -> list[str]:
    return sorted(
        key
        for key, value in record.items()
        if key.startswith("metrics.")
        and not key.startswith("metrics.system/")
        and _finite(value) is not None
    )


def resolve_metric_column(metric: str, columns: list[str]) -> tuple[str | None, list[str]]:
    """사용자가 말한 metric을 실제 컬럼(metrics.xxx)으로 바꾼다.

    접두사 유무, 대소문자, 'mAP50B' → 'metrics/mAP50B'처럼 경로 끝 일치, 유일한 부분 일치를 허용한다.
    반환: (컬럼 또는 None, 후보 목록). 후보가 여럿이면 컬럼은 None이다.
    """
    wanted = metric.strip().removeprefix("metrics.")
    names = {_metric_name(c): c for c in columns if c.startswith("metrics.")}
    if wanted in names:
        return names[wanted], []
    lowered = wanted.lower()
    exact = [c for n, c in names.items() if n.lower() == lowered]
    if len(exact) == 1:
        return exact[0], []
    tail = [c for n, c in names.items() if n.lower().split("/")[-1] == lowered]
    if len(tail) == 1:
        return tail[0], []
    if len(tail) > 1:
        return None, tail
    model_names = {n: c for n, c in names.items() if not n.startswith("system/")}
    partial = [c for n, c in model_names.items() if lowered in n.lower()]
    if len(partial) == 1:
        return partial[0], []
    return None, partial


def _metric_choices_text(columns: list[str], limit: int = 15) -> str:
    names = [_metric_name(c) for c in columns if not _metric_name(c).startswith("system/")]
    shown = ", ".join(f"`{n}`" for n in names[:limit])
    more = f" 외 {len(names) - limit}개" if len(names) > limit else ""
    return shown + more if shown else "(없음)"


def _source_suffix(source: str | None) -> str:
    return "" if source in (None, "질문") else f"({source})"


def _route_step(func):
    """분기 함수에 들어갈 때 이름을 ctx.path에 남긴다. 첫 인자는 _Context여야 한다."""

    @functools.wraps(func)
    def wrapper(ctx: "_Context", *args: Any, **kwargs: Any):
        ctx.path.append(func.__name__)
        return func(ctx, *args, **kwargs)

    return wrapper


class _Context:
    """요청 하나를 처리하는 동안 확정한 값과 출처를 모은다."""

    def __init__(
        self,
        req: dict[str, Any],
        *,
        message: str,
        history: list[ChatMessage] | None,
        report_text: str | None,
        ui_experiment: str | None,
        ui_metric: str | None,
        previous: dict[str, Any] | None,
    ) -> None:
        self.req = req
        self.message = message
        self.history = history
        self.report_text = report_text
        self.ui_experiment = (ui_experiment or "").strip() or None
        self.ui_metric = (ui_metric or "").strip() or None
        self.previous = previous or {}
        self.sources: dict[str, str] = {}
        self.defaults: list[str] = []
        self.resolved: dict[str, Any] = {}
        self._experiment_names: list[str] | None = None
        # 답변까지 거쳐 간 함수 이름. 분기 함수(@_route_step)에 들어갈 때마다 쌓인다.
        self.path: list[str] = ["answer_request"]

    def source_of(self, field: str) -> str:
        return "이전 대화" if field in self.req["inherited"] else "질문"

    def experiment_names(self) -> list[str]:
        if self._experiment_names is None:
            self._experiment_names = list_experiment_names()
        return self._experiment_names

    def match_experiment(self, name: str) -> str | None:
        names = self.experiment_names()
        if name in names:
            return name
        folded = name.casefold()
        matches = [n for n in names if n.casefold() == folded]
        return matches[0] if len(matches) == 1 else None

    def metadata(self, **extra: Any) -> dict[str, Any]:
        # 답변을 만든 함수 = metadata()를 부른 함수. 분기 함수가 아닌 곳(_resolve_*)에서
        # 답이 끝났으면 경로 끝에 붙여, 답변이 어떤 경로로 왔는지 끝까지 보이게 한다.
        caller = sys._getframe(1).f_code.co_name
        path = list(self.path)
        if caller not in path:
            path.append(caller)
        meta: dict[str, Any] = {
            "route": {"parser": PARSER_NAME, "path": path, "handler": caller},
            "request": self.req,
            "resolved": self.resolved,
            "applied": {"sources": self.sources, "defaults": self.defaults},
        }
        if self.resolved.get("experiment"):
            meta["experiment_name"] = self.resolved["experiment"]
        if self.resolved.get("metric_column"):
            meta["metric_key"] = self.resolved["metric_column"]
        meta.update(extra)
        return meta


# ── 3. 실행 ─────────────────────────────────────────────


def answer_request(
    req: dict[str, Any],
    *,
    message: str,
    history: list[ChatMessage] | None = None,
    report_text: str | None = None,
    ui_experiment: str | None = None,
    ui_metric: str | None = None,
    previous: dict[str, Any] | None = None,
) -> tuple[str, dict] | None:
    """요청 구조대로 답한다. MCP로 넘겨야 하면 None."""
    ctx = _Context(
        dict(req),
        message=message,
        history=history,
        report_text=report_text,
        ui_experiment=ui_experiment,
        ui_metric=ui_metric,
        previous=previous,
    )
    req = ctx.req
    if req["action"] == "unsupported":
        return _answer_unsupported(ctx)
    if req["target"] == "report":
        return _answer_report(ctx)
    if req["target"] == "other":
        return None

    _fill_inherited_targets(ctx)
    _fix_misplaced_targets(ctx)
    _fill_runs_from_message(ctx)

    # 가리키는 run이 없고 순위로 고를 수도 없으면, 실험을 묻기 전에 run부터 확인한다.
    if (
        req["target"] == "run"
        and not req["runs"]
        and "run" in req["unresolved"]
        and not req["sort_by"]
        and not req["pick"]
    ):
        return _clarify(ctx, ["run"])

    error = _resolve_experiment(ctx)
    if error:
        return error

    if req["target"] == "metric_key":
        return _answer_metric_keys(ctx)
    if req["target"] == "experiment":
        return _answer_experiments(ctx)
    if req["runs"]:
        return _answer_named_runs(ctx)
    return _answer_ranked_runs(ctx)


def _fill_inherited_targets(ctx: _Context) -> None:
    """파서가 이어받았다고 했지만 값을 비운 대상은 직전 답변의 대상으로 채운다."""
    req = ctx.req
    resolved = ctx.previous.get("resolved") or {}
    if "experiment" in req["inherited"] and not req["experiment"] and resolved.get("experiment"):
        req["experiment"] = resolved["experiment"]
    if "runs" in req["inherited"] and not req["runs"] and resolved.get("runs"):
        req["runs"] = list(resolved["runs"])


def _fix_misplaced_targets(ctx: _Context) -> None:
    """실험 이름이 runs에 들어갔거나, run 이름이 experiment에 들어간 경우를 바로잡는다."""
    req = ctx.req
    if req["runs"]:
        kept: list[str] = []
        for name in req["runs"]:
            matched = ctx.match_experiment(name)
            if matched and not req["experiment"]:
                req["experiment"] = matched
            elif not matched:
                kept.append(name)
        req["runs"] = kept
    if req["experiment"] and not ctx.match_experiment(req["experiment"]):
        if find_runs(run_name=req["experiment"]):
            req["runs"] = [req["experiment"], *req["runs"]]
            req["experiment"] = None
            if req["target"] == "experiment":
                req["target"] = "run"


def _fill_runs_from_message(ctx: _Context) -> None:
    """파서가 run 이름을 놓친 경우 질문에서 run 이름·ID 형태를 찾아 보완한다."""
    req = ctx.req
    if req["runs"] or req["target"] != "run" or req["action"] not in {"lookup", "compare", "summarize", "evaluate"}:
        return
    found = _MLFLOW_RUN_ID_RE.findall(ctx.message) + _MLFLOW_RUN_NAME_RE.findall(ctx.message)
    req["runs"] = [name for name in dict.fromkeys(found) if not ctx.match_experiment(name)]


def _needs_metric(req: dict[str, Any]) -> bool:
    return req["sort_by"] == "metric" or req["pick"] == "both_ends"


def _resolve_experiment(ctx: _Context) -> tuple[str, dict] | None:
    """실험을 확정한다. 실패하면 사용자에게 보낼 답변을 돌려준다."""
    req = ctx.req
    if req["experiment"]:
        matched = ctx.match_experiment(req["experiment"])
        if not matched:
            names = ctx.experiment_names()
            listing = ", ".join(f"`{n}`" for n in names[:MAX_EXPERIMENT_ROWS])
            return (
                f"`{req['experiment']}` 실험을 찾지 못했습니다. 등록된 실험: {listing}",
                ctx.metadata(),
            )
        ctx.resolved["experiment"] = matched
        ctx.sources["experiment"] = ctx.source_of("experiment")
        return None

    if req["experiment_pick"] and req["target"] != "experiment":
        order_by = "last_update_time" if req["experiment_pick"] == "latest_updated" else "creation_time"
        ascending = req["experiment_pick"] == "oldest"
        picked = list_recent_experiments(limit=1, order_by=order_by, ascending=ascending)
        if not picked:
            return "등록된 실험이 없습니다.", ctx.metadata()
        ctx.resolved["experiment"] = picked[0]["experiment_name"]
        ctx.sources["experiment"] = {
            "latest": "가장 최근 실험",
            "oldest": "가장 오래된 실험",
            "latest_updated": "가장 최근 업데이트된 실험",
        }[req["experiment_pick"]]
        return None

    needs_experiment = req["target"] == "metric_key" or (req["target"] == "run" and not req["runs"])
    if not needs_experiment:
        return None
    if "experiment" not in req["unresolved"]:
        previous_experiment = (ctx.previous.get("resolved") or {}).get("experiment")
        if previous_experiment and ctx.match_experiment(previous_experiment):
            ctx.resolved["experiment"] = previous_experiment
            ctx.sources["experiment"] = "이전 대화"
            return None
    if ctx.ui_experiment and ctx.match_experiment(ctx.ui_experiment):
        ctx.resolved["experiment"] = ctx.match_experiment(ctx.ui_experiment)
        ctx.sources["experiment"] = "UI 선택"
        return None
    missing = ["experiment"]
    if "metric" in req["unresolved"] or (_needs_metric(req) and not req["metric"] and not ctx.ui_metric):
        missing.append("metric")
    return _clarify(ctx, missing)


def _resolve_metric_for(ctx: _Context, columns: list[str]) -> tuple[str | None, tuple[str, dict] | None]:
    """요청 metric을 컬럼으로 확정한다. 필요 없으면 (None, None)."""
    req = ctx.req
    metric = req["metric"]
    source = ctx.source_of("metric")
    if not metric and _needs_metric(req) and "metric" not in req["unresolved"] and ctx.ui_metric:
        metric, source = ctx.ui_metric, "UI 선택"
    if not metric:
        if _needs_metric(req) or "metric" in req["unresolved"]:
            return None, _clarify(ctx, ["metric"], metric_columns=columns)
        return None, None
    column, candidates = resolve_metric_column(metric, columns)
    if column is None:
        scope = f"`{ctx.resolved.get('experiment')}`" if ctx.resolved.get("experiment") else "대상 run"
        if candidates:
            text = (
                f"`{metric}`에 해당하는 metric이 여러 개입니다: {_metric_choices_text(candidates)}. "
                "하나를 골라 다시 물어봐 주세요."
            )
        else:
            text = (
                f"{scope}에 `{metric}` metric이 없습니다. "
                f"사용할 수 있는 metric: {_metric_choices_text(columns)}"
            )
        return None, (text, ctx.metadata())
    ctx.resolved["metric"] = _metric_name(column)
    ctx.resolved["metric_column"] = column
    ctx.sources["metric"] = source
    return column, None


# ── 3-1. 지원 범위 밖, 보고서, 확인 질문 ─────────────────────


@_route_step
def _answer_unsupported(ctx: _Context) -> tuple[str, dict]:
    reason = ctx.req["unsupported_reason"]
    label = f"요청하신 작업({reason})" if reason else "요청하신 작업"
    answer = (
        f"{label}은 이 챗봇에서 할 수 없습니다. 이 챗봇은 MLflow를 조회만 합니다.\n"
        "할 수 있는 것: 실험·run 목록과 상세 조회, 조건 검색, run 비교·요약·평가, 첨부 보고서 질의응답"
    )
    return answer, ctx.metadata()


_ACTION_LABELS = {
    "list": "조회",
    "count": "개수",
    "lookup": "조회",
    "compare": "비교",
    "summarize": "요약",
    "evaluate": "평가",
}


def _scope_text(req: dict[str, Any]) -> str:
    bits = [f"작업={_ACTION_LABELS.get(req['action'], req['action'])}"]
    if req["runs"]:
        bits.append("run=" + ", ".join(req["runs"]))
    if req["metric"]:
        bits.append(f"metric={req['metric']}")
    if req["sort_by"] == "time":
        bits.append("기준=최근 순" if req["order"] != "asc" else "기준=오래된 순")
    elif req["sort_by"] == "metric":
        bits.append("기준=metric 순위")
    if req["limit"]:
        bits.append(f"개수={req['limit']}")
    if req["output"]:
        bits.append(f"답변 범위={req['output']}")
    return ", ".join(bits)


@_route_step
def _answer_report(ctx: _Context) -> tuple[str, dict]:
    if not ctx.report_text:
        answer = (
            "첨부된 보고서가 없어 보고서 기준으로 답할 수 없습니다. "
            "보고서를 생성하거나 선택한 뒤 다시 물어봐 주세요."
        )
        return answer, ctx.metadata()
    ctx.path.append("chat_service.answer_report_question")
    answer, meta = cs.answer_report_question(
        message=ctx.message,
        report_text=ctx.report_text,
        experiment_name=None,
        metric_key=None,
        history=ctx.history,
        request_scope=_scope_text(ctx.req),
    )
    return answer, ctx.metadata(**{k: v for k, v in meta.items() if v is not None})


def _understood_text(req: dict[str, Any]) -> str:
    target = {"run": "run", "experiment": "실험", "metric_key": "metric 목록"}.get(req["target"], req["target"])
    parts = [target]
    if req["metric"]:
        parts.append(f"`{req['metric']}` 기준")
    if req["sort_by"] == "metric":
        parts.append("낮은 순" if req["order"] == "asc" else "높은 순")
    if req["limit"]:
        parts.append(f"{req['limit']}개")
    parts.append(_ACTION_LABELS.get(req["action"], ""))
    return " ".join(p for p in parts if p)


@_route_step
def _clarify(
    ctx: _Context,
    missing: list[str],
    *,
    metric_columns: list[str] | None = None,
) -> tuple[str, dict]:
    lines = [f"요청({_understood_text(ctx.req)})을 처리하려면 다음을 알려주세요."]
    experiment = ctx.resolved.get("experiment")
    for item in missing:
        if item == "experiment":
            names = ctx.experiment_names()
            listing = ", ".join(f"`{n}`" for n in names[:MAX_EXPERIMENT_ROWS])
            lines.append(f"- 어느 실험에서 찾을까요? 실험: {listing}")
        elif item == "metric":
            columns = metric_columns
            if columns is None and experiment:
                columns = list_run_metrics(experiment)
            choices = f" `{experiment}`의 metric: {_metric_choices_text(columns)}" if columns else ""
            lines.append(f"- 어떤 metric 기준으로 볼까요?{choices}")
        elif item == "run":
            example = ""
            if experiment:
                runs = sorted(get_runs(experiment), key=lambda r: _epoch(r.get("start_time")) or 0, reverse=True)
                if runs:
                    example = " `" + experiment + "`의 최근 run: " + ", ".join(
                        f"`{_run_name(r)}`" for r in runs[:5]
                    )
            lines.append(f"- 어떤 run인가요? run 이름이나 32자리 run ID로 알려주세요.{example}")
    return "\n".join(lines), ctx.metadata(missing=missing)


# ── 3-2. metric 목록, 실험 ───────────────────────────────


def _recent_experiments_label(order_by: str, ascending: bool, single: bool = False) -> str:
    prefix = "가장 " if single else ""
    if order_by == "last_update_time":
        return "가장 오래전에 업데이트된 실험" if ascending else f"{prefix}최근 업데이트된 실험"
    return "가장 오래된 실험" if ascending else f"{prefix}최근 실험"


def _format_time_cell(value: object) -> str:
    date_s, time_s = split_run_start_time(value)
    return date_s if time_s == "-" else f"{date_s} {time_s}"


def _recent_experiments_answer(query: dict[str, Any]) -> tuple[str, dict]:
    limit = query["limit"]
    order_by = query["order_by"]
    ascending = query["ascending"]
    response_mode = query["response_mode"]
    label = _recent_experiments_label(order_by, ascending, single=limit == 1)
    time_key = "last_update_time" if order_by == "last_update_time" else "creation_time"
    time_label = "업데이트" if order_by == "last_update_time" else "생성"

    experiments = list_recent_experiments(limit=limit, order_by=order_by, ascending=ascending)
    metadata: dict[str, Any] = {"count": len(experiments), "recent_query": dict(query)}
    if not experiments:
        return f"{label}이 없습니다.", metadata

    names = [exp["experiment_name"] for exp in experiments]
    metadata["experiment_names"] = names
    if len(experiments) == 1:
        # 후속 질문("그 실험 run 목록")이 이 실험을 이어받도록 남긴다.
        metadata["experiment_name"] = names[0]

    if response_mode == "list":
        answer = f"{label}({len(names)}개)\n" + "\n".join(f"- {name}" for name in names)
        return answer, metadata

    if response_mode == "summary":
        if len(experiments) == 1:
            exp = experiments[0]
            answer = (
                f"{label}은 `{exp['experiment_name']}`입니다. "
                f"(ID `{exp['experiment_id']}`, {time_label} {_format_time_cell(exp.get(time_key))})"
            )
            return answer, metadata
        lines = [f"{label}({len(experiments)}개)"]
        for exp in experiments:
            lines.append(
                f"- `{exp['experiment_name']}` · ID `{exp['experiment_id']}` · "
                f"{time_label} {_format_time_cell(exp.get(time_key))}"
            )
        return "\n".join(lines), metadata

    # detail: 시간순으로 고른 실험마다 get_experiment 상세를 붙인다.
    sections: list[str] = []
    display_rows: list[dict] = []
    for exp in experiments:
        detail = get_experiment(exp["experiment_name"])
        row = ExpDisplayRow.from_experiment_dict(exp["experiment_name"], detail)
        sections.append(row.to_markdown())
        display_rows.append(row.to_metadata())
    header = f"{label}" if len(experiments) == 1 else f"{label}({len(experiments)}개)"
    if len(display_rows) == 1:
        metadata["experiment_display_row"] = display_rows[0]
    else:
        metadata["experiment_display_rows"] = display_rows
    return header + " 상세\n\n" + "\n\n---\n\n".join(sections), metadata


@_route_step
def _answer_metric_keys(ctx: _Context) -> tuple[str, dict]:
    experiment = ctx.resolved["experiment"]
    columns = list_run_metrics(experiment)
    if not columns:
        return f"`{experiment}`에 기록된 metric이 없습니다.", ctx.metadata(count=0)
    model = [_metric_name(c) for c in columns if not _metric_name(c).startswith("system/")]
    system = [_metric_name(c) for c in columns if _metric_name(c).startswith("system/")]
    lines = [f"`{experiment}` metric({len(model)}개)"]
    lines.extend(f"- {name}" for name in model)
    if system:
        lines.append(f"(system metric {len(system)}개는 생략)")
    return _join(_conditions_line(ctx), "\n".join(lines)), ctx.metadata(count=len(model))


@_route_step
def _answer_experiments(ctx: _Context) -> tuple[str, dict]:
    req = ctx.req
    names = ctx.experiment_names()

    if req["action"] == "count":
        return f"등록된 실험은 **{len(names)}개**입니다.", ctx.metadata(count=len(names))

    if req["experiment"]:
        experiment = ctx.resolved["experiment"]
        detail = get_experiment(experiment)
        row = ExpDisplayRow.from_experiment_dict(experiment, detail)
        return (
            row.to_markdown(),
            "get_experiment",
            ctx.metadata(experiment_display_row=row.to_metadata()),
        )

    time_sorted = req["sort_by"] in {"time", "update_time"} or req["experiment_pick"] is not None
    if time_sorted:
        order_by = "last_update_time" if (
            req["sort_by"] == "update_time" or req["experiment_pick"] == "latest_updated"
        ) else "creation_time"
        ascending = req["order"] == "asc" or req["experiment_pick"] == "oldest"
        limit = req["limit"]
        if req["experiment_pick"] and not limit:
            limit = 1
        if req["all"]:
            limit = len(names)
        defaulted = False
        if not limit:
            limit = DEFAULT_RECENT_EXPERIMENTS
            defaulted = True
            ctx.defaults.append(f"개수 {limit}개")
        output = req["output"]
        if output in (None, "value"):
            output = "brief" if limit == 1 else "names"
        if req["action"] == "lookup" and output not in ("names", "brief"):
            output = "full"
        response_mode = {"names": "list", "brief": "summary", "table": "summary", "full": "detail"}.get(output, "list")
        query = {
            "limit": min(limit, MAX_EXPERIMENT_ROWS if response_mode != "detail" else 5),
            "order_by": order_by,
            "ascending": ascending,
            "response_mode": response_mode,
        }
        ctx.path.append(_recent_experiments_answer.__name__)
        answer, meta = _recent_experiments_answer(query)
        if meta.get("experiment_name"):
            ctx.resolved["experiment"] = meta["experiment_name"]
        if defaulted:
            answer += f"\n\n(개수를 지정하지 않아 기본 {limit}개를 보여 드렸습니다.)"
        meta.pop("experiment_name", None)
        return answer, ctx.metadata(**meta)

    shown = names if req["all"] or not req["limit"] else names[: req["limit"]]
    preview = shown[:MAX_EXPERIMENT_ROWS]
    lines = [f"실험 목록({len(names)}개)" if len(shown) == len(names) else f"실험 {len(shown)}개 (전체 {len(names)}개)"]
    lines.extend(f"- {name}" for name in preview)
    if len(shown) > len(preview):
        lines.append(f"(너무 길어 {len(preview)}개만 표시)")
    return "\n".join(lines), ctx.metadata(count=len(names))


# ── 3-3. 조건으로 고르는 run ─────────────────────────────


def _join(header: str, body: str, notes: list[str] | None = None) -> str:
    parts = [p for p in (header, body) if p]
    text = "\n\n".join(parts)
    if notes:
        text += "\n\n" + "\n".join(f"※ {n}" for n in notes)
    return text


def _conditions_line(ctx: _Context, extra: list[str] | None = None) -> str:
    req = ctx.req
    items: list[str] = []
    experiment = ctx.resolved.get("experiment")
    if experiment:
        items.append(f"실험 `{experiment}`{_source_suffix(ctx.sources.get('experiment'))}")
    filters = req["filters"]
    filter_source = _source_suffix(ctx.source_of("filters"))
    if filters.get("recent_days"):
        items.append(f"기간 최근 {filters['recent_days']}일{filter_source}")
    if filters.get("status"):
        items.append(f"상태 {filters['status']}{filter_source}")
    if filters.get("user"):
        items.append(f"실험자 {filters['user']}{filter_source}")
    if filters.get("tag_key"):
        items.append(f"태그 {filters['tag_key']}={filters['tag_value']}{filter_source}")
    items.extend(extra or [])
    cleared = [CLEARED_LABELS[f] for f in req["cleared"] if f in CLEARED_LABELS]
    if cleared:
        items.append("이전 " + "·".join(dict.fromkeys(cleared)) + " 해제")
    return "적용 조건: " + " · ".join(items) if items else ""


def _select_runs(
    ctx: _Context,
    records: list[dict],
    metric_column: str | None,
) -> tuple[list[dict], int, list[str], list[str]]:
    """필터·정렬·개수를 적용한다. 반환: (선택 run, 조건에 맞는 run 수, 조건 설명, 참고 사항)."""
    req = ctx.req
    filtered = _apply_search_runs_filters(records, req["filters"])
    notes: list[str] = []
    extra: list[str] = []

    if req["sort_by"] == "metric" or req["pick"] == "both_ends":
        scored = [r for r in filtered if _finite(r.get(metric_column)) is not None]
        missing = len(filtered) - len(scored)
        if missing:
            notes.append(f"`{_metric_name(metric_column)}` 값이 없는 run {missing}개는 순위에서 제외했습니다.")
        descending = req["order"] != "asc"
        scored.sort(key=lambda r: _finite(r.get(metric_column)) or 0.0, reverse=descending)
        ordered = scored
        direction = "높은 순" if descending else "낮은 순"
        extra.append(f"정렬 `{_metric_name(metric_column)}` {direction}{_source_suffix(ctx.sources.get('metric'))}")
    else:
        descending = req["order"] != "asc"
        ordered = sorted(filtered, key=lambda r: _epoch(r.get("start_time")) or 0.0, reverse=descending)
        if req["sort_by"] == "time":
            extra.append("정렬 시작 시각 " + ("최신순" if descending else "오래된순"))

    matched = len(ordered)
    if req["pick"] == "both_ends":
        selected = [ordered[0], ordered[-1]] if len(ordered) >= 2 else ordered
        extra.append("최고·최저 run")
    elif req["limit"]:
        selected = ordered[: req["limit"]]
        extra.append(f"{req['limit']}개")
    else:
        selected = ordered
    return selected, matched, extra, notes


@_route_step
def _answer_ranked_runs(
    ctx: _Context,
    records: list[dict] | None = None,
    scope_text: str | None = None,
    notes: list[str] | None = None,
) -> tuple[str, dict]:
    """조건(필터·정렬·개수)으로 run을 고른 뒤 요청한 작업을 한다.

    records가 없으면 확정된 실험의 run 전체에서, 있으면 그 run들(예: 직전 결과) 안에서 고른다.
    """
    req = ctx.req
    notes = list(notes or [])
    if records is None:
        experiment = ctx.resolved["experiment"]
        records = get_runs(experiment)
        for record in records:
            record.setdefault("experiment_name", experiment)
        if not records:
            return f"`{experiment}`에 run이 없습니다.", ctx.metadata(count=0)

    columns = sorted({k for r in records for k in r if k.startswith("metrics.")})
    metric_column, error = _resolve_metric_for(ctx, columns)
    if error:
        return error

    picks_one_run = req["action"] in {"lookup", "summarize", "evaluate"}
    if req["action"] == "compare" and not req["limit"] and req["pick"] != "both_ends":
        req["limit"] = DEFAULT_COMPARE_RUNS
        ctx.defaults.append(f"비교 개수 {DEFAULT_COMPARE_RUNS}개")
    if picks_one_run and not req["limit"]:
        req["limit"] = 1

    selected, matched, extra, select_notes = _select_runs(ctx, records, metric_column)
    notes += select_notes
    if scope_text:
        extra.insert(0, scope_text)
    header = _conditions_line(ctx, extra)

    if not selected:
        return _join(header, "조건에 맞는 run이 없습니다.", notes), ctx.metadata(count=0)

    if req["action"] == "count":
        body = f"조건에 맞는 run은 **{matched}개**입니다."
        return _join(header, body, notes), ctx.metadata(count=matched)

    if req["action"] == "compare":
        return _answer_compare(ctx, selected[:MAX_COMPARE_RUNS], metric_column, header, notes)

    if picks_one_run:
        ctx.resolved["runs"] = [_run_name(r) for r in selected]
        return _answer_run_records(ctx, selected, metric_column, header, notes)

    return _answer_run_list(ctx, selected, matched, metric_column, header, notes)


@_route_step
def _answer_run_list(
    ctx: _Context,
    selected: list[dict],
    matched: int,
    metric_column: str | None,
    header: str,
    notes: list[str],
) -> tuple[str, dict]:
    req = ctx.req
    output = req["output"]
    ctx.resolved["runs"] = [_run_name(r) for r in selected][:MAX_RUN_NAMES]
    rows = [RunDisplayRow.from_mlflow_raw(r, metric_column) for r in selected[:MAX_RUN_ROWS]]
    meta_extra: dict[str, Any] = {
        "count": matched,
        "run_display_rows": [row.model_dump(mode="json", by_alias=True) for row in rows],
    }
    if req["limit"] and metric_column:
        meta_extra["top_k"] = req["limit"]

    count_text = f"{matched}개 중 {len(selected)}개" if len(selected) != matched else f"{matched}개"

    if output == "names":
        names = [_run_name(r) for r in selected[:MAX_RUN_NAMES]]
        body = f"run {count_text}\n" + "\n".join(f"- {n}" for n in names)
        if len(selected) > MAX_RUN_NAMES:
            body += f"\n(너무 길어 {MAX_RUN_NAMES}개만 표시)"
        return _join(header, body, notes), ctx.metadata(**meta_extra)

    if len(selected) == 1 and output in (None, "brief", "value"):
        run = selected[0]
        if metric_column:
            body = (
                f"`{_run_name(run)}` — `{_metric_name(metric_column)}` "
                f"{_fmt_metric_value(run.get(metric_column))} ({run.get('status', '-')}, {_start_text(run)})"
            )
        else:
            body = f"`{_run_name(run)}` ({run.get('status', '-')}, {_start_text(run)})"
        return _join(header, body, notes), ctx.metadata(**meta_extra)

    metric_label = _metric_name(metric_column) if metric_column else None
    columns = ["Run"] + ([metric_label] if metric_label else []) + ["Status", "Start"]
    lines = [f"run {count_text}", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for run in selected[:MAX_RUN_ROWS]:
        cells = [_md_cell(_run_name(run))]
        if metric_column:
            value = run.get(metric_column)
            cells.append(_fmt_metric_value(value) if _finite(value) is not None else "-")
        cells.extend([str(run.get("status", "-")), _start_text(run)])
        lines.append("| " + " | ".join(cells) + " |")
    if len(selected) > MAX_RUN_ROWS:
        lines.append(f"\n(너무 길어 {MAX_RUN_ROWS}개만 표시)")
    return _join(header, "\n".join(lines), notes), ctx.metadata(**meta_extra)


# ── 3-4. 이름으로 지목한 run ─────────────────────────────


def _load_named_runs(ctx: _Context) -> tuple[list[dict], list[str], list[str]]:
    """runs의 각 항목을 찾는다. 반환: (run별 대표 레코드, 찾지 못한 이름, 참고 사항)."""
    experiment = ctx.resolved.get("experiment")
    primaries: list[dict] = []
    missing: list[str] = []
    notes: list[str] = []
    for ref in ctx.req["runs"]:
        is_id = bool(_RUN_ID_RE.fullmatch(ref))
        records = find_runs(
            run_name=None if is_id else ref,
            run_id=ref if is_id else None,
            experiment_name=experiment,
        )
        if not records:
            missing.append(ref)
            continue
        ordered = sorted(records, key=lambda r: _epoch(r.get("start_time")) or 0.0, reverse=True)
        primary = ordered[0]
        if len(ordered) > 1:
            notes.append(f"`{ref}` 이름의 run이 {len(ordered)}개라 가장 최근 run을 사용했습니다.")
        if primary.get("matched_outside_experiment"):
            notes.append(
                f"`{ref}`는 `{primary['matched_outside_experiment']}`이 아니라 "
                f"`{primary.get('experiment_name')}`에서 찾았습니다."
            )
        primaries.append(primary)
    return primaries, missing, notes


@_route_step
def _answer_named_runs(ctx: _Context) -> tuple[str, dict]:
    req = ctx.req
    primaries, missing, notes = _load_named_runs(ctx)
    ctx.sources["runs"] = ctx.source_of("runs")
    if not primaries:
        labels = ", ".join(f"`{m}`" for m in missing)
        return f"{labels} run을 찾지 못했습니다.", ctx.metadata(count=0)
    if missing:
        notes.append("찾지 못한 run: " + ", ".join(f"`{m}`" for m in missing))
    if not ctx.resolved.get("experiment"):
        experiments = {r.get("experiment_name") for r in primaries}
        if len(experiments) == 1:
            ctx.resolved["experiment"] = next(iter(experiments))
            ctx.sources.setdefault("experiment", "run에서 확인")

    from_previous = ctx.sources["runs"] == "이전 대화"
    # '그 중에 ~가 제일 낮은 건?'처럼 지목한 run 여러 개 안에서 순위로 고르는 요청
    if len(primaries) > 1 and (req["sort_by"] or req["pick"]):
        scope = f"대상 직전 결과 run {len(primaries)}개" if from_previous else f"대상 run {len(primaries)}개"
        return _answer_ranked_runs(ctx, records=primaries, scope_text=scope, notes=notes)

    ctx.resolved["runs"] = [_run_name(r) for r in primaries]
    columns = sorted({k for r in primaries for k in r if k.startswith("metrics.")})
    metric_column = None
    if req["metric"]:
        metric_column, error = _resolve_metric_for(ctx, columns)
        if error:
            return error

    header = ""
    if from_previous:
        header = "대상 run: " + ", ".join(f"`{n}`" for n in ctx.resolved["runs"]) + "(이전 대화)"

    if req["action"] == "compare":
        if len(primaries) < 2:
            body = "비교하려면 run이 두 개 이상 필요합니다. 비교할 다른 run을 알려주세요."
            return _join(header, body, notes), ctx.metadata()
        return _answer_compare(ctx, primaries[:MAX_COMPARE_RUNS], metric_column, header, notes)
    return _answer_run_records(ctx, primaries, metric_column, header, notes)


@_route_step
def _answer_run_records(
    ctx: _Context,
    primaries: list[dict],
    metric_column: str | None,
    header: str,
    notes: list[str],
) -> tuple[str, dict]:
    """정해진 run(들)에 대해 값·항목·상세·요약·평가 중 요청한 범위로 답한다."""
    req = ctx.req
    run_meta = _run_detail_metadata(primaries[:1])
    run_meta.pop("experiment_name", None)

    wants_summary = req["action"] == "summarize" or (
        req["action"] == "lookup" and req["output"] == "brief" and not metric_column and not req["fields"]
    )
    if wants_summary:
        body = _llm_run_answer(ctx, primaries[0], RUN_SUMMARY_PROMPT, with_siblings=False)
        return _join(header, body, notes), ctx.metadata(**run_meta, grounded_on_run=True)

    if req["action"] == "evaluate":
        body = _llm_run_answer(ctx, primaries[0], RUN_EVALUATION_PROMPT, with_siblings=True)
        return _join(header, body, notes), ctx.metadata(**run_meta, grounded_on_run=True)

    if metric_column and not req["fields"]:
        lines = []
        for run in primaries:
            value = run.get(metric_column)
            if _finite(value) is None:
                lines.append(f"`{_run_name(run)}`에는 `{_metric_name(metric_column)}` 값이 없습니다.")
            else:
                lines.append(
                    f"`{_run_name(run)}`의 `{_metric_name(metric_column)}`는 {_fmt_metric_value(value)}입니다."
                )
        return _join(header, "\n".join(lines), notes), ctx.metadata(**run_meta)

    if req["fields"]:
        sections = [_run_fields_markdown(run, req["fields"]) for run in primaries]
        return _join(header, "\n\n".join(sections), notes), ctx.metadata(**run_meta)

    sections = [_format_run_detail_markdown([run]) for run in primaries]
    return _join(header, "\n\n---\n\n".join(sections), notes), ctx.metadata(**run_meta)


def _run_fields_markdown(run: dict, fields: list[str]) -> str:
    lines = [f"`{_run_name(run)}` ({run.get('experiment_name', '-')})"]
    if "id" in fields:
        lines.append(f"- Run ID: `{run.get('run_id', '-')}`")
    if "status" in fields:
        lines.append(f"- Status: {run.get('status', '-')}")
    if "time" in fields:
        end_date, end_time = split_run_start_time(run.get("end_time"))
        lines.append(f"- Start: {_start_text(run)} / End: {end_date} {end_time}".rstrip())
    if "params" in fields:
        params = sorted((k.removeprefix("params."), v) for k, v in run.items() if k.startswith("params."))
        lines.extend(["", "| Parameter | Value |", "|---|---|"])
        if params:
            lines.extend(f"| {_md_cell(k)} | {_md_cell(v)} |" for k, v in params)
        else:
            lines.append("| - | - |")
    if "metrics" in fields:
        metrics = _model_metric_columns(run)
        lines.extend(["", "| Metric | Value |", "|---|---|"])
        if metrics:
            lines.extend(f"| {_md_cell(_metric_name(c))} | {_fmt_metric_value(run[c])} |" for c in metrics)
        else:
            lines.append("| - | - |")
    return "\n".join(lines)


@_route_step
def _answer_compare(
    ctx: _Context,
    runs: list[dict],
    metric_column: str | None,
    header: str,
    notes: list[str],
) -> tuple[str, dict]:
    req = ctx.req
    names = [_run_name(r) for r in runs]
    ctx.resolved["runs"] = names
    labels = names
    if req["pick"] == "both_ends" and len(runs) == 2:
        if req["order"] != "asc":
            labels = [f"{names[0]} (최고)", f"{names[1]} (최저)"]
        else:
            labels = [f"{names[0]} (최저)", f"{names[1]} (최고)"]

    if metric_column:
        metric_rows = [metric_column]
        if req["output"] != "value":
            others = [c for c in _model_metric_columns(runs[0]) if c != metric_column]
            metric_rows += others[: MAX_COMPARE_METRICS - 1]
    else:
        common = set(_model_metric_columns(runs[0]))
        for run in runs[1:]:
            common &= set(_model_metric_columns(run))
        metric_rows = sorted(common)[:MAX_COMPARE_METRICS]

    lines = [
        "| 항목 | " + " | ".join(_md_cell(label) for label in labels) + " |",
        "|---|" + "---|" * len(runs),
    ]
    for column in metric_rows:
        values = [_fmt_metric_value(r.get(column)) if _finite(r.get(column)) is not None else "-" for r in runs]
        lines.append(f"| {_md_cell(_metric_name(column))} | " + " | ".join(values) + " |")

    if req["output"] != "value":
        param_keys = sorted({k for r in runs for k in r if k.startswith("params.")})
        differing = [k for k in param_keys if len({str(r.get(k, "-")) for r in runs}) > 1]
        for key in differing[:MAX_COMPARE_METRICS]:
            values = [_md_cell(r.get(key, "-")) for r in runs]
            lines.append(f"| param: {_md_cell(key.removeprefix('params.'))} | " + " | ".join(values) + " |")
        if param_keys and not differing:
            notes.append("비교한 run의 파라미터는 모두 같습니다.")

    summary = ""
    if metric_column and len(runs) == 2:
        a, b = _finite(runs[0].get(metric_column)), _finite(runs[1].get(metric_column))
        if a is not None and b is not None:
            diff = a - b
            higher = names[0] if diff > 0 else names[1]
            summary = (
                f"`{_metric_name(metric_column)}` 차이: `{names[0]}` − `{names[1]}` = {diff:+.6g}"
                + (f" (`{higher}`가 더 높음)" if diff != 0 else " (같음)")
            )
    body = "\n".join(lines) + (f"\n\n{summary}" if summary else "")
    run_rows = [RunDisplayRow.from_mlflow_raw(r, metric_column) for r in runs]
    return _join(header, body, notes), ctx.metadata(
        count=len(runs),
        run_display_rows=[row.model_dump(mode="json", by_alias=True) for row in run_rows],
    )


# ── 3-5. LLM 답변(요약·평가) ─────────────────────────────


def _lower_is_better(metric_name: str) -> bool:
    lowered = metric_name.lower()
    return any(token in lowered for token in _LOWER_IS_BETTER)


def _sibling_stats_markdown(run: dict, metric_column: str | None) -> str:
    """같은 실험 run 대비 위치를 표로 만든다(평가 근거)."""
    experiment = run.get("experiment_name")
    if not experiment:
        return ""
    siblings = get_runs(experiment)
    columns = [metric_column] if metric_column else _model_metric_columns(run)[:MAX_EVAL_METRICS]
    lines = [
        f"같은 실험 비교 기준: `{experiment}` run {len(siblings)}개",
        "",
        "| metric | 이 run | 순위 | 최고 | 중앙값 | 최저 | 좋은 방향 |",
        "|---|---|---|---|---|---|---|",
    ]
    for column in columns:
        mine = _finite(run.get(column))
        values = [v for v in (_finite(s.get(column)) for s in siblings) if v is not None]
        if mine is None or not values:
            continue
        lower_better = _lower_is_better(_metric_name(column))
        ranked = sorted(values, reverse=not lower_better)
        rank = 1 + sum(1 for v in values if (v < mine if lower_better else v > mine))
        lines.append(
            f"| {_metric_name(column)} | {_fmt_metric_value(mine)} | {len(values)}개 중 {rank}위 | "
            f"{_fmt_metric_value(ranked[0])} | {_fmt_metric_value(statistics.median(values))} | "
            f"{_fmt_metric_value(ranked[-1])} | {'낮을수록' if lower_better else '높을수록'} |"
        )
    return "\n".join(lines)


def _llm_run_answer(ctx: _Context, run: dict, system_prompt: str, *, with_siblings: bool) -> str:
    context = "run 상세:\n" + _format_run_detail_markdown([run])
    if with_siblings:
        stats = _sibling_stats_markdown(run, ctx.resolved.get("metric_column"))
        if stats:
            context += "\n\n" + stats
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": context},
        *cs.history_to_messages(ctx.history),
        {"role": "user", "content": ctx.message},
    ]
    record_tool_call("llm_answer", {"context": "run_detail_with_siblings" if with_siblings else "run_detail"})
    completion = cs.llm_client().chat.completions.create(
        model=settings.chat_model,
        temperature=settings.chat_temperature,
        messages=messages,
    )
    return cs.strip_think_tags(completion.choices[0].message.content or "")
