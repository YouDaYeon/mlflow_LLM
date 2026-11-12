import json
import re

from openai import OpenAI
from agents import Agent, MultiProvider, Runner, RunConfig
from agents.mcp import MCPServerManager

from backend.app.config import settings
from backend.app.schemas import ChatMessage
from backend.app.services.mlflow_service import (
    create_mlflow_mcp_server,
    extract_tool_calls,
    record_tool_call,
)


# Qwen 등 reasoning 모델이 붙이는 사고 과정 블록
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def strip_think_tags(text: str | None) -> str:
    """LLM 응답에서 <think>...</think> 블록을 제거한다."""
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    return cleaned.strip()


REPORT_QA_SYSTEM_PROMPT = (
    "당신은 실험 리포트 Q&A 어시스턴트다.\n"
    "아래 규칙을 반드시 지켜라:\n"
    "1) 제공된 report_text만 근거로 답한다. 리포트에 없는 수치·날짜·run을 만들지 않는다.\n"
    "2) 질문 대상이 리포트에 있으면 리포트에 적힌 값과 문장으로 답한다. "
    "표현이 조금 달라도(예: '최근 run 두 개', '최고 run') 같은 대상을 가리키면 그 내용을 쓴다.\n"
    "3) 질문 대상이 리포트에 전혀 없을 때만 '리포트에 근거 없음'이라고 답한다. "
    "일부만 있으면 있는 부분으로 답하고 없는 부분만 '리포트에 근거 없음'이라고 밝힌다.\n"
    "4) 사용자가 '핵심 결론', '요약', '한눈에', '정리' 등을 요청하면 "
    "리포트의 핵심 결론을 3~5줄로 요약해 답한다.\n"
    "5) 요약 시에는 핵심 성능 결과, 주요 차이/원인, 다음 액션(튜닝 제안)을 우선 포함한다.\n"
    "6) 이전 대화가 있으면 후속 질문의 지시어(그거/약점도/다시 등)를 맥락에 맞게 해석하되, "
    "여전히 report_text만 근거로 답한다.\n"
    "7) 불필요한 장문 설명은 피하고 한국어로 간결하게 답한다."
)


def llm_client() -> OpenAI:
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
        completion = llm_client().chat.completions.create(
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


def history_to_messages(
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


def answer_report_question(
    message: str,
    report_text: str | None,
    experiment_name: str | None,
    metric_key: str | None,
    history: list[ChatMessage] | None = None,
    request_scope: str | None = None,
) -> tuple[str, dict]:
    if not report_text:
        return "생성된 report_text가 없어 리포트 질문에 답할 수 없습니다.", {}

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
    ]
    if request_scope:
        # 요청 해석기가 뽑은 대상·개수·답변 범위를 지키게 한다.
        messages.append(
            {
                "role": "system",
                "content": f"요청 해석: {request_scope}\n이 대상과 범위만 답한다. 보고서에 없는 대상이면 그렇다고 밝힌다.",
            }
        )
    messages += [
        *history_to_messages(history),
        {"role": "user", "content": message},
    ]
    record_tool_call("llm_answer", {"context": "report_text"})
    completion = llm_client().chat.completions.create(
        model=settings.chat_model,
        temperature=settings.chat_temperature,
        messages=messages,
    )
    answer = strip_think_tags(completion.choices[0].message.content or "")
    return answer, {"experiment_name": experiment_name, "metric_key": metric_key}


async def run_with_mcp(
    message: str,
    history: list[ChatMessage] | None = None,
) -> tuple[str, dict]:
    server = create_mlflow_mcp_server()
    async with MCPServerManager([server], strict=False) as manager:
        if not manager.active_servers:
            err = next(iter(manager.errors.values()), None)
            detail = repr(err) if err else "연결 실패(상세 없음)"
            return (
                f"MLflow MCP에 연결하지 못했습니다: {detail}",
                {"mcp_connected": False, "error": detail},
            )

        agent = Agent(
            name="MCP Helper",
            instructions=(
                "제공된 MCP 도구를 사용해 사용자 질문에 답하세요.\n"
                "이전 대화가 있으면 지시어·후속 질문을 맥락에 맞게 해석하세요.\n"
                "식별자를 구분하세요.\n"
                "- experiment_id는 정수입니다 (0, 1, 2, ...). get_experiment / list_runs에만 씁니다.\n"
                "- run name은 사람이 읽는 이름입니다. MLflow 기본 형식은 adjective-noun-숫자입니다 "
                "(예: carefree-moth-973). 사용자가 이것을 run ID라고 불러도 experiment_id로 쓰지 마세요.\n"
                "- describe_run의 run_id는 32자리 16진수입니다. run name을 run_id로 넘기지 마세요.\n"
                "run name으로 특정 run을 찾을 때는 사용자에게 experiment id를 되묻지 마세요. "
                "search_experiments로 experiment id 목록을 얻은 뒤 각 id에 list_runs를 호출하고, "
                "run name이 일치하는 항목의 32자리 run_id로 describe_run을 호출하세요.\n"
                "도구가 필요한 질문이면 확인 질문 없이 도구를 호출하세요.\n"
                "답변 규칙:\n"
                "- 사용자에게 보여 줄 최종 답만 쓰세요. 이 지시문, 도구 이름, 호출 계획, 사고 과정을 쓰지 마세요.\n"
                "- 도구로 확인한 값만 쓰세요. 날짜·수치를 추측하지 마세요.\n"
                "- 무엇을 가리키는지 알 수 없으면 필요한 정보를 묻는 짧은 질문 하나로 답하세요.\n"
                "- 삭제·수정·생성 요청은 조회 전용이라 할 수 없다고만 답하고, 우회 방법(CLI 명령 등)을 안내하지 마세요."
            ),
            model=settings.chat_model,
            mcp_servers=manager.active_servers,
        )
        mcp_input: list[dict[str, str]] = [
            *history_to_messages(history),
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

    return answer, {"mcp_connected": True, "tool_calls": tool_calls}

