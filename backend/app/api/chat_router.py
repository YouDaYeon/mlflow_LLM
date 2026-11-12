from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.db.session import get_db
from backend.app.schemas import (
    ChatMessage,
    ChatMessageItem,
    ChatMessageListResponse,
    ChatRequest,
    ChatResponse,
    ChatSessionCreateResponse,
    ChatSessionListItem,
    ChatSessionListResponse,
)
from backend.app.services.repository_service import (
    append_chat_message,
    create_chat_session,
    delete_chat_session,
    get_chat_session,
    get_last_assistant_context,
    get_recent_chat_history,
    list_chat_messages,
    list_chat_sessions,
    update_chat_session_title,
)

router = APIRouter(prefix="/chat", tags=["chat"])


def _to_session_item(row) -> ChatSessionListItem:
    return ChatSessionListItem(
        session_id=row.id,
        title=row.title,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.post("/sessions", response_model=ChatSessionCreateResponse)
def create_session(db: Session = Depends(get_db)) -> ChatSessionCreateResponse:
    row = create_chat_session(db)
    return ChatSessionCreateResponse(
        session_id=row.id,
        title=row.title,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("/sessions", response_model=ChatSessionListResponse)
def get_sessions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> ChatSessionListResponse:
    rows = list_chat_sessions(db, limit=limit, offset=offset)
    return ChatSessionListResponse(items=[_to_session_item(row) for row in rows])


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_session(
    session_id: UUID,
    db: Session = Depends(get_db),
) -> Response:
    deleted = delete_chat_session(db, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="chat session not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/sessions/{session_id}/messages", response_model=ChatMessageListResponse)
def get_session_messages(
    session_id: UUID,
    db: Session = Depends(get_db),
) -> ChatMessageListResponse:
    session = get_chat_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    rows = list_chat_messages(db, session_id=session_id)
    return ChatMessageListResponse(
        session_id=session_id,
        items=[
            ChatMessageItem(
                id=row.id,
                role=row.role,
                content=row.content,
                metadata=row.message_metadata or {},
                created_at=row.created_at,
            )
            for row in rows
        ],
    )


@router.post("/ask", response_model=ChatResponse)
async def chat_with_experiment(
    payload: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    # 다른 라우트 기동에 영향 없도록 chat 의존성은 실제 호출 시 로드한다.
    from backend.app.services.chat_request import PARSER_NAME, answer_request, parse_request
    from backend.app.services.chat_service import generate_session_title, run_with_mcp
    from backend.app.services.mlflow_service import start_tool_call_recording

    if payload.session_id is None:
        session = create_chat_session(db)
    else:
        session = get_chat_session(db, payload.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="chat session not found")

    session_id = session.id
    prior_messages = list_chat_messages(db, session_id=session_id)
    is_first_turn = len(prior_messages) == 0

    db_history = get_recent_chat_history(
        db,
        session_id=session_id,
        n=settings.chat_history_limit,
    )
    if db_history:
        history = [ChatMessage(role=m["role"], content=m["content"]) for m in db_history]
    else:
        history = list(payload.history or [])

    # 직전 턴의 요청 구조와 답변 대상을 함께 넘겨, 파서가 조건의 유지·변경·해제를 판단하게 한다.
    previous = get_last_assistant_context(db, session_id=session_id)
    request = parse_request(payload.message, history=history, previous=previous)

    # 이 요청에서 호출된 MLflow 조회/LLM 답변을 tool_calls로 남긴다(MCP 경로는 Agent 기록을 그대로 쓴다).
    recorded_tool_calls = start_tool_call_recording()
    result = answer_request(
        request,
        message=payload.message,
        history=history,
        report_text=payload.report_text,
        ui_experiment=payload.experiment_name,
        ui_metric=payload.metric_key,
        previous=previous,
    )
    if result is None:
        # 요청 구조로 표현할 수 없는 질문(trace 등)은 MLflow MCP 에이전트가 처리한다.
        answer, metadata = await run_with_mcp(message=payload.message, history=history)
        metadata = {
            **metadata,
            # answer_request가 요청 구조로 답할 수 없다고 넘긴(None) 경우다.
            "route": {
                "parser": PARSER_NAME,
                "path": ["answer_request", run_with_mcp.__name__],
                "handler": run_with_mcp.__name__,
            },
            "request": request,
        }
    else:
        answer, metadata = result

    if not isinstance(metadata, dict):
        metadata = {}
    if not metadata.get("tool_calls"):
        metadata["tool_calls"] = list(recorded_tool_calls) or ["No tool calls"]

    append_chat_message(
        db,
        session_id=session_id,
        role="user",
        content=payload.message,
    )
    append_chat_message(
        db,
        session_id=session_id,
        role="assistant",
        content=answer,
        metadata=metadata,
    )

    if is_first_turn:
        title = generate_session_title(payload.message, answer)
        update_chat_session_title(db, session_id=session_id, title=title)

    return ChatResponse(
        answer=answer,
        metadata=metadata,
        session_id=session_id,
    )
