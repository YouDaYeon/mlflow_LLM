import re
from typing import Any
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


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _resolve_experiment_args(
    *,
    intent_args: dict[str, Any] | None,
    payload: ChatRequest,
    last_meta: dict[str, Any],
) -> tuple[str | None, str | None, int | None, bool]:
    """intent args → payload → 이전 assistant metadata 순으로 빈 값을 채운다."""
    args = intent_args or {}
    experiment_name = _first_present(
        args.get("experiment_name"),
        payload.experiment_name,
        last_meta.get("experiment_name"),
    )
    metric_key = _first_present(
        args.get("metric_key"),
        payload.metric_key,
        last_meta.get("metric_key"),
    )
    top_k = _first_present(args.get("top_k"), last_meta.get("top_k"))
    if isinstance(top_k, str) and top_k.isdigit():
        top_k = int(top_k)
    elif top_k is not None and not isinstance(top_k, int):
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = None

    ascending_raw = _first_present(args.get("ascending"), last_meta.get("ascending"))
    if ascending_raw is None:
        ascending = False
    else:
        ascending = bool(ascending_raw)

    return experiment_name, metric_key, top_k, ascending


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
                intent=row.intent,
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
    from backend.app.services.chat_service import (
        answer_experiment_question,
        answer_report_question,
        answer_run_question,
        classify_intent,
        extract_single_run_reference,
        generate_session_title,
        is_run_judgment_question,
        run_with_mcp,
    )

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

    intent, intent_args = classify_intent(payload.message, history=history)
    last_meta = get_last_assistant_context(db, session_id=session_id)
    experiment_name, metric_key, top_k, ascending = _resolve_experiment_args(
        intent_args=intent_args,
        payload=payload,
        last_meta=last_meta,
    )
    guessed_run_id, guessed_run_name = extract_single_run_reference(payload.message)
    run_name = _first_present((intent_args or {}).get("run_name"), guessed_run_name)
    run_id = _first_present((intent_args or {}).get("run_id"), guessed_run_id)
    if isinstance(run_name, str):
        run_name = run_name.strip() or None
    if isinstance(run_id, str):
        run_id = run_id.strip() or None
    # 이전 턴에서 이어받은 metric_key로는 run 전체를 좁히지 않는다.
    asked_metric_key = (intent_args or {}).get("metric_key")
    if isinstance(asked_metric_key, str):
        asked_metric_key = asked_metric_key.strip() or None
    else:
        asked_metric_key = None

    report_text = payload.report_text

    experiment_intents = [
        "list_experiments",
        "list_recent_experiments",
        "get_experiment",
        "get_runs",
        "get_run_info",
        "list_metric_keys",
        "filter_runs_by_metric",
        "search_runs",
    ]

    is_comparison = bool(
        re.search(r"비교|차이|versus|\bvs\b", payload.message or "", flags=re.IGNORECASE)
    )
    # 평가 질문, 또는 run이 있는데 실험 조회 intent가 아닌 질문은 상세 표 대신 모델이 답한다.
    route_run_qa = bool(run_name or run_id) and not is_comparison and intent != "report_qa" and (
        is_run_judgment_question(payload.message)
        or intent not in experiment_intents
    )
    if route_run_qa:
        answer, intent, metadata = answer_run_question(
            payload.message,
            experiment_name=experiment_name,
            run_name=run_name if isinstance(run_name, str) else None,
            run_id=run_id if isinstance(run_id, str) else None,
            history=history,
        )
    elif intent in experiment_intents:
        answer, intent, metadata = answer_experiment_question(
            intent=intent,
            message=payload.message,
            experiment_name=experiment_name,
            metric_key=metric_key,
            top_k=top_k,
            ascending=ascending,
            run_name=run_name if isinstance(run_name, str) else None,
            run_id=run_id if isinstance(run_id, str) else None,
            asked_metric_key=asked_metric_key,
        )
    elif intent == "report_qa":
        answer, intent, metadata = answer_report_question(
            intent=intent,
            message=payload.message,
            report_text=report_text,
            experiment_name=experiment_name,
            metric_key=metric_key,
            history=history,
        )
    else:
        answer, intent, metadata = await run_with_mcp(
            message=payload.message,
            history=history,
        )

    if not isinstance(metadata, dict):
        metadata = {}
    # 후속 턴 복원용으로 해석된 컨텍스트를 metadata에 남긴다.
    if experiment_name and "experiment_name" not in metadata:
        metadata["experiment_name"] = experiment_name
    if metric_key and "metric_key" not in metadata:
        metadata["metric_key"] = metric_key
    if top_k is not None and "top_k" not in metadata:
        metadata["top_k"] = top_k
    metadata["ascending"] = ascending

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
        intent=intent,
        metadata=metadata,
    )

    if is_first_turn:
        title = generate_session_title(payload.message, answer)
        update_chat_session_title(db, session_id=session_id, title=title)

    return ChatResponse(
        answer=answer,
        intent=intent,
        metadata=metadata,
        session_id=session_id,
    )
