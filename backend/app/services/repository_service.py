from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import ChatMessageRow, ChatSession, Report
from backend.app.schemas import RunSummary

_SESSION_TITLE_MAX = 80


def save_report(
    db: Session,
    *,
    experiment_name: str,
    metric_key: str,
    report_title: str,
    report: str,
    model: str,
    top_runs: list[RunSummary] | list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> Report:
    top_runs_payload = [
        run.model_dump() if isinstance(run, RunSummary) else run for run in top_runs
    ]
    row = Report(
        experiment_name=experiment_name,
        metric_key=metric_key,
        report_title=report_title,
        report=report,
        model=model,
        top_runs=top_runs_payload,
        report_metadata=metadata or {},
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_report_list(
    db: Session,
    *,
    experiment_name: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[Report]:
    query = select(Report).order_by(Report.created_at.desc())
    if experiment_name:
        query = query.where(Report.experiment_name == experiment_name)
    query = query.limit(limit).offset(offset)
    return list(db.scalars(query).all())


def get_report_by_id(db: Session, report_id: UUID) -> Report | None:
    return db.get(Report, report_id)


def update_report(
    db: Session,
    report_id: UUID,
    *,
    report: str,
    report_title: str | None = None,
) -> Report | None:
    row = get_report_by_id(db, report_id)
    if row is None:
        return None
    row.report = report
    if report_title is not None and report_title.strip():
        row.report_title = report_title.strip()
    row.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return row


def delete_report(db: Session, report_id: UUID) -> bool:
    row = get_report_by_id(db, report_id)
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True


def create_chat_session(db: Session, *, title: str | None = None) -> ChatSession:
    row = ChatSession(title=title)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_chat_session(db: Session, session_id: UUID) -> ChatSession | None:
    return db.get(ChatSession, session_id)


def list_chat_sessions(
    db: Session,
    *,
    limit: int = 20,
    offset: int = 0,
) -> list[ChatSession]:
    query = (
        select(ChatSession)
        .order_by(ChatSession.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.scalars(query).all())


def append_chat_message(
    db: Session,
    *,
    session_id: UUID,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> ChatMessageRow:
    session = get_chat_session(db, session_id)
    if session is None:
        raise ValueError(f"chat session not found: {session_id}")

    row = ChatMessageRow(
        session_id=session_id,
        role=role,
        content=content,
        message_metadata=metadata or {},
    )
    db.add(row)

    now = datetime.now(timezone.utc)
    session.updated_at = now
    if role == "user" and not session.title:
        truncated = content.strip().replace("\n", " ")
        if len(truncated) > _SESSION_TITLE_MAX:
            truncated = truncated[:_SESSION_TITLE_MAX].rstrip() + "…"
        session.title = truncated or "새 대화"

    db.commit()
    db.refresh(row)
    return row


def list_chat_messages(
    db: Session,
    *,
    session_id: UUID,
    limit: int | None = None,
) -> list[ChatMessageRow]:
    query = (
        select(ChatMessageRow)
        .where(ChatMessageRow.session_id == session_id)
        .order_by(ChatMessageRow.created_at.asc(), ChatMessageRow.id.asc())
    )
    if limit is not None and limit > 0:
        query = query.limit(limit)
    return list(db.scalars(query).all())


def get_recent_chat_history(
    db: Session,
    *,
    session_id: UUID,
    n: int,
) -> list[dict[str, str]]:
    """LLM용 최근 N개 메시지 ({role, content})."""
    if n <= 0:
        return []
    rows = list_chat_messages(db, session_id=session_id)
    recent = rows[-n:]
    return [{"role": row.role, "content": row.content} for row in recent]


def get_last_assistant_context(
    db: Session,
    *,
    session_id: UUID,
) -> dict[str, Any]:
    """최근 assistant 메시지의 metadata에서 experiment/metric/top_k 등 컨텍스트를 복원한다."""
    rows = list_chat_messages(db, session_id=session_id)
    for row in reversed(rows):
        if row.role != "assistant":
            continue
        meta = row.message_metadata or {}
        if not isinstance(meta, dict):
            continue
        ctx: dict[str, Any] = {}
        if meta.get("experiment_name"):
            ctx["experiment_name"] = meta["experiment_name"]
        if meta.get("metric_key"):
            ctx["metric_key"] = meta["metric_key"]
        if meta.get("top_k") is not None:
            ctx["top_k"] = meta["top_k"]
        if meta.get("ascending") is not None:
            ctx["ascending"] = meta["ascending"]
        # 요청 해석기가 남긴 직전 요청 구조와 답변 대상(다음 턴의 유지·변경·해제 판단용)
        if isinstance(meta.get("request"), dict):
            ctx["request"] = meta["request"]
        if isinstance(meta.get("resolved"), dict):
            ctx["resolved"] = meta["resolved"]
        if ctx:
            return ctx
    return {}


def update_chat_session_title(
    db: Session,
    *,
    session_id: UUID,
    title: str,
) -> ChatSession | None:
    session = get_chat_session(db, session_id)
    if session is None:
        return None
    cleaned = title.strip().replace("\n", " ")
    if len(cleaned) > _SESSION_TITLE_MAX:
        cleaned = cleaned[:_SESSION_TITLE_MAX].rstrip() + "…"
    session.title = cleaned or session.title
    session.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)
    return session


def delete_chat_session(db: Session, session_id: UUID) -> bool:
    """세션 hard delete (messages는 FK CASCADE)."""
    session = get_chat_session(db, session_id)
    if session is None:
        return False
    db.delete(session)
    db.commit()
    return True

