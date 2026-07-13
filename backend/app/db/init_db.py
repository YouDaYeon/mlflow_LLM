from sqlalchemy import text

from backend.app.db.base import Base
from backend.app.db.session import engine
# create_all이 새 테이블을 인식하도록 모델 import
from backend.app.db.models import ChatMessageRow, ChatSession, Report  # noqa: F401


def _ensure_reports_updated_at() -> None:
    """기존 DB에 updated_at 컬럼이 없으면 추가한다."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                ALTER TABLE reports
                ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                """
            )
        )
        conn.execute(
            text(
                """
                UPDATE reports
                SET updated_at = created_at
                WHERE updated_at IS NULL OR updated_at < created_at
                """
            )
        )


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_reports_updated_at()
