"""평가 실행 중 생긴 빈 채팅 세션(메시지 0개)을 실행이 끝나면 지운다.

/chat/ask는 답변을 만들기 전에 세션을 먼저 저장하므로, 요청이 실패하면 메시지 없는 세션이 남고
응답에 session_id가 없어 스크립트가 지울 수도 없다. 실행 시작 이후 만들어진 빈 세션만 지운다.

backend 패키지를 불러오므로 --code-root를 sys.path에 넣은 뒤에 사용한다.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import text


@contextmanager
def cleanup_empty_sessions() -> Iterator[None]:
    from backend.app.db.session import engine

    # 세션 created_at은 DB now()로 찍히므로 시작 시각도 DB 시계로 잡는다.
    with engine.connect() as conn:
        started_at = conn.execute(text("SELECT now()")).scalar_one()
    try:
        yield
    finally:
        with engine.begin() as conn:
            deleted = conn.execute(
                text(
                    "DELETE FROM chat_sessions s "
                    "WHERE s.created_at >= :started_at "
                    "AND NOT EXISTS (SELECT 1 FROM chat_messages m WHERE m.session_id = s.id)"
                ),
                {"started_at": started_at},
            ).rowcount
        if deleted:
            print(f"빈 채팅 세션 {deleted}개 삭제", flush=True)
