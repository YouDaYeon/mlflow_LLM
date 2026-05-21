from backend.app.db.base import Base
from backend.app.db.session import engine

# 앱 시작 시 테이블이 없으면 자동 생성
def init_db() -> None:
    Base.metadata.create_all(bind=engine)