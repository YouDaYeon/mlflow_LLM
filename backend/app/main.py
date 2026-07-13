from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.chat_router import router as chat_router
from backend.app.api.mlflow_router import router as mlflow_router
from backend.app.api.report_router import router as report_router
from backend.app.db.init_db import init_db


# uvicorn 기동 → reports / chat_* 테이블 자동 생성 (없을 때만)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션이 시작되기 전 실행
    init_db()
    yield
    # 애플리케이션이 종료되기 전 실행


app = FastAPI(lifespan=lifespan)
app.include_router(router=mlflow_router)
app.include_router(router=report_router)
app.include_router(router=chat_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}