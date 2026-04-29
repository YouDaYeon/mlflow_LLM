from fastapi import FastAPI
from backend.app.api.mlflow_router import router as mlflow_router
from backend.app.api.report_router import router as report_router
from backend.app.api.chat_router import router as chat_router

app = FastAPI()
app.include_router(router=mlflow_router)
app.include_router(router=report_router)
app.include_router(router=chat_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}