from fastapi import APIRouter

from backend.app.config import settings
from backend.app.schemas import McpHealthResponse
from backend.app.services.mlflow_service import (
    get_tracking_uri as get_tracking_uri_service,
    list_experiment_names,
    get_experiment as get_experiment_service,
    list_run_metrics,
    get_runs as get_runs_service,
    get_top_runs as get_top_runs_service,

    test as test_service,
)

router = APIRouter(prefix="/mlflow", tags=["mlflow"])

@router.get("/mcp-health", response_model=McpHealthResponse)
async def mlflow_mcp_health_check() -> McpHealthResponse:
    """MLflow MCP stdio 서버 기동 및 initialize/list_tools까지 성공하는지 확인한다."""
    from backend.app.services.mlflow_service import mlflow_mcp_health

    ok, tool_names, err = await mlflow_mcp_health()
    return McpHealthResponse(
        ok=ok,
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
        tool_names=tool_names,
        error=err,
    )

# mlflow 설정 확인용
@router.get("/tracking-uri")
def get_tracking_uri():
    return get_tracking_uri_service()

@router.get("/names")
def get_experiment_names():
    return {"experiment_names": list_experiment_names()}

@router.get("/experiment")
def get_experiment(experiment_name: str):
    return {"experiment": get_experiment_service(experiment_name)}

@router.get("/runs")
def get_runs(experiment_name: str):
    return {"runs": get_runs_service(experiment_name)}

@router.get("/top-runs")
def get_top_runs(experiment_name: str, metric_key: str, top_k: int, ascending: bool = False):
    return {"top_runs": get_top_runs_service(experiment_name, metric_key, top_k, ascending)}

@router.get("/metric-keys")
def get_metric_keys(experiment_name: str):
    return {"metric_keys": list_run_metrics(experiment_name)}

@router.get("/test")
def get_test():
    return test_service()