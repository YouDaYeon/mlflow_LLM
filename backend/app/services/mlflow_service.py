import mlflow
import pandas as pd

from typing import Any

from fastapi import HTTPException
from backend.app.config import settings
from backend.app.schemas import RunSummary

# MLflow 실험 설정
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

METRIC_PREFIX = "metrics."
PARAM_PREFIX = "params."


def get_tracking_uri():
    """MLflow tracking URI 조회"""
    tracking_uri = mlflow.get_tracking_uri()
    return tracking_uri

def list_experiment_names() -> list[str]:
    """MLflow에 등록된 Experiment 이름 목록(정렬)."""
    experiments = mlflow.search_experiments()
    names = [experiment.name for experiment in experiments]
    return sorted(set(names))


def _get_experiment_or_404(experiment_name: str):
    """실험 정보 조회 또는 404 예외 발생"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"실험을 찾을 수 없습니다: {experiment_name}")
    return experiment


def get_experiment(experiment_name: str) -> dict:
    experiment = _get_experiment_or_404(experiment_name)
    return {
        "experiment_name": experiment.name,
        "experiment_id": experiment.experiment_id,
        "artifact_location": experiment.artifact_location,
        "lifecycle_stage": experiment.lifecycle_stage,
        "tags": experiment.tags,
        "creation_time": experiment.creation_time,
        "last_update_time": experiment.last_update_time,
        "trace_location": experiment.trace_location,
        "workspace": experiment.workspace,
    }

def _search_runs_df(experiment_name: str) -> pd.DataFrame:
    experiment = _get_experiment_or_404(experiment_name)
    return mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
def get_runs(experiment_name: str) -> list[dict]:
    runs = _search_runs_df(experiment_name)
    return runs.to_dict(orient="records")

## tags 관련 함수
# [c for i, c in enumerate(runs_df.columns) if c.startswith("tags.")]
# ['tags.mlflow.user', 'tags.mlflow.source.type', 'tags.mlflow.source.name', 'tags.mlflow.runName']
def list_run_names(experiment_name: str) -> list[str]:
    runs_df = _search_runs_df(experiment_name)
    return runs_df["tags.mlflow.runName"].unique().tolist()

def list_run_users(experiment_name: str) -> list[str]:
    runs_df = _search_runs_df(experiment_name)
    return runs_df["tags.mlflow.user"].unique().tolist()

## metric 관련 함수
def list_run_metrics(experiment_name: str) -> list[str]:
    runs_df = _search_runs_df(experiment_name)
    return sorted(runs_df.columns[runs_df.columns.str.startswith("metrics.")].tolist())


def _scored_runs_df(runs_df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    scored_df = runs_df[pd.to_numeric(runs_df[metric_column], errors="coerce").notna()].copy()
    if scored_df.empty:
        return scored_df
    scored_df[metric_column] = pd.to_numeric(scored_df[metric_column], errors="coerce")
    return scored_df

def _to_iso_start_time(value: object) -> str | None:
    if pd.isna(value):
        return None
    dt_value = pd.to_datetime(value, unit="ms", errors="coerce")
    if pd.isna(dt_value):
        return None
    return dt_value.isoformat()


def _row_to_run_summary(row: pd.Series, metric_column: str) -> RunSummary:
    params = {
        key.removeprefix(PARAM_PREFIX): str(value)
        for key, value in row.items()
        if key.startswith(PARAM_PREFIX) and pd.notna(value)
    }
    metrics = {
        key.removeprefix(METRIC_PREFIX): float(value)
        for key, value in row.items()
        if key.startswith(METRIC_PREFIX) and pd.notna(value)
    }
    return RunSummary(
        run_id=str(row.get("run_id", "")),
        run_name=str(row.get("tags.mlflow.runName", "")),
        status=str(row.get("status", "")),
        metric_score=float(row[metric_column]),
        metric_key=metric_column,
        start_time=_to_iso_start_time(row.get("start_time")),
        params=params,
        metrics=metrics,
    )

def get_top_runs(
    experiment_name: str,
    metric_key: str,
    top_k: int,
    ascending: bool = False,
) -> list[RunSummary]:
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k는 1 이상의 정수여야 합니다.")

    runs_df = _search_runs_df(experiment_name)
    metric_column = metric_key
    if metric_column not in runs_df.columns:
        raise HTTPException(status_code=404, detail=f"metric_key를 찾을 수 없습니다: {metric_key}")

    scored_df = _scored_runs_df(runs_df, metric_column)
    if scored_df.empty:
        return []

    top_df = scored_df.sort_values(by=metric_column, ascending=ascending).head(top_k)
    return [_row_to_run_summary(row, metric_column) for _, row in top_df.iterrows()]


def list_recent_experiment_names(limit: int = 5) -> list[str]:
    experiments = mlflow.search_experiments()
    sorted_experiments = sorted(experiments, key=lambda exp: exp.creation_time or 0, reverse=True)
    return [exp.name for exp in sorted_experiments[:limit]]


#### test
def test():
    experiments = mlflow.search_experiments()

    # if runs is None or runs.empty:
    #     return {"message": "No runs"}

    return experiments


def mlflow_mcp_stdio_params() -> dict[str, Any]:
    """agents.mcp.MCPServerStdio용 stdio 파라미터(uv가 mlflow[mcp] 의존성을 함께 받아온다)."""
    return {
        "command": "uv",
        "args": ["run", "--with", "mlflow[mcp]>=3.5.1", "mlflow", "mcp", "run"],
        "env": {"MLFLOW_TRACKING_URI": settings.mlflow_tracking_uri},
    }


async def mlflow_mcp_health() -> tuple[bool, list[str], str | None]:
    """MCP 세션 초기화까지 성공하면 True와 노출된 도구 이름 목록을 반환한다."""
    from agents.mcp import MCPServerManager, MCPServerStdio

    server = MCPServerStdio(
        params=mlflow_mcp_stdio_params(),
        name="mlflow-mcp",
        cache_tools_list=True,
    )
    try:
        async with MCPServerManager([server], strict=False) as manager:
            if not manager.active_servers:
                err = next(iter(manager.errors.values()), None)
                return False, [], repr(err) if err else "MCP 연결 실패(원인 없음)"
            mcp = manager.active_servers[0]
            tools = await mcp.list_tools(None, None)
            names = sorted({t.name for t in tools if t.name})
            return True, names, None
    except Exception as exc:  # noqa: BLE001  # 헬스 체크용
        return False, [], f"{type(exc).__name__}: {exc}"