import json
import re
import sys

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


_RUN_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def _mlflow_filter_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _run_lookup_filter(*, run_name: str | None, run_id: str | None) -> str | None:
    """run 이름 또는 32자리 run id로 MLflow search filter를 만든다."""
    clauses: list[str] = []
    raw_id = (run_id or "").strip()
    raw_name = (run_name or "").strip()
    if raw_id and _RUN_ID_RE.fullmatch(raw_id):
        clauses.append(f"attributes.run_id = {_mlflow_filter_quote(raw_id)}")
    elif raw_id and not raw_name:
        # 사용자가 run 이름을 run ID라고 부른 경우
        raw_name = raw_id
    if raw_name:
        clauses.append(f"attributes.run_name = {_mlflow_filter_quote(raw_name)}")
    if not clauses:
        return None
    return " OR ".join(clauses)


def _frame_to_run_records(frame: pd.DataFrame, id_to_name: dict[str, str]) -> list[dict]:
    if frame is None or frame.empty:
        return []
    records: list[dict] = []
    for _, row in frame.iterrows():
        item: dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                continue
            if isinstance(value, pd.Timestamp):
                item[str(key)] = value.isoformat()
            elif hasattr(value, "item") and not isinstance(value, (str, bytes)):
                try:
                    item[str(key)] = value.item()
                except (ValueError, AttributeError):
                    item[str(key)] = value
            else:
                item[str(key)] = value
        exp_id = str(item.get("experiment_id", ""))
        item["experiment_name"] = id_to_name.get(exp_id, exp_id)
        records.append(item)
    return records


def find_runs(
    *,
    run_name: str | None = None,
    run_id: str | None = None,
    experiment_name: str | None = None,
) -> list[dict]:
    """run 이름 또는 run id로 run을 찾는다. experiment_name이 있으면 그 실험을 먼저 본다."""
    filter_string = _run_lookup_filter(run_name=run_name, run_id=run_id)
    if not filter_string:
        return []

    experiments = mlflow.search_experiments()
    id_to_name = {exp.experiment_id: exp.name for exp in experiments}
    if not id_to_name:
        return []

    scoped_ids: list[str] | None = None
    if experiment_name:
        scoped_ids = [exp_id for exp_id, name in id_to_name.items() if name == experiment_name]

    search_ids = scoped_ids or list(id_to_name)
    frame = mlflow.search_runs(experiment_ids=search_ids, filter_string=filter_string)
    records = _frame_to_run_records(frame, id_to_name)
    if records or not scoped_ids:
        return records

    frame = mlflow.search_runs(experiment_ids=list(id_to_name), filter_string=filter_string)
    records = _frame_to_run_records(frame, id_to_name)
    for record in records:
        record["matched_outside_experiment"] = experiment_name
    return records

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
    """agents.mcp.MCPServerStdio용 stdio 파라미터(현재 Python + mlflow[mcp] 필요)."""
    return {
        "command": sys.executable,
        "args": ["-m", "mlflow", "mcp", "run"],
        "env": {"MLFLOW_TRACKING_URI": settings.mlflow_tracking_uri},
    }


def create_mlflow_mcp_server():
    """MLflow MCP stdio 서버 인스턴스(공통 타임아웃·캐시 설정)."""
    from agents.mcp import MCPServerStdio

    return MCPServerStdio(
        params=mlflow_mcp_stdio_params(),
        name="mlflow-mcp",
        cache_tools_list=True,
        client_session_timeout_seconds=settings.mcp_client_session_timeout_seconds,
    )


async def mlflow_mcp_health() -> tuple[bool, list[str], str | None]:
    """MCP 세션 초기화까지 성공하면 True와 노출된 도구 이름 목록을 반환한다."""
    from agents.mcp import MCPServerManager

    server = create_mlflow_mcp_server()
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

def _coerce_tool_arguments(raw: Any) -> Any:
    """도구 인자를 metadata에 넣을 수 있는 JSON 값으로 맞춘다."""
    if isinstance(raw, dict):
        arguments = raw.get("arguments")
    else:
        arguments = getattr(raw, "arguments", None)
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return arguments if arguments is not None else {}


# MCP 결과에서 tool_calls 추출. Runner 결과는 steps가 없고 new_items에 ToolCallItem이 있다.
def extract_tool_calls(result):
    from agents.items import ToolCallItem

    tool_calls = []
    for item in getattr(result, "new_items", []) or []:
        if not isinstance(item, ToolCallItem):
            continue
        name = item.tool_name
        if not name:
            continue
        tool_calls.append({
            "tool": name,
            "arguments": _coerce_tool_arguments(item.raw_item),
        })
    if not tool_calls:
        tool_calls = ["No tool calls"]
    return tool_calls