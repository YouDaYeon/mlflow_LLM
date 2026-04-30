import mlflow
import pandas as pd

from fastapi import HTTPException
from backend.app.config import settings
from backend.app.schemas import RunBase, RunSummary

# MLflow 실험 설정
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

METRIC_PREFIX = "metrics."
PARAM_PREFIX = "params."

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
        "experiment_id": experiment.experiment_id,
        "artifact_location": experiment.artifact_location,
        "tags": experiment.tags,
        "lifecycle_stage": experiment.lifecycle_stage,
        "creation_time": experiment.creation_time,
    }


def _search_runs_df(experiment_name: str) -> pd.DataFrame:
    experiment = _get_experiment_or_404(experiment_name)
    return mlflow.search_runs(experiment_ids=[experiment.experiment_id])

################ 여기 get_runs 나중에 지울거임 (지금은 일단 두고 각 run 세부 함수 추가 후 지울 수 있게)
def get_runs(experiment_name: str) -> list[dict]:
    runs = _search_runs_df(experiment_name)
    return runs.to_dict(orient="records")

## tags 관련 함수 추가
# [c for i, c in enumerate(runs_df.columns) if c.startswith("tags.")]
# ['tags.mlflow.user', 'tags.mlflow.source.type', 'tags.mlflow.source.name', 'tags.mlflow.runName']
def list_run_names(experiment_name: str) -> list[str]:
    runs_df = _search_runs_df(experiment_name)
    return runs_df["tags.mlflow.runName"].unique().tolist()

def list_run_users(experiment_name: str) -> list[str]:
    runs_df = _search_runs_df(experiment_name)
    return runs_df["tags.mlflow.user"].unique().tolist()

## metric 관련 함수 추가
def list_run_metrics(experiment_name: str) -> list[str]:
    runs_df = _search_runs_df(experiment_name)
    return runs_df.columns[runs_df.columns.str.startswith("metrics.")].tolist()

def _normalize_metric_key(metric_key: str) -> str:
    return metric_key if metric_key.startswith(METRIC_PREFIX) else f"{METRIC_PREFIX}{metric_key}"


def _metric_columns(runs_df: pd.DataFrame) -> list[str]:
    return sorted([col for col in runs_df.columns if col.startswith(METRIC_PREFIX)])


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








# chat service에서 이용
def search_runs_with_filters(
    experiment_name: str,
    *,
    recent_days: int | None = None,
    user: str | None = None,
    status: str | None = None,
    tag_key: str | None = None,
    tag_value: str | None = None,
    top_k: int | None = None,
    metric_key: str | None = None,
) -> list[dict]:
    runs_df = _search_runs_df(experiment_name)
    if runs_df.empty:
        return []

    filtered = runs_df.copy()

    if recent_days is not None and recent_days > 0 and "start_time" in filtered.columns:
        start_times = pd.to_datetime(filtered["start_time"], unit="ms", errors="coerce")
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=recent_days)
        filtered = filtered[start_times >= cutoff]

    if user:
        user_col = "tags.mlflow.user" if "tags.mlflow.user" in filtered.columns else None
        if user_col:
            user_mask = filtered[user_col].astype(str).str.lower() == user.lower()
            filtered = filtered[user_mask]
        else:
            return []

    if status and "status" in filtered.columns:
        filtered = filtered[filtered["status"].astype(str).str.upper() == status.upper()]

    if tag_key:
        full_tag_col = tag_key if tag_key.startswith("tags.") else f"tags.{tag_key}"
        if full_tag_col in filtered.columns:
            if tag_value:
                filtered = filtered[filtered[full_tag_col].astype(str) == tag_value]
            else:
                filtered = filtered[filtered[full_tag_col].notna()]
        else:
            return []

    if filtered.empty:
        return []

    if top_k is not None and top_k > 0:
        if not metric_key:
            raise HTTPException(status_code=400, detail="top_k 조회 시 metric_key가 필요합니다.")

        top_runs = get_top_runs(
            experiment_name=experiment_name,
            metric_key=metric_key,
            top_k=top_k,
            ascending=False,
        )
        allowed_run_ids = set(filtered["run_id"].astype(str).tolist()) if "run_id" in filtered.columns else set()
        top_run_ids = [run.run_id for run in top_runs if run.run_id in allowed_run_ids]
        if not top_run_ids:
            return []

        filtered_by_top = filtered[filtered["run_id"].astype(str).isin(top_run_ids)].copy()
        order_map = {run_id: idx for idx, run_id in enumerate(top_run_ids)}
        filtered_by_top["_rank_order"] = filtered_by_top["run_id"].astype(str).map(order_map)
        filtered = filtered_by_top.sort_values(by="_rank_order", ascending=True)
    elif "start_time" in filtered.columns:
        sort_start = pd.to_datetime(filtered["start_time"], unit="ms", errors="coerce")
        filtered = filtered.assign(_sort_start=sort_start).sort_values(by="_sort_start", ascending=False)

    result: list[dict] = []
    for _, row in filtered.iterrows():
        result.append(
            {
                "run_id": str(row.get("run_id", "")),
                "run_name": str(row.get("tags.mlflow.runName", "")),
                "status": str(row.get("status", "")),
                "start_time": _to_iso_start_time(row.get("start_time")),
                "created_by": str(row.get("tags.mlflow.user") or row.get("user_id") or ""),
            }
        )
    return result


def get_metric_keys(experiment_name: str) -> list[str]:
    runs_df = _search_runs_df(experiment_name)
    return _metric_columns(runs_df)


def _resolve_metric_column(metric_columns: list[str], metric_query: str) -> str | None:
    if not metric_columns:
        return None

    q = metric_query.strip().lower()
    if not q:
        return None

    exact = [col for col in metric_columns if col.lower() == q]
    if exact:
        return exact[0]

    with_prefix = q if q.startswith(METRIC_PREFIX) else f"{METRIC_PREFIX}{q}"
    exact_with_prefix = [col for col in metric_columns if col.lower() == with_prefix]
    if exact_with_prefix:
        return exact_with_prefix[0]

    no_prefix_exact = [col for col in metric_columns if col.removeprefix(METRIC_PREFIX).lower() == q]
    if no_prefix_exact:
        return no_prefix_exact[0]

    contains = [col for col in metric_columns if q in col.lower() or q in col.removeprefix(METRIC_PREFIX).lower()]
    if contains:
        return sorted(contains, key=len)[0]

    return None


def filter_runs_by_metric_threshold(
    experiment_name: str,
    metric_query: str,
    conditions: list[tuple[str, float]],
    limit: int = 20,
) -> tuple[str, list[dict]]:
    runs_df = _search_runs_df(experiment_name)
    metric_columns = _metric_columns(runs_df)
    metric_column = _resolve_metric_column(metric_columns, metric_query)
    if not metric_column:
        raise HTTPException(status_code=404, detail=f"metric_key를 찾을 수 없습니다: {metric_query}")

    scored_df = _scored_runs_df(runs_df, metric_column)
    if scored_df.empty:
        return metric_column, []

    filtered = scored_df
    for operator, threshold in conditions:
        if operator == ">":
            filtered = filtered[filtered[metric_column] > threshold]
        elif operator == ">=":
            filtered = filtered[filtered[metric_column] >= threshold]
        elif operator == "<":
            filtered = filtered[filtered[metric_column] < threshold]
        elif operator == "<=":
            filtered = filtered[filtered[metric_column] <= threshold]
        elif operator == "==":
            filtered = filtered[filtered[metric_column] == threshold]
        else:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 연산자입니다: {operator}")

    if filtered.empty:
        return metric_column, []

    has_upper_bound = any(op in {"<", "<="} for op, _ in conditions)
    sorted_df = filtered.sort_values(by=metric_column, ascending=has_upper_bound).head(limit)
    result = []
    for _, row in sorted_df.iterrows():
        result.append(
            {
                "run_id": str(row.get("run_id", "")),
                "run_name": str(row.get("tags.mlflow.runName", "")),
                "status": str(row.get("status", "")),
                "metric_score": float(row[metric_column]),
                "metric_key": metric_column,
            }
        )
    return metric_column, result


def get_top_runs(
    experiment_name: str,
    metric_key: str,
    top_k: int,
    ascending: bool = False,
) -> list[RunSummary]:
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k는 1 이상의 정수여야 합니다.")

    runs_df = _search_runs_df(experiment_name)
    metric_column = _normalize_metric_key(metric_key)
    if metric_column not in runs_df.columns:
        raise HTTPException(status_code=404, detail=f"metric_key를 찾을 수 없습니다: {metric_key}")

    scored_df = _scored_runs_df(runs_df, metric_column)
    if scored_df.empty:
        return []

    top_df = scored_df.sort_values(by=metric_column, ascending=ascending).head(top_k)
    return [_row_to_run_summary(row, metric_column) for _, row in top_df.iterrows()]

# chat service에서 이용
def list_recent_experiment_names(limit: int = 5) -> list[str]:
    experiments = mlflow.search_experiments()
    sorted_experiments = sorted(experiments, key=lambda exp: exp.creation_time or 0, reverse=True)
    return [exp.name for exp in sorted_experiments[:limit]]