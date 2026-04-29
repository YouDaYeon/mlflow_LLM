from fastapi import APIRouter

from backend.app.services.mlflow_service import (
    get_experiment as get_experiment_service,
    get_metric_keys as get_metric_keys_service,
    get_runs as get_runs_service,
    get_top_runs as get_top_runs_service,
    list_experiment_names,
)

router = APIRouter(prefix="/mlflow", tags=["mlflow"])



@router.get("/names")
def get_experiment_names():
    return {"experiment_names": list_experiment_names()}

@router.get("/experiment")
def get_experiment(experiment_name: str):
    return {"experiment": get_experiment_service(experiment_name)}

@router.get("/runs")
def get_runs(experiment_name: str):
    return {"runs": get_runs_service(experiment_name)}

@router.get("/metric-keys")
def get_metric_keys(experiment_name: str):
    return {"metric_keys": get_metric_keys_service(experiment_name)}

@router.get("/top-runs")
def get_top_runs(experiment_name: str, metric_key: str, top_k: int, ascending: bool = False):
    return {"top_runs": get_top_runs_service(experiment_name, metric_key, top_k, ascending)}