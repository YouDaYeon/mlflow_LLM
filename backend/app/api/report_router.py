from fastapi import APIRouter


from backend.app.schemas import ReportRequest, ReportResponse
from backend.app.services.report_service import generate_report

router = APIRouter(prefix="/report", tags=["report"])


@router.post("/", response_model=ReportResponse)
def create_report(payload: ReportRequest) -> ReportResponse:
    report_text, model, report_title = generate_report(
        experiment_name=payload.experiment_name,
        metric_key=payload.metric_key,
        top_runs=payload.top_runs,
    )

    metadata = {
        "runs_list": [run.run_name for run in payload.top_runs],
    }
    return ReportResponse(
        experiment_name=payload.experiment_name,
        metric_key=payload.metric_key,
        report=report_text,
        report_title=report_title,
        model=model,
        metadata=metadata,
    )
