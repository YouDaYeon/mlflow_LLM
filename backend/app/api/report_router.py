from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.db.models import Report
from backend.app.schemas import (
    ReportListItem,
    ReportListResponse,
    ReportRequest,
    ReportResponse,
    ReportUpdateRequest,
)
from backend.app.services.report_service import generate_report
from backend.app.services.repository_service import (
    delete_report,
    get_report_by_id,
    get_report_list,
    save_report,
    update_report,
)

router = APIRouter(prefix="/report", tags=["report"])


def _to_report_response(row: Report) -> ReportResponse:
    return ReportResponse(
        id=row.id,
        created_at=row.created_at,
        updated_at=getattr(row, "updated_at", None) or row.created_at,
        experiment_name=row.experiment_name,
        metric_key=row.metric_key,
        report=row.report,
        report_title=row.report_title,
        model=row.model,
        metadata=row.report_metadata,
    )


def _to_report_list_item(row: Report) -> ReportListItem:
    return ReportListItem(
        id=row.id,
        created_at=row.created_at,
        updated_at=getattr(row, "updated_at", None) or row.created_at,
        experiment_name=row.experiment_name,
        metric_key=row.metric_key,
        report_title=row.report_title,
    )


@router.post("/", response_model=ReportResponse)
def create_report(payload: ReportRequest, db: Session = Depends(get_db)) -> ReportResponse:
    report_text, model, report_title = generate_report(
        experiment_name=payload.experiment_name,
        metric_key=payload.metric_key,
        top_runs=payload.top_runs,
    )

    metadata = {
        "runs_list": [run.run_name for run in payload.top_runs],
    }
    saved = save_report(
        db,
        experiment_name=payload.experiment_name,
        metric_key=payload.metric_key,
        report_title=report_title,
        report=report_text,
        model=model,
        top_runs=payload.top_runs,
        metadata=metadata,
    )
    return _to_report_response(saved)


@router.get("/", response_model=ReportListResponse)
def list_reports(
    experiment_name: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> ReportListResponse:
    rows = get_report_list(
        db,
        experiment_name=experiment_name,
        limit=limit,
        offset=offset,
    )

    return ReportListResponse(items=[_to_report_list_item(row) for row in rows])


@router.get("/{report_id}", response_model=ReportResponse)
def get_report(report_id: UUID, db: Session = Depends(get_db)) -> ReportResponse:
    row = get_report_by_id(db, report_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return _to_report_response(row)


@router.patch("/{report_id}", response_model=ReportResponse)
def patch_report(
    report_id: UUID,
    payload: ReportUpdateRequest,
    db: Session = Depends(get_db),
) -> ReportResponse:
    row = update_report(
        db,
        report_id,
        report=payload.report,
        report_title=payload.report_title,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return _to_report_response(row)


@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_report(report_id: UUID, db: Session = Depends(get_db)) -> Response:
    deleted = delete_report(db, report_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Report not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
