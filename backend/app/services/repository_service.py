from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import Report
from backend.app.schemas import RunSummary


def save_report(
    db: Session,
    *,
    experiment_name: str,
    metric_key: str,
    report_title: str,
    report: str,
    model: str,
    top_runs: list[RunSummary] | list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> Report:
    top_runs_payload = [
        run.model_dump() if isinstance(run, RunSummary) else run for run in top_runs
    ]
    row = Report(
        experiment_name=experiment_name,
        metric_key=metric_key,
        report_title=report_title,
        report=report,
        model=model,
        top_runs=top_runs_payload,
        report_metadata=metadata or {},
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row

def get_report_list(
    db: Session,
    *,
    experiment_name: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[Report]:
    query = select(Report).order_by(Report.created_at.desc())
    if experiment_name:
        query = query.where(Report.experiment_name == experiment_name) # 실험 이름으로 보고서 필터링
    query = query.limit(limit).offset(offset) # 페이지네이션
    return list(db.scalars(query).all())