from typing import Any, ClassVar

from pydantic import BaseModel, Field



class RunBase(BaseModel):

    BASE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "run_id",
        "experiment_id",
        "status",
        "artifact_uri",
        "start_time",
        "end_time",
    )
    COLUMN_PREFIX_GROUPS: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("metrics.",),
        ("params.",),
        ("tags.",),
    )

    @classmethod
    def select_columns(cls, all_columns: list[str]) -> list[str]:
        base_columns = [column for column in cls.BASE_COLUMNS if column in all_columns]
        grouped_columns = [
            column
            for column in all_columns
            if any(column.startswith(prefix) for group in cls.COLUMN_PREFIX_GROUPS for prefix in group)
        ]
        return [*base_columns, *grouped_columns]



class RunSummary(BaseModel):
    run_id: str
    run_name: str
    status: str
    metric_score: float
    metric_key: str
    start_time: str | None = None
    params: dict[str, str]
    metrics: dict[str, float]



# report
class ReportRequest(BaseModel):
    experiment_name: str = Field(..., min_length=1)
    metric_key: str = Field(default="metrics/mAP50-95(B)")
    top_runs: list[RunSummary] = Field(..., min_length=1, max_length=3)

class ReportResponse(BaseModel):
    experiment_name: str
    metric_key: str
    report: str
    report_title: str
    model: str
    metadata: dict[str, Any]


# chat
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    report_text: str | None = None
    experiment_name: str | None = None
    metric_key: str | None = None


class ChatResponse(BaseModel):
    answer: str
    intent: str
    metadata: dict[str, Any] = Field(default_factory=dict)
