from __future__ import annotations

import json
from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


def split_run_start_time(started: object) -> tuple[str, str]:
    """MLflow start_time(ms) / ISO 문자열 등을 UI용 날짜·시각으로 나눈다."""
    if started is None:
        return "-", "-"
    if isinstance(started, (int, float)) and not isinstance(started, bool):
        ts = float(started)
        if ts > 1e12:
            ts /= 1000.0
        try:
            dt = datetime.fromtimestamp(ts)
            return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
        except (ValueError, OSError):
            return "-", "-"
    s = str(started).strip()
    if not s or s.lower() in ("none", "nan", "nat"):
        return "-", "-"
    try:
        ts = float(s)
        if ts > 1e12:
            ts /= 1000.0
        dt = datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
    except ValueError:
        return s, "-"


def format_display_datetime(value: object) -> str:
    """UI 출력용 datetime 문자열(YYYY-MM-DD HH:MM:SS)로 정규화한다."""
    date_s, time_s = split_run_start_time(value)
    if date_s == "-" and time_s == "-":
        return "-"
    if time_s == "-":
        return date_s
    return f"{date_s} {time_s}"


class RunDisplayDetails(BaseModel):
    """Streamlit `Parameters` 컬럼에 넣는 JSON 페이로드."""

    run_id: str | None = None
    metric_key: str | None = None
    params: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)


class RunDisplayRow(BaseModel):
    """실험 run 표시 한 줄(프론트 `st.dataframe` / 챗봇 미리보기 공통)."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(alias="Name")
    score: float | None = Field(default=None, alias="Score")
    status: str = Field(alias="Status")
    date: str = Field(alias="Date")
    time: str = Field(alias="Time")
    parameters: str = Field(alias="Parameters")

    @classmethod
    def from_run_fields(
        cls,
        *,
        run_name: str | None,
        run_id: str | None,
        metric_key: str | None,
        metric_score: float | None,
        status: str | None,
        start_time: object,
        params: dict[str, str],
        metrics: dict[str, float],
    ) -> RunDisplayRow:
        date_s, time_s = split_run_start_time(start_time)
        details = RunDisplayDetails(
            run_id=run_id,
            metric_key=metric_key,
            params=params,
            metrics=metrics,
        )
        return cls(
            Name=run_name or "-",
            Score=metric_score,
            Status=status or "-",
            Date=date_s,
            Time=time_s,
            Parameters=json.dumps(details.model_dump(), ensure_ascii=False),
        )

    @classmethod
    def from_summary_dict(cls, r: dict[str, Any]) -> RunDisplayRow:
        """`_build_run_summaries_from_raw_runs` 형태 dict -> 표시 행."""
        return cls.from_run_fields(
            run_name=r.get("run_name"),
            run_id=r.get("run_id"),
            metric_key=r.get("metric_key"),
            metric_score=r.get("metric_score"),
            status=r.get("status"),
            start_time=r.get("start_time"),
            params=r.get("params") or {},
            metrics=r.get("metrics") or {},
        )

    @classmethod
    def from_mlflow_raw(cls, raw: dict[str, Any], metric_key: str | None = None) -> RunDisplayRow:
        """MLflow `search_runs` 레코드(dict) -> 표시 행. metric_key가 있으면 해당 컬럼으로 Score."""
        metric_col: str | None = None
        if metric_key and str(metric_key).strip():
            mk = str(metric_key).strip()
            metric_col = mk if mk.startswith("metrics.") else f"metrics.{mk}"
        metric_score = None
        if metric_col and raw.get(metric_col) is not None:
            try:
                metric_score = float(raw[metric_col])
            except (TypeError, ValueError):
                pass

        params = {
            k.removeprefix("params."): str(v)
            for k, v in raw.items()
            if k.startswith("params.") and v is not None
        }
        metrics: dict[str, float] = {}
        for k, v in raw.items():
            if k.startswith("metrics.") and v is not None:
                try:
                    metrics[k.removeprefix("metrics.")] = float(v)
                except (TypeError, ValueError):
                    continue

        run_id = str(raw.get("run_id", "") or "") or None
        run_name = str(raw.get("tags.mlflow.runName", "") or "") or None
        status = str(raw.get("status", "") or "") or None

        return cls.from_run_fields(
            run_name=run_name,
            run_id=run_id,
            metric_key=metric_col,
            metric_score=metric_score,
            status=status,
            start_time=raw.get("start_time"),
            params=params,
            metrics=metrics,
        )

    def to_dataframe_row(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)


class ExpDisplayRow(BaseModel):
    """실험 정보 표시 한 줄(챗봇/프론트 공통 포맷)."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(alias="Name")
    experiment_id: str = Field(alias="Experiment ID")
    lifecycle_stage: str = Field(alias="Lifecycle Stage")
    created_date: str = Field(alias="Created Date")
    created_time: str = Field(alias="Created Time")
    artifact_location: str = Field(alias="Artifact Location")
    tags: dict[str, str] = Field(default_factory=dict, alias="Tags")

    @classmethod
    def from_experiment_dict(cls, experiment_name: str, exp: dict[str, Any]) -> ExpDisplayRow:
        created_date, created_time = split_run_start_time(exp.get("creation_time"))
        raw_tags = exp.get("tags") or {}
        tags = {str(k): str(v) for k, v in raw_tags.items()}
        return cls(
            Name=experiment_name or "-",
            **{
                "Experiment ID": str(exp.get("experiment_id", "-")),
                "Lifecycle Stage": str(exp.get("lifecycle_stage", "-")),
                "Created Date": created_date,
                "Created Time": created_time,
                "Artifact Location": str(exp.get("artifact_location", "-")),
                "Tags": tags,
            },
        )

    def to_markdown(self) -> str:
        d = self.model_dump(by_alias=True)
        tags = d["Tags"]
        tag_text = ", ".join(f"{k}={v}" for k, v in tags.items()) if tags else "-"
        return "\n".join(
            [
                f"`{d['Name']}` 실험 정보",
                "",
                "| Name | Experiment ID | Lifecycle Stage | Created Date | Created Time |",
                "|------|----------------|-----------------|--------------|--------------|",
                f"| {d['Name']} | {d['Experiment ID']} | {d['Lifecycle Stage']} | {d['Created Date']} | {d['Created Time']} |",
                "",
                f"- Artifact Location: `{d['Artifact Location']}`",
                f"- Tags: {tag_text}",
            ]
        )

    def to_metadata(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)


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
    metric_key: str = Field(...)
    top_runs: list[RunSummary] = Field(..., min_length=1, max_length=3)

class ReportResponse(BaseModel):
    id: UUID
    created_at: datetime
    updated_at: datetime | None = None
    experiment_name: str
    metric_key: str
    report: str
    report_title: str
    model: str
    metadata: dict[str, Any]


class ReportUpdateRequest(BaseModel):
    report: str = Field(..., min_length=1)
    report_title: str | None = None


class ReportListItem(BaseModel):
    id: UUID
    created_at: datetime
    updated_at: datetime | None = None
    experiment_name: str
    metric_key: str
    report_title: str


class ReportListResponse(BaseModel):
    items: list[ReportListItem]


# chat
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)  # 직전 턴들 (session_id 없을 때 fallback)
    session_id: UUID | None = None
    report_text: str | None = None
    experiment_name: str | None = None
    metric_key: str | None = None

class ChatResponse(BaseModel):
    answer: str
    intent: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_id: UUID


class ChatSessionCreateResponse(BaseModel):
    session_id: UUID
    title: str | None = None
    created_at: datetime
    updated_at: datetime


class ChatSessionListItem(BaseModel):
    session_id: UUID
    title: str | None = None
    created_at: datetime
    updated_at: datetime


class ChatSessionListResponse(BaseModel):
    items: list[ChatSessionListItem]


class ChatMessageItem(BaseModel):
    id: UUID
    role: str
    content: str
    intent: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ChatMessageListResponse(BaseModel):
    session_id: UUID
    items: list[ChatMessageItem]


class McpHealthResponse(BaseModel):
    """MLflow MCP(stdio) 세션·도구 목록 로드 여부."""

    ok: bool
    mlflow_tracking_uri: str
    tool_names: list[str] = Field(default_factory=list)
    error: str | None = None
