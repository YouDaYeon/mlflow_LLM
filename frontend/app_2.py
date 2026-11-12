import html
import json
import importlib
import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from streamlit.errors import StreamlitAPIException

from backend.app import schemas as _backend_schemas

# Streamlit은 import를 캐시하므로 schemas 변경이 UI에 안 먹을 수 있음 → 매 실행 reload
importlib.reload(_backend_schemas)
from backend.app.schemas import format_display_datetime
from backend.app.services.report_service import infer_metric_direction

_DISPLAY_TZ = ZoneInfo("Asia/Seoul")


def _format_ui_datetime(value: object) -> str:
    """UI 시각을 항상 Asia/Seoul로 표시 (Streamlit import 캐시 대비 로컬 보정)."""
    if value is None or value == "":
        return "-"
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S")
    s = str(value).strip()
    if not s:
        return "-"
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        # MLflow epoch 등
        return format_display_datetime(value)

st.set_page_config(
    page_title="MLflow + FastAPI + Streamlit 템플릿",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Keep chatbot markdown tables/messages inside dialog width */
    [data-testid="stChatMessage"] table {
        width: 100% !important;
        table-layout: fixed;
    }
    [data-testid="stChatMessage"] th,
    [data-testid="stChatMessage"] td {
        white-space: normal !important;
        word-break: break-word;
    }
    [data-testid="stChatMessage"] code {
        white-space: pre-wrap !important;
        word-break: break-all;
    }

    /* Floating QA chatbot button (bottom-right).
       Streamlit attaches `st-key-<key>` to the element container when a
       widget is created with a `key`. We use that to make only the QA
       chatbot button float at the bottom-right. */
    .st-key-qa_floating_chat_btn {
        position: fixed !important;
        right: 28px !important;
        bottom: 28px !important;
        z-index: 9999 !important;
        width: auto !important;
    }
    .st-key-qa_floating_chat_btn button {
        border-radius: 999px !important;
        padding: 14px 22px !important;
        font-weight: 600 !important;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25) !important;
    }

    /* 실험 요약 */
    .exp-header { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 4px; }
    .exp-title { font-size: 2rem; font-weight: 700; line-height: 1.2; }
    .exp-subline { font-size: 0.9rem; opacity: 0.6; margin: 2px 0 14px; }
    .exp-field { margin-bottom: 12px; }
    .exp-field-label { font-size: 0.8rem; opacity: 0.6; margin-bottom: 4px; }
    .exp-field-value { font-size: 1rem; word-break: break-all; }
    .exp-empty { opacity: 0.6; }
    .exp-tag {
        display: inline-block; font-size: 0.78rem; padding: 2px 8px; margin: 0 6px 6px 0;
        border-radius: 6px; background: rgba(128, 128, 128, 0.15);
    }
    /* 사이드바 파트 제목 */
    .sidebar-section-title { font-size: 1.25rem; font-weight: 700; margin: 4px 0 8px; }
    .sidebar-group-title { font-size: 0.95rem; font-weight: 600; }
    .report-criteria { font-size: 0.9rem; margin: 2px 0 8px; }
    .report-list-scope { font-size: 0.85rem; font-weight: 600; margin: 4px 0 2px; word-break: break-all; }
    .report-title { font-size: 1.35rem; font-weight: 700; line-height: 1.35; }

    /* 저장된 보고서: 두 줄 목록 (라디오 원형 숨김, 선택 항목 배경 강조) */
    .st-key-sidebar_report_list [role="radiogroup"] { gap: 2px; width: 100%; }
    .st-key-sidebar_report_list [role="radiogroup"] > label {
        width: 100%; margin: 0; padding: 8px 10px; border-radius: 8px; cursor: pointer;
    }
    .st-key-sidebar_report_list [role="radiogroup"] > label:hover { background: rgba(128, 128, 128, 0.12); }
    .st-key-sidebar_report_list [role="radiogroup"] > label:has(input:checked) {
        background: rgba(255, 75, 75, 0.12);
    }
    .st-key-sidebar_report_list [role="radiogroup"] > label > div:first-child { display: none; }
    .st-key-sidebar_report_list [role="radiogroup"] > label [data-testid="stMarkdownContainer"] p {
        display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
        overflow: hidden; font-weight: 500; line-height: 1.35; margin: 0;
    }
    .st-key-sidebar_report_list [role="radiogroup"] > label [data-testid="stCaptionContainer"] p {
        display: block; -webkit-line-clamp: unset; font-weight: 400;
    }
    /* ── 보고서 팝업: 브라우저 보이는 영역에 비례 (report-dialog-marker가 있는 팝업에만 적용) ── */
    div[role="dialog"]:has(.report-dialog-marker) {
        --report-dialog-h: 90vh;
        --report-fixed-h: 250px;  /* 제목줄 + 정보 2줄 + 하단 버튼 + 여백 */
        --report-body-h: max(160px, calc(var(--report-dialog-h) - var(--report-fixed-h)));
        width: 92vw !important; max-width: 92vw !important;
        height: var(--report-dialog-h) !important; max-height: var(--report-dialog-h) !important;
    }
    div[role="dialog"]:has(.report-notice-marker) {
        --report-fixed-h: 320px;  /* 생성·수정 안내 문구가 있을 때 */
    }
    /* 본문만 스크롤: 고정 높이는 바깥 stLayoutWrapper의 flex-basis로 걸리므로 그쪽을 덮어쓴다 */
    div[role="dialog"]:has(.report-dialog-marker)
        [data-testid="stLayoutWrapper"]:has(> .st-key-report_dialog_body) {
        flex: 0 0 var(--report-body-h) !important; height: var(--report-body-h) !important;
    }
    .report-criteria-line { font-size: 0.9rem; margin-bottom: 10px; }
    .chat-report-ref { font-size: 0.9rem; margin-bottom: 4px; }

    /* 챗봇 팝업: 넓게 + 대화 목록 */
    /* ── 챗봇 팝업: 브라우저 보이는 영역에 비례 (chat-dialog-marker가 있는 팝업에만 적용) ── */
    div[role="dialog"]:has(.chat-dialog-marker) {
        --chat-dialog-h: 90vh;
        width: 92vw !important; max-width: 92vw !important;
        height: var(--chat-dialog-h) !important; max-height: var(--chat-dialog-h) !important;
    }
    /* 왼쪽 대화 목록 240px 고정, 채팅 본문은 남은 너비 전부 */
    div[role="dialog"]:has(.chat-dialog-marker)
        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-chat_session_list)
        > [data-testid="stColumn"]:first-child {
        flex: 0 0 240px !important; width: 240px !important; min-width: 240px !important;
    }
    div[role="dialog"]:has(.chat-dialog-marker)
        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-chat_session_list)
        > [data-testid="stColumn"]:last-child {
        flex: 1 1 0 !important; width: auto !important; min-width: 0 !important;
    }
    /* 높이 배분: 제목·참조 문맥·입력창은 고정, 대화 내용(과 목록)만 스크롤.
       빼는 값 = 팝업 제목줄 + 채팅 제목 + 참조 문맥 + 입력창 + 여백 */
    /* st.container(height=...)의 고정 높이는 바깥 stLayoutWrapper의 flex-basis(예: 0 0 400px)로
       걸리므로 height가 아니라 flex-basis를 덮어쓴다 */
    div[role="dialog"]:has(.chat-dialog-marker) {
        --chat-fixed-h: 256px;  /* 목록 펼침: 팝업 제목줄+채팅 제목+참조 문맥+입력창+여백 */
        --chat-msgs-h: max(140px, calc(var(--chat-dialog-h) - var(--chat-fixed-h)));
        --chat-list-h: max(160px, calc(var(--chat-dialog-h) - var(--chat-list-fixed-h, 220px)));
    }
    div[role="dialog"]:has(.chat-list-collapsed-marker) {
        --chat-fixed-h: 300px;  /* 목록 접힘: 제목 줄에 '목록 열기' 버튼이 있어 조금 더 높음 */
    }
    div[role="dialog"]:has(.chat-dialog-marker)
        [data-testid="stLayoutWrapper"]:has(> .st-key-chat_messages_box) {
        flex: 0 0 var(--chat-msgs-h) !important; height: var(--chat-msgs-h) !important;
    }
    div[role="dialog"]:has(.chat-dialog-marker)
        [data-testid="stLayoutWrapper"]:has(> .st-key-chat_session_list) {
        flex: 0 0 var(--chat-list-h) !important; height: var(--chat-list-h) !important;
    }
    /* 전송 버튼 칸은 글자 폭만큼 확보 (좁은 화면에서 '전송'이 세로로 줄바꿈되지 않게) */
    div[role="dialog"]:has(.chat-dialog-marker)
        [data-testid="stColumn"]:has([data-testid="stFormSubmitButton"]) {
        flex: 0 0 auto !important; width: auto !important; min-width: max-content !important;
    }
    div[role="dialog"]:has(.chat-dialog-marker) [data-testid="stFormSubmitButton"] button { white-space: nowrap; }
    /* 좁은 화면에서도 입력창과 전송 버튼을 한 줄에 (Streamlit 기본은 칸을 세로로 쌓음) */
    div[role="dialog"]:has(.chat-dialog-marker)
        [data-testid="stHorizontalBlock"]:has([data-testid="stFormSubmitButton"]) {
        flex-wrap: nowrap !important;
    }
    div[role="dialog"]:has(.chat-dialog-marker)
        [data-testid="stHorizontalBlock"]:has([data-testid="stFormSubmitButton"])
        > [data-testid="stColumn"]:first-child {
        flex: 1 1 0 !important; width: auto !important; min-width: 0 !important;
    }
    /* 질문 입력창 안의 'Press Enter to submit form' 안내 숨김 (전송 버튼이 옆에 있음) */
    .st-key-chat_prompt_text [data-testid="InputInstructions"] { display: none; }
    .chat-title { font-size: 1.15rem; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    [class*="st-key-chat_row_"] { border-radius: 8px; padding: 2px 4px; }
    /* 대화 목록 ⋯: 칸·화살표 없이 아이콘만 */
    [class*="st-key-chat_row_"] [data-testid="stPopover"] button {
        border: none !important; background: transparent !important; padding: 4px !important; min-height: 0 !important;
    }
    /* 대화 목록 삭제 메뉴: 기본 320px 판 대신 내용 크기로 (화면 밖으로 나가지 않게) */
    [data-testid="stPopoverBody"]:has([class*="st-key-chat_del_"]) {
        min-width: 0 !important; width: max-content !important; padding: 4px 8px !important;
    }
    [data-testid="stPopoverBody"]:has([class*="st-key-chat_del_"]) button {
        font-size: 0.85rem !important; min-height: 0 !important; padding: 4px 6px !important;
    }
    /* 버튼 안 구조: [⋯ 아이콘 묶음][⌄ 펼침 화살표 묶음] → 화살표 묶음만 숨김 */
    [class*="st-key-chat_row_"] [data-testid="stPopover"] button > div > div:last-child:not(:first-child) {
        display: none !important;
    }
    [class*="st-key-chat_row_"]:hover { background: rgba(128, 128, 128, 0.10); }
    [class*="st-key-chat_row_"] [data-testid="stCaptionContainer"] { margin-top: -10px; padding-left: 2px; }
    .st-key-chat_session_list [class*="st-key-chat_pick_"] button {
        width: 100%; min-width: 0; justify-content: flex-start; text-align: left; padding: 0;
    }
    .st-key-chat_session_list [class*="st-key-chat_pick_"] button div,
    .st-key-chat_session_list [class*="st-key-chat_pick_"] button p {
        min-width: 0; max-width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .run-detail-title { font-size: 1.05rem; font-weight: 600; margin: 8px 0 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8011")
# 브라우저에서 여는 링크이므로 도커 내부 주소(mlflow:2522)가 아닌 외부 접속 주소를 사용
def _env_from_dotenv(key: str) -> str | None:
    """로컬 실행 시 프로젝트 루트 .env에서 특정 키만 읽는다.
    (.env의 API_BASE_URL은 도커용이라 전체 로드하면 로컬 백엔드 주소가 바뀜)"""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return None
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    return dotenv_values(env_path).get(key) or None


MLFLOW_UI_URL = (
    os.getenv("MLFLOW_UI_URL") or _env_from_dotenv("MLFLOW_UI_URL") or "http://localhost:2522"
).rstrip("/")
DEFAULT_TOP_K_LABEL = "Default(전체)"
NO_SELECTION_LABEL = "(선택 안함)"
DIRECT_INPUT_LABEL = "(직접 입력)"

CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "6"))
CHAT_WELCOME_MESSAGE = "안녕하세요. 실험 관련 질문을 입력해 주세요."

SESSION_DEFAULTS: dict[str, Any] = {
    "experiment_names": [],
    "names_fetch_error": "",
    "selected_experiment": "",
    "selected_metric": "",
    "selected_top_k": DEFAULT_TOP_K_LABEL,
    "selected_runs_data": [],
    "selected_runs_meta": {},
    "experiment_info": {},
    "generated_report_text": "",
    "generated_report_title": "",
    "generated_report_meta": {},
    "report_edit_mode": False,
    "report_edit_draft": "",
    "report_edit_title_draft": "",
    "chat_report": {},  # '이 보고서로 질문하기'로 챗봇에 연결한 보고서
    "chat_session_id": None,
    "chat_list_collapsed": False,
    "chat_resolved_context": {},  # 답변 metadata에서 확정된 experiment/metric/top_k
    "chat_messages": [
        {
            "role": "assistant",
            "content": CHAT_WELCOME_MESSAGE,
        }
    ],
}

_RUN_STATUS_LABELS = {
    "FINISHED": "완료",
    "RUNNING": "실행 중",
    "SCHEDULED": "예약됨",
    "FAILED": "실패",
    "KILLED": "중단",
}
_RUN_STATUS_COLORS = {
    "완료": "#1a7f37",
    "실행 중": "#0969da",
    "실패": "#cf222e",
    "중단": "#cf222e",
}
# 상세 보기 '학습 설정'에 먼저 보여줄 항목 (label, 후보 param key)
_TRAINING_PARAM_SPECS: list[tuple[str, list[str]]] = [
    ("모델 종류", ["model_name", "model", "model_type", "architecture", "arch"]),
    ("학습률", ["learning_rate", "lr"]),
    ("배치 크기", ["batch_size", "batch"]),
    ("Epoch", ["epochs", "epoch", "num_epochs", "n_epochs"]),
    ("Step 수", ["n_steps", "steps", "max_steps", "num_steps"]),
    ("Optimizer", ["optimizer"]),
]
_SYSTEM_PARAM_KEYS = {"device", "gpu", "gpus", "num_gpus", "cuda", "hostname", "host", "num_workers"}

CHAT_EXAMPLE_QUESTIONS = [
    "실험 목록 보여줘",
    "최근 실험 알려줘",
    "이 실험의 metric key 목록 알려줘",
    "최근 30일 running 상태 run 보여줘",
    "runs top k=10 보여줘",
    "보고서 기반 요약 해줘"
]


def _init_session_defaults() -> None:
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _safe_request(method: str, endpoint: str, *, timeout: int, **kwargs: Any) -> requests.Response | None:
    try:
        return requests.request(method, f"{API_BASE_URL}{endpoint}", timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        st.error(f"백엔드 요청 실패: {exc}")
        return None


def _build_run_summaries_from_raw_runs(raw_runs: list[dict[str, Any]], metric_key: str) -> list[dict[str, Any]]:
    metric_col = metric_key if metric_key.startswith("metrics.") else f"metrics.{metric_key}"
    out: list[dict[str, Any]] = []
    for r in raw_runs:
        score = r.get(metric_col)
        try:
            metric_score = float(score)
        except (TypeError, ValueError):
            continue

        params = {
            key.removeprefix("params."): str(value)
            for key, value in r.items()
            if key.startswith("params.") and value is not None
        }
        metrics = {}
        for key, value in r.items():
            if key.startswith("metrics.") and value is not None:
                try:
                    metrics[key.removeprefix("metrics.")] = float(value)
                except (TypeError, ValueError):
                    continue

        out.append(
            {
                "run_id": str(r.get("run_id", "")),
                "run_name": str(r.get("tags.mlflow.runName", "")),
                "status": str(r.get("status", "")),
                "metric_score": metric_score,
                "metric_key": metric_col,
                "start_time": r.get("start_time"),
                "params": params,
                "metrics": metrics,
            }
        )
    return out



def _refresh_experiment_names() -> None:
    response = _safe_request("GET", "/mlflow/names", timeout=5)
    if response is None:
        st.session_state["experiment_names"] = []
        st.session_state["names_fetch_error"] = "실험 목록 요청에 실패했습니다."
        return
    if response.status_code == 200:
        st.session_state["experiment_names"] = response.json().get("experiment_names", [])
        st.session_state["names_fetch_error"] = ""
        return
    st.session_state["experiment_names"] = []
    st.session_state["names_fetch_error"] = f"실험 목록 조회 실패 ({response.status_code})"


def _fetch_metric_keys(experiment_name: str) -> list[str]:
    response = _safe_request(
        "GET",
        "/mlflow/metric-keys",
        params={"experiment_name": experiment_name},
        timeout=10,
    )
    if response is None:
        return []
    if response.status_code != 200:
        st.error(f"Metric key 목록 조회 실패 ({response.status_code})")
        return []
    return response.json().get("metric_keys", [])


def _fetch_runs_for_view(exp: str, metric: str, selected_top_k: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    direction = infer_metric_direction(metric)
    ascending = direction == "minimize"
    if selected_top_k == DEFAULT_TOP_K_LABEL:
        response = _safe_request(
            "GET",
            "/mlflow/runs",
            params={"experiment_name": exp},
            timeout=30,
        )
        if response is None:
            return [], {}
        if response.status_code != 200:
            st.error(f"Runs 조회 실패 ({response.status_code})")
            st.code(response.text)
            return [], {}
        raw_runs = response.json().get("runs", [])
        runs_data = _build_run_summaries_from_raw_runs(raw_runs, metric)
        runs_data = sorted(runs_data, key=lambda x: x.get("metric_score", 0.0), reverse=not ascending)
        return runs_data, {
            "experiment_name": exp,
            "metric_key": metric,
            "metric_direction": direction,
            "top_k": DEFAULT_TOP_K_LABEL,
        }

    response = _safe_request(
        "GET",
        "/mlflow/top-runs",
        params={
            "experiment_name": exp,
            "metric_key": metric,
            "top_k": int(selected_top_k),
            "ascending": ascending,
        },
        timeout=30,
    )
    if response is None:
        return [], {}
    if response.status_code != 200:
        st.error(f"Top runs 조회 실패 ({response.status_code})")
        st.code(response.text)
        return [], {}
    raw_runs = response.json().get("top_runs", [])

    return raw_runs, {
        "experiment_name": exp,
        "metric_key": metric,
        "metric_direction": direction,
        "top_k": int(selected_top_k),
    }


def _apply_report_from_body(
    body: dict[str, Any],
    *,
    experiment_name: str = "",
    metric_key: str = "",
    source: str = "generated",
) -> None:
    st.session_state["generated_report_text"] = body.get("report", "")
    st.session_state["generated_report_title"] = body.get("report_title", "")
    st.session_state["generated_report_meta"] = {
        "id": str(body.get("id", "")),
        "experiment_name": experiment_name or body.get("experiment_name", ""),
        "metric_key": metric_key or body.get("metric_key", ""),
        "model": body.get("model", "-"),
        "metadata": body.get("metadata", {}),
        "created_at": body.get("created_at", ""),
        "updated_at": body.get("updated_at", "") or body.get("created_at", ""),
        "source": source,
    }
    # widget key는 인스턴스화 이후에 수정하면 안 됨 → 다음 런에서 정리
    st.session_state["report_edit_mode"] = False
    st.session_state["_pending_clear_report_edit_drafts"] = True


# 보고서 목록 조회
def _fetch_saved_reports(
    experiment_name: str | None = None,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": limit}
    if experiment_name:
        params["experiment_name"] = experiment_name

    response = _safe_request("GET", "/report/", params=params, timeout=30)
    if response is None:
        return []
    if response.status_code != 200:
        st.error(f"저장된 보고서 목록 조회 실패 ({response.status_code})")
        st.code(response.text)
        return []
    return response.json().get("items", [])


def _fetch_saved_report(report_id: str) -> dict[str, Any] | None:
    response = _safe_request("GET", f"/report/{report_id}", timeout=30)
    if response is None:
        return None
    if response.status_code == 404:
        st.error("선택한 보고서를 찾을 수 없습니다.")
        return None
    if response.status_code != 200:
        st.error(f"저장된 보고서 조회 실패 ({response.status_code})")
        st.code(response.text)
        return None
    return response.json()


def _delete_saved_report(report_id: str) -> bool:
    response = _safe_request("DELETE", f"/report/{report_id}", timeout=30)
    if response is None:
        return False
    if response.status_code == 404:
        st.error("삭제할 보고서를 찾을 수 없습니다.")
        return False
    if response.status_code not in (200, 204):
        st.error(f"보고서 삭제 실패 ({response.status_code})")
        st.code(response.text)
        return False
    return True


def _update_saved_report(
    report_id: str,
    *,
    report: str,
    report_title: str | None = None,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {"report": report}
    if report_title is not None:
        payload["report_title"] = report_title
    response = _safe_request("PATCH", f"/report/{report_id}", json=payload, timeout=30)
    if response is None:
        return None
    if response.status_code == 404:
        st.error("수정할 보고서를 찾을 수 없습니다.")
        return None
    if response.status_code != 200:
        st.error(f"보고서 수정 실패 ({response.status_code})")
        st.code(response.text)
        return None
    return response.json()


def _clear_main_report_if_matching(report_id: str) -> None:
    if str((st.session_state.get("chat_report") or {}).get("id") or "") == str(report_id):
        st.session_state["chat_report"] = {}
    meta = st.session_state.get("generated_report_meta") or {}
    if str(meta.get("id") or "") == str(report_id):
        st.session_state["generated_report_text"] = ""
        st.session_state["generated_report_title"] = ""
        st.session_state["generated_report_meta"] = {}
        st.session_state["report_edit_mode"] = False
        st.session_state["_pending_clear_report_edit_drafts"] = True


REPORT_SCOPE_CURRENT = "현재 실험"
REPORT_SCOPE_ALL = "전체 실험"
REPORT_MAX_RUNS = 3  # 보고서 API가 비교하는 최대 run 수


def _metric_short(metric_key: Any) -> str:
    return str(metric_key or "-").removeprefix("metrics.")


def _report_top_k(selected_top_k: Any) -> int:
    if selected_top_k == DEFAULT_TOP_K_LABEL:
        return REPORT_MAX_RUNS
    return min(int(selected_top_k), REPORT_MAX_RUNS)


def _format_short_datetime(value: object) -> str:
    """목록용 짧은 시각: '11월 11일 10:20' (올해가 아니면 연도 포함)."""
    full = _format_ui_datetime(value)
    try:
        dt = datetime.strptime(full, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return full
    label = f"{dt.month}월 {dt.day}일 {dt:%H:%M}"
    if dt.year != datetime.now(_DISPLAY_TZ).year:
        label = f"{dt.year}년 {label}"
    return label


def _is_default_report_title(item: dict[str, Any]) -> bool:
    title = str(item.get("report_title") or "").strip()
    return not title or title == f"{item.get('experiment_name') or ''} 실험 분석 보고서"


def _report_short_title(item: dict[str, Any]) -> str:
    """자동 생성 제목('<실험명> 실험 분석 보고서', backend report_service._report_title)은
    실험명이 반복되므로 '상위 N개 Run 분석'으로 줄이고, 직접 수정한 제목은 그대로 쓴다."""
    if _is_default_report_title(item):
        n = item.get("run_count")
        return f"상위 {n}개 Run 분석" if n else "Run 분석 보고서"
    return str(item.get("report_title")).strip()


def _report_list_title(item: dict[str, Any], *, show_experiment: bool) -> str:
    """목록 첫 줄: 전체 실험 보기에서는 실험명, 현재 실험 보기에서는 보고서 제목."""
    if show_experiment:
        return str(item.get("experiment_name") or "-")
    return _report_short_title(item)


def _report_list_caption(item: dict[str, Any], *, show_experiment: bool) -> str:
    """목록 둘째 줄: (보고서 제목 ·) 생성일시 · 기준 지표 (· Run 수)."""
    parts: list[str] = []
    if show_experiment:
        parts.append(_report_short_title(item))
    parts.append(_format_short_datetime(item.get("created_at")))
    parts.append(_metric_short(item.get("metric_key")))
    # 자동 제목에는 이미 Run 수가 들어 있으므로 직접 수정한 제목일 때만 표시
    if not _is_default_report_title(item) and item.get("run_count"):
        parts.append(f"Run {item['run_count']}개")
    return " · ".join(parts)


_ZERO_WIDTH_CHARS = ("\u200b", "\u200c", "\u200d", "\u2060")


def _invisible_tag(report_id: str) -> str:
    """보고서 id로 만든 보이지 않는 꼬리표.
    st.radio는 선택값을 '표시 문자열'로 저장하므로 제목이 같은 항목이 있으면 구별하지 못한다.
    표시 문자열 끝에 붙여 화면은 그대로 두고 항목마다 고유하게 만든다."""
    digits = re.sub(r"[^0-9a-f]", "", report_id.lower())[:12] or "0"
    return "".join(
        _ZERO_WIDTH_CHARS[int(c, 16) // 4] + _ZERO_WIDTH_CHARS[int(c, 16) % 4] for c in digits
    )


def _on_report_list_select(list_key: str) -> None:
    report_id = st.session_state.get(list_key)
    if not report_id:
        return
    st.session_state.pop("_confirm_delete_report_id", None)
    body = _fetch_saved_report(str(report_id))
    if body is not None:
        _apply_report_from_body(body, source="loaded")
        st.session_state["_open_report_dialog"] = True


def _render_saved_reports_panel(*, experiment_name: str, panel_key: str) -> None:
    title_col, refresh_col = st.columns([5, 1], vertical_alignment="center")
    with title_col:
        st.markdown('<div class="sidebar-group-title">저장된 보고서</div>', unsafe_allow_html=True)
    with refresh_col:
        refresh_clicked = st.button(
            "",
            icon=":material/refresh:",
            key=f"{panel_key}_refresh_reports",
            help="목록 새로고침",
            type="tertiary",
        )

    scope = st.segmented_control(
        "보고서 범위",
        [REPORT_SCOPE_CURRENT, REPORT_SCOPE_ALL],
        default=REPORT_SCOPE_CURRENT,
        key=f"{panel_key}_report_scope",
        label_visibility="collapsed",
    ) or REPORT_SCOPE_CURRENT

    has_experiment = bool(experiment_name) and experiment_name != DIRECT_INPUT_LABEL
    if scope == REPORT_SCOPE_CURRENT and not has_experiment:
        st.caption("실험을 선택하면 해당 실험의 보고서가 표시됩니다.")
        return
    exp_filter = experiment_name.strip() if scope == REPORT_SCOPE_CURRENT else None

    cache_key = f"{panel_key}_saved_reports"
    cache_filter_key = f"{panel_key}_saved_reports_filter"
    if (
        refresh_clicked
        or cache_key not in st.session_state
        or st.session_state.get(cache_filter_key) != exp_filter
    ):
        st.session_state[cache_key] = _fetch_saved_reports(exp_filter)
        st.session_state[cache_filter_key] = exp_filter

    if exp_filter:
        # 현재 실험 보기: 실험명은 목록 위에 한 번만
        st.markdown(
            f'<div class="report-list-scope">{html.escape(exp_filter)}</div>',
            unsafe_allow_html=True,
        )

    items: list[dict[str, Any]] = [
        item for item in st.session_state.get(cache_key, []) if item.get("id")
    ]
    if not items:
        st.caption("저장된 보고서가 없습니다.")
        return

    by_id = {str(item["id"]): item for item in items}
    ids = list(by_id)
    list_key = f"{panel_key}_report_list"
    # 선택 표시는 팝업에 열린 보고서를 기준으로 맞춘다 (팝업을 닫으면 해제 → 같은 항목 다시 클릭 가능).
    # 매 실행 덮어쓰면 사용자의 다음 클릭이 무시되므로, 값이 다를 때만 맞춘다.
    open_id = str((st.session_state.get("generated_report_meta") or {}).get("id") or "")
    desired = open_id if open_id in by_id else None
    # 키가 없을 때도 None으로 맞춰야 첫 항목이 열리지 않은 채 선택돼 보이지 않는다
    if list_key not in st.session_state or st.session_state[list_key] != desired:
        st.session_state[list_key] = desired

    show_experiment = exp_filter is None
    st.radio(
        "저장된 보고서 목록",
        ids,
        key=list_key,
        format_func=lambda rid: _report_list_title(by_id[rid], show_experiment=show_experiment)
        + _invisible_tag(rid),
        captions=[_report_list_caption(by_id[rid], show_experiment=show_experiment) for rid in ids],
        label_visibility="collapsed",
        on_change=_on_report_list_select,
        args=(list_key,),
    )


def _generate_report(exp: str, metric: str, selected_top_k: Any) -> None:
    report_top_k = _report_top_k(selected_top_k)

    direction = infer_metric_direction(metric)
    runs_response = _safe_request(
        "GET",
        "/mlflow/top-runs",
        params={
            "experiment_name": exp,
            "metric_key": metric,
            "top_k": report_top_k,
            "ascending": direction == "minimize",
        },
        timeout=30,
    )
    if runs_response is None:
        return
    if runs_response.status_code != 200:
        st.error(f"Top runs 조회 실패 ({runs_response.status_code})")
        st.code(runs_response.text)
        return

    top_runs = runs_response.json().get("top_runs", [])
    if not top_runs:
        st.warning("보고서를 만들 run 데이터가 없습니다.")
        return

    report_response = _safe_request(
        "POST",
        "/report/",
        json={
            "experiment_name": exp,
            "metric_key": metric,
            "metric_direction": direction,
            "top_runs": top_runs,
        },
        timeout=120,
    )
    if report_response is None:
        return
    if report_response.status_code != 200:
        st.error(f"보고서 생성 실패 ({report_response.status_code})")
        st.code(report_response.text)
        return

    body = report_response.json()
    _apply_report_from_body(body, experiment_name=exp, metric_key=metric, source="generated")
    # 새 보고서가 사이드바 목록에 보이도록 목록을 다시 불러오고, 팝업으로 연다
    st.session_state.pop("sidebar_saved_reports", None)
    st.session_state["_open_report_dialog"] = True
    st.rerun()


def _format_route_caption(route: Any) -> str | None:
    """metadata route(파서와 답변까지 거쳐 간 함수)를 한 줄로 만든다."""
    if not isinstance(route, dict):
        return None
    parser = route.get("parser") or "-"
    path = route.get("path") or [route.get("handler") or "-"]
    return f"route: {parser} ⇒ " + " → ".join(str(step) for step in path)


def _format_tool_calls_caption(tool_calls: Any) -> str | None:
    """metadata tool_calls를 route 캡션과 같은 한 줄로 만든다."""
    if tool_calls is None or tool_calls == "" or tool_calls == []:
        return None
    items = tool_calls if isinstance(tool_calls, list) else [tool_calls]
    labels: list[str] = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("tool") or item.get("name")
            if name:
                labels.append(str(name))
        elif item is not None and str(item).strip():
            labels.append(str(item).strip())
    if not labels:
        return None
    return "tool calls: " + ", ".join(labels)


def _render_tool_calls_caption(tool_calls: Any) -> None:
    caption = _format_tool_calls_caption(tool_calls)
    if caption:
        st.caption(caption)


def _render_route_caption(route: Any) -> None:
    caption = _format_route_caption(route)
    if caption:
        st.caption(caption)


def _render_chat_bubble(role: str, content: str) -> None:
    if role == "assistant":
        st.markdown(content)
    else:
        st.write(content)


def _build_chat_history() -> list[dict[str, str]]:
    """현재 질문 제외·인사말 제외한 직전 메시지 중 최근 N개."""
    prior = st.session_state["chat_messages"][:-1]
    history: list[dict[str, str]] = []
    for msg in prior:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant" and content == CHAT_WELCOME_MESSAGE:
            continue
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": content})
    return history[-CHAT_HISTORY_LIMIT:]


def _reset_chat_to_new() -> None:
    st.session_state["chat_session_id"] = None
    st.session_state["chat_resolved_context"] = {}
    st.session_state["chat_messages"] = [
        {"role": "assistant", "content": CHAT_WELCOME_MESSAGE}
    ]


def _context_from_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    """assistant metadata에서 UI 표시용 후처리 문맥만 추출."""
    if not isinstance(meta, dict):
        return {}
    ctx: dict[str, Any] = {}
    if meta.get("experiment_name"):
        ctx["experiment_name"] = str(meta["experiment_name"])
    if meta.get("metric_key"):
        ctx["metric_key"] = str(meta["metric_key"])
    if meta.get("top_k") is not None and meta.get("top_k") != "":
        ctx["top_k"] = meta["top_k"]
    if meta.get("ascending") is not None:
        ctx["ascending"] = bool(meta["ascending"])
    return ctx


def _set_chat_resolved_context_from_items(items: list[dict[str, Any]]) -> None:
    """메시지 목록에서 가장 최근 assistant metadata 문맥을 복원."""
    for item in reversed(items):
        if item.get("role") != "assistant":
            continue
        ctx = _context_from_metadata(item.get("metadata") or {})
        if ctx:
            st.session_state["chat_resolved_context"] = ctx
            return
    st.session_state["chat_resolved_context"] = {}


def _chat_reference_html(*, sidebar_experiment: str, sidebar_metric: str) -> str:
    """채팅 본문 상단의 참조 문맥: 실험 · 지표 (· top_k) · 보고서."""
    resolved = st.session_state.get("chat_resolved_context") or {}
    chat_report = st.session_state.get("chat_report") or {}
    exp = resolved.get("experiment_name") or chat_report.get("experiment_name") or (
        sidebar_experiment if sidebar_experiment and sidebar_experiment != DIRECT_INPUT_LABEL else None
    )
    metric = resolved.get("metric_key") or chat_report.get("metric_key") or (
        sidebar_metric if sidebar_metric and sidebar_metric != NO_SELECTION_LABEL else None
    )
    empty = '<span class="exp-empty">-</span>'

    def field(label: str, value: Any) -> str:
        shown = html.escape(str(value)) if value not in (None, "") else empty
        return f'<span class="exp-field-label">{label}</span> {shown}'

    parts = [field("실험", exp), field("지표", _metric_short(metric) if metric else None)]
    if resolved.get("top_k") is not None:
        parts.append(field("Top K", resolved.get("top_k")))
    parts.append(field("보고서", chat_report.get("short_title")))
    return '<div class="chat-report-ref">' + " &nbsp;·&nbsp; ".join(parts) + "</div>"


def _rerun_dialog() -> None:
    """팝업(fragment)만 다시 그린다.
    scope="fragment"는 팝업 안 조작으로 인한 재실행 중에만 허용되므로,
    앱 전체 실행 중이면(이미 최신 상태로 그려지는 중) 그냥 넘어간다."""
    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        pass


def _on_new_chat_click() -> None:
    """팝업 유지: st.rerun() 없이 세션만 초기화."""
    _reset_chat_to_new()


def _delete_chat_session(session_id: str) -> bool:
    response = _safe_request("DELETE", f"/chat/sessions/{session_id}", timeout=10)
    if response is None:
        return False
    if response.status_code not in (200, 204):
        st.error(f"대화 삭제 실패 ({response.status_code})")
        st.code(response.text)
        return False
    return True


def _on_chat_session_pick(session_id: str) -> None:
    if session_id != str(st.session_state.get("chat_session_id") or ""):
        _load_chat_session(session_id)


def _on_chat_session_delete(session_id: str) -> None:
    st.session_state[f"chat_menu_{session_id}"] = False  # ⋯ 메뉴 닫기
    if _delete_chat_session(session_id):
        if session_id == str(st.session_state.get("chat_session_id") or ""):
            _reset_chat_to_new()
        st.session_state["_chat_delete_notice"] = "대화를 삭제했습니다."


def _attach_example_datalist() -> None:
    """질문 입력창에 예시 질문 자동완성 목록(<datalist>)을 붙인다.
    Streamlit text_input에는 추천 목록 기능이 없어 브라우저 기본 datalist를 사용한다.
    (예시 문구는 코드 상수라 외부 입력이 섞이지 않는다)"""
    # 주의: <script> 안에 '<option>' 같은 HTML 태그 문자열이 있으면 Streamlit HTML 정리 과정에서
    # 스크립트 전체가 실행되지 않는다 → 예시는 JSON 배열로 넘기고 option은 JS로 만든다
    questions_json = json.dumps(CHAT_EXAMPLE_QUESTIONS, ensure_ascii=False).replace("</", "<\\/")
    st.html(
        f"""
        <script>
        (function () {{
          const LIST_ID = "chat-example-questions";
          let list = document.getElementById(LIST_ID);
          if (!list) {{
            list = document.createElement("datalist");
            list.id = LIST_ID;
            document.body.appendChild(list);
          }}
          const questions = {questions_json};
          list.replaceChildren(...questions.map((q) => {{
            const opt = document.createElement("option");
            opt.value = q;
            return opt;
          }}));
          const attach = () => {{
            document.querySelectorAll(".st-key-chat_prompt_text input:not([list])")
              .forEach((el) => el.setAttribute("list", LIST_ID));
          }};
          const autoCollapse = () => {{
            // 좁은 화면: 대화 목록을 기본으로 접는다 (브라우저 세션당 한 번만, 이후엔 사용자 선택 유지)
            if (window.innerWidth >= 768) return;
            let done = false;
            try {{ done = sessionStorage.getItem("chatListAutoCollapsed") === "1"; }} catch (e) {{}}
            if (done) return;
            const btn = document.querySelector(".st-key-chat_list_fold_btn button");
            if (!btn) return;
            try {{ sessionStorage.setItem("chatListAutoCollapsed", "1"); }} catch (e) {{}}
            btn.click();
          }};
          const onChange = () => {{ attach(); autoCollapse(); }};
          onChange();
          // 팝업이 다시 그려져 입력창이 새로 생겨도 다시 붙인다
          if (!window.__chatExampleObserver) {{
            window.__chatExampleObserver = new MutationObserver(onChange);
            window.__chatExampleObserver.observe(document.body, {{ childList: true, subtree: true }});
          }}
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def _toggle_chat_list() -> None:
    st.session_state["chat_list_collapsed"] = not st.session_state.get("chat_list_collapsed", False)


def _fetch_chat_sessions(limit: int = 20) -> list[dict[str, Any]]:
    response = _safe_request("GET", "/chat/sessions", params={"limit": limit}, timeout=10)
    if response is None or response.status_code != 200:
        return []
    return list(response.json().get("items") or [])


def _load_chat_session(session_id: str) -> bool:
    response = _safe_request(
        "GET",
        f"/chat/sessions/{session_id}/messages",
        timeout=10,
    )
    if response is None:
        return False
    if response.status_code != 200:
        st.error(f"대화 이력 조회 실패 ({response.status_code})")
        st.code(response.text)
        return False
    items = response.json().get("items") or []
    messages: list[dict[str, Any]] = []
    for item in items:
        role = item.get("role", "")
        content = item.get("content", "")
        if role in ("user", "assistant") and content:
            message: dict[str, Any] = {"role": role, "content": content}
            item_meta = item.get("metadata") or {}
            if item_meta.get("tool_calls"):
                message["tool_calls"] = item_meta["tool_calls"]
            if item_meta.get("route"):
                message["route"] = item_meta["route"]
            messages.append(message)
    if not messages:
        messages = [{"role": "assistant", "content": CHAT_WELCOME_MESSAGE}]
    st.session_state["chat_session_id"] = session_id
    st.session_state["chat_messages"] = messages
    _set_chat_resolved_context_from_items(items)
    return True


def _append_and_answer_chat(
    user_prompt: str,
    current_experiment: str,
    current_metric: str,
) -> None:
    st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    chat_payload: dict[str, Any] = {
        "message": user_prompt,
        "history": _build_chat_history(),
    }
    session_id = st.session_state.get("chat_session_id")
    if session_id:
        chat_payload["session_id"] = session_id

    # 챗봇은 '이 보고서로 질문하기'로 연결한 보고서만 참조한다
    report_meta = st.session_state.get("chat_report") or {}
    report_text = report_meta.get("text", "")
    if report_text:
        chat_payload["report_text"] = report_text
    if report_meta.get("experiment_name"):
        chat_payload["experiment_name"] = report_meta.get("experiment_name")
    elif current_experiment and current_experiment != DIRECT_INPUT_LABEL:
        chat_payload["experiment_name"] = current_experiment
    if report_meta.get("metric_key"):
        chat_payload["metric_key"] = report_meta.get("metric_key")
    elif current_metric and current_metric != NO_SELECTION_LABEL:
        chat_payload["metric_key"] = current_metric

    chat_response = _safe_request("POST", "/chat/ask", json=chat_payload, timeout=120)
    tool_calls = None
    route = None
    if chat_response is None:
        answer = "챗봇 요청 실패: 백엔드 요청에 실패했습니다."
    elif chat_response.status_code == 200:
        data = chat_response.json()
        answer = data.get("answer", "응답이 비어 있습니다.")
        meta = data.get("metadata") or {}
        tool_calls = meta.get("tool_calls")
        route = meta.get("route")
        returned_session_id = data.get("session_id")
        if returned_session_id:
            st.session_state["chat_session_id"] = returned_session_id
        resolved = _context_from_metadata(meta)
        if resolved:
            prev = dict(st.session_state.get("chat_resolved_context") or {})
            prev.update(resolved)
            st.session_state["chat_resolved_context"] = prev
    else:
        answer = f"챗봇 응답 실패 ({chat_response.status_code})"

    assistant_message: dict[str, Any] = {"role": "assistant", "content": answer}
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls
    if route:
        assistant_message["route"] = route
    st.session_state["chat_messages"].append(assistant_message)
    with st.chat_message("assistant"):
        _render_route_caption(route)
        _render_tool_calls_caption(tool_calls)
        resolved = st.session_state.get("chat_resolved_context") or {}
        if resolved:
            bits: list[str] = []
            if resolved.get("experiment_name"):
                bits.append(f"experiment `{resolved['experiment_name']}`")
            if resolved.get("metric_key"):
                bits.append(f"metric `{resolved['metric_key']}`")
            if resolved.get("top_k") is not None:
                bits.append(f"top_k `{resolved['top_k']}`")
            if bits:
                st.caption("후처리 문맥: " + " · ".join(bits))
        _render_chat_bubble("assistant", answer)


def _fetch_experiment_info(exp_name: str) -> dict[str, Any] | None:
    response = _safe_request(
        "GET",
        "/mlflow/experiment",
        params={"experiment_name": exp_name},
        timeout=10,
    )
    if response is None:
        return None
    if response.status_code != 200:
        st.error(f"실험 정보 조회 실패 ({response.status_code})")
        st.code(response.text)
        return None
    return response.json().get("experiment", {}) or {}


def _field_html(label: str, value_html: str) -> str:
    return (
        f'<div class="exp-field"><div class="exp-field-label">{html.escape(label)}</div>'
        f'<div class="exp-field-value">{value_html}</div></div>'
    )


def _render_experiment_summary(data: dict[str, Any]) -> None:
    name = str(data.get("experiment_name") or "-")
    exp_id = str(data.get("experiment_id") or "-")
    workspace = str(data.get("workspace") or "-")

    st.markdown(
        f'''<div class="exp-header">
            <span class="exp-title">{html.escape(name)}</span>
        </div>
        <div class="exp-subline">실험 ID {html.escape(exp_id)} · 워크스페이스 {html.escape(workspace)}</div>''',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.markdown(
        _field_html("생성일", html.escape(_format_ui_datetime(data.get("creation_time")))),
        unsafe_allow_html=True,
    )
    c2.markdown(
        _field_html("최근 수정일", html.escape(_format_ui_datetime(data.get("last_update_time")))),
        unsafe_allow_html=True,
    )

    if exp_id != "-":
        mlflow_url = f"{_mlflow_ui_base()}/#/experiments/{quote(exp_id, safe='')}"
        st.markdown('<div class="exp-field-label">MLflow</div>', unsafe_allow_html=True)
        st.link_button("MLflow에서 실험 보기", mlflow_url)

    tags = data.get("tags") or {}
    if tags:
        chips = "".join(
            f'<span class="exp-tag">{html.escape(str(k))}: {html.escape(str(v))}</span>'
            for k, v in tags.items()
        )
    else:
        chips = '<span class="exp-empty">없음</span>'
    st.markdown(_field_html("태그", chips), unsafe_allow_html=True)

    trace = data.get("trace_location")
    trace_html = html.escape(str(trace)) if trace else '<span class="exp-empty">미설정</span>'
    st.markdown(_field_html("트레이스 경로", trace_html), unsafe_allow_html=True)


def _status_label(status: Any) -> str:
    raw = str(status or "").upper()
    return _RUN_STATUS_LABELS.get(raw, raw or "-")


def _split_params(params: dict[str, str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """params -> (학습 설정[(label, value)], 시스템 정보[(key, value)])."""
    remaining = dict(params)
    training: list[tuple[str, str]] = []
    for label, candidates in _TRAINING_PARAM_SPECS:
        for key in candidates:
            if key in remaining:
                training.append((f"{label} ({key})", remaining.pop(key)))
                break
    system: list[tuple[str, str]] = []
    for key in sorted(remaining):
        if key.lower() in _SYSTEM_PARAM_KEYS:
            system.append((key, remaining.pop(key)))
    training.extend(sorted(remaining.items()))
    return training, system


def _run_table_rows(runs: list[dict[str, Any]], score_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in runs:
        score = r.get("metric_score")
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        rows.append(
            {
                "Run 이름": str(r.get("run_name") or "-"),
                score_label: score,
                "상태": _status_label(r.get("status")),
                "실행 시각": _format_ui_datetime(r.get("start_time")),
            }
        )
    return rows


def _render_run_detail(run: dict[str, Any]) -> None:
    params = {str(k): str(v) for k, v in (run.get("params") or {}).items()}
    metrics = run.get("metrics") or {}
    training, system_params = _split_params(params)

    st.markdown(
        f'<div class="run-detail-title">상세 보기 · {html.escape(str(run.get("run_name") or "-"))}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="exp-field-label">학습 설정 ({len(training)})</div>', unsafe_allow_html=True
    )
    if training:
        query = st.text_input(
            "학습 설정 검색",
            key=f"run_param_search_{run.get('run_id', '')}",
            placeholder="항목명 또는 값으로 검색 (예: learning_rate, batch)",
            label_visibility="collapsed",
        ).strip().lower()
        rows = [
            {"항목": label, "값": value}
            for label, value in training
            if not query or query in label.lower() or query in value.lower()
        ]
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.caption(f"'{query}'에 해당하는 학습 설정이 없습니다.")
    else:
        st.caption("기록된 학습 설정이 없습니다.")

    general_metrics = {k: v for k, v in metrics.items() if not str(k).startswith("system/")}
    system_metrics = {k: v for k, v in metrics.items() if str(k).startswith("system/")}

    with st.expander(f"전체 지표 ({len(general_metrics)})", expanded=False):
        if general_metrics:
            st.dataframe(
                [{"지표": k, "값": v} for k, v in sorted(general_metrics.items())],
                hide_index=True,
                use_container_width=True,
                column_config={"값": st.column_config.NumberColumn(format="%.4f")},
            )
        else:
            st.caption("기록된 지표가 없습니다.")

    with st.expander("시스템 정보", expanded=False):
        sys_rows = [{"항목": "run_id", "값": str(run.get("run_id") or "-")}]
        sys_rows += [{"항목": k, "값": v} for k, v in system_params]
        sys_rows += [{"항목": k, "값": f"{float(v):.4f}"} for k, v in sorted(system_metrics.items())]
        st.dataframe(sys_rows, hide_index=True, use_container_width=True)


def _render_runs_section(runs_data: list[dict[str, Any]], runs_meta: dict[str, Any]) -> None:
    st.subheader("Run 목록")
    if not runs_data:
        st.caption("Metric key를 선택한 뒤 **실험 정보 조회**를 누르면 run 목록이 표시됩니다.")
        return

    metric_key = str(runs_meta.get("metric_key") or "Score")
    score_label = metric_key.removeprefix("metrics.")
    direction = runs_meta.get("metric_direction")
    direction_label = {"maximize": "높을수록 좋음", "minimize": "낮을수록 좋음"}.get(direction, "방향 미정")
    st.caption(f"`{score_label}` 기준 정렬 ({direction_label}) · {len(runs_data)}개 · K={runs_meta.get('top_k', '-')}")

    table = pd.DataFrame(_run_table_rows(runs_data, score_label))
    styled = table.style.map(
        lambda v: f"color: {_RUN_STATUS_COLORS[v]}; font-weight: 600" if v in _RUN_STATUS_COLORS else "",
        subset=["상태"],
    )
    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="runs_table",
        column_config={
            score_label: st.column_config.NumberColumn(score_label, format="%.2f"),
        },
    )
    selected_rows = list(getattr(getattr(event, "selection", None), "rows", []) or [])
    if selected_rows and 0 <= selected_rows[0] < len(runs_data):
        _render_run_detail(runs_data[selected_rows[0]])
    else:
        st.caption("행을 선택하면 학습 설정과 지표 상세가 표시됩니다.")


def _render_chat_session_list(sessions: list[dict[str, Any]], current_sid: str) -> None:
    """왼쪽: 새 채팅 · 접기 / 최근 대화 목록(이 영역만 스크롤)."""
    new_col, fold_col = st.columns([5, 1], vertical_alignment="center")
    new_col.button(
        "새 채팅",
        icon=":material/add:",
        key="chat_new_btn",
        use_container_width=True,
        on_click=_on_new_chat_click,
    )
    fold_col.button(
        "",
        icon=":material/left_panel_close:",
        key="chat_list_fold_btn",
        type="tertiary",
        help="대화 목록 접기",
        on_click=_toggle_chat_list,
    )

    notice = st.session_state.pop("_chat_delete_notice", None)
    if notice:
        st.caption(notice)

    with st.container(height=520, border=False, key="chat_session_list"):
        if not sessions:
            st.caption("저장된 대화가 없습니다.")
            return
        for item in sessions:
            sid = str(item.get("session_id") or "")
            if not sid:
                continue
            title = (item.get("title") or "제목 없음").strip() or "제목 없음"
            if sid == current_sid:
                # 선택한 대화 배경 강조
                st.markdown(
                    f"<style>.st-key-chat_row_{sid} {{ background: rgba(255, 75, 75, 0.12); }}</style>",
                    unsafe_allow_html=True,
                )
            with st.container(key=f"chat_row_{sid}"):
                pick_col, menu_col = st.columns([6, 1], vertical_alignment="center", gap="small")
                with pick_col:
                    st.button(
                        title,
                        key=f"chat_pick_{sid}",
                        type="tertiary",
                        help=title,
                        on_click=_on_chat_session_pick,
                        args=(sid,),
                    )
                    st.caption(_format_short_datetime(item.get("updated_at") or item.get("created_at")))
                # key + on_change로 열림 상태를 서버에서 관리 → 삭제 후 코드로 닫을 수 있다
                with menu_col.popover(
                    # help 말풍선은 작은 삭제 메뉴를 가려서 뺌
                    "", icon=":material/more_horiz:", type="tertiary",
                    key=f"chat_menu_{sid}", on_change="rerun",
                ):
                    st.button(
                        "삭제",
                        key=f"chat_del_{sid}",
                        icon=":material/delete:",
                        type="tertiary",
                        on_click=_on_chat_session_delete,
                        args=(sid,),
                    )


def _render_chat_body(sessions: list[dict[str, Any]], current_sid: str, *, list_collapsed: bool) -> None:
    """오른쪽: 대화 제목 / 참조 문맥 / 대화 내용(스크롤) / 예시 질문 · 입력창."""
    current_experiment = (st.session_state.get("selected_experiment") or "").strip()
    current_metric = (st.session_state.get("selected_metric") or "").strip()

    title = "새 채팅"
    for item in sessions:
        if str(item.get("session_id") or "") == current_sid:
            title = (item.get("title") or "제목 없음").strip() or "제목 없음"
            break

    if list_collapsed:
        open_col, title_col = st.columns([1, 14], vertical_alignment="center")
        open_col.button(
            "",
            icon=":material/left_panel_open:",
            key="chat_list_open_btn",
            type="tertiary",
            help="대화 목록 열기",
            on_click=_toggle_chat_list,
        )
    else:
        title_col = st.container()
    title_col.markdown(f'<div class="chat-title">{html.escape(title)}</div>', unsafe_allow_html=True)

    ref_col, unlink_col = st.columns([6, 1], vertical_alignment="center")
    ref_col.markdown(
        _chat_reference_html(sidebar_experiment=current_experiment, sidebar_metric=current_metric),
        unsafe_allow_html=True,
    )
    if st.session_state.get("chat_report"):
        unlink_col.button(
            "보고서 연결 해제",
            key="chat_report_unlink",
            type="tertiary",
            on_click=_unlink_chat_report,
        )

    messages_box = st.container(height=400, border=False, key="chat_messages_box")
    with messages_box:
        for message in st.session_state["chat_messages"]:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    _render_route_caption(message.get("route"))
                    _render_tool_calls_caption(message.get("tool_calls"))
                _render_chat_bubble(message["role"], message["content"])

    # 질문 입력: 클릭하면 예시 질문이 아래 목록으로 펼쳐지고, 고르면 입력창에 채워져 수정 가능.
    # Enter 또는 전송 버튼으로 전송 (폼 제출), 전송 후 입력창은 비워진다.
    with st.form("chat_prompt_form", clear_on_submit=True, border=False, enter_to_submit=True):
        input_col, send_col = st.columns([8, 1], vertical_alignment="bottom")
        typed = input_col.text_input(
            "질문 입력",
            key="chat_prompt_text",
            placeholder="질문을 입력하거나 예시 질문을 선택하세요",
            autocomplete="off",
            label_visibility="collapsed",
        )
        sent = send_col.form_submit_button(
            "전송", icon=":material/send:", type="primary", use_container_width=True
        )
    _attach_example_datalist()
    final_prompt = (typed or "").strip() if sent else ""
    if final_prompt:
        with messages_box:
            _append_and_answer_chat(final_prompt, current_experiment, current_metric)
        # 새 대화 제목·목록(최신순)을 갱신하기 위해 팝업만 다시 그린다
        _rerun_dialog()


@st.dialog("실험 Q&A 챗봇", width="large")
def show_chat_dialog() -> None:
    # 팝업 폭을 넓히기 위한 표식 (CSS :has 선택자에서 사용)
    st.markdown('<span class="chat-dialog-marker"></span>', unsafe_allow_html=True)
    sessions = _fetch_chat_sessions(limit=30)
    current_sid = str(st.session_state.get("chat_session_id") or "")
    # 접힘 상태·현재 대화는 session_state에 있어 팝업을 닫았다 열어도 유지된다
    list_collapsed = bool(st.session_state.get("chat_list_collapsed", False))

    if list_collapsed:
        st.markdown('<span class="chat-list-collapsed-marker"></span>', unsafe_allow_html=True)
        _render_chat_body(sessions, current_sid, list_collapsed=True)
        return
    list_col, body_col = st.columns([1, 3.2], gap="medium")
    with list_col:
        _render_chat_session_list(sessions, current_sid)
    with body_col:
        _render_chat_body(sessions, current_sid, list_collapsed=False)


def _unlink_chat_report() -> None:
    st.session_state["chat_report"] = {}


def _close_report_view() -> None:
    """팝업에 열린 보고서 상태를 비운다 (사이드바 선택 표시도 해제됨)."""
    st.session_state["generated_report_text"] = ""
    st.session_state["generated_report_title"] = ""
    st.session_state["generated_report_meta"] = {}
    st.session_state["report_edit_mode"] = False
    st.session_state["_pending_clear_report_edit_drafts"] = True
    st.session_state.pop("_confirm_delete_report_id", None)


def _on_ask_with_report() -> None:
    """'이 보고서로 질문하기': 보고서를 챗봇에 연결하고, 팝업을 닫은 뒤 챗봇을 연다."""
    meta = st.session_state.get("generated_report_meta") or {}
    runs_list = (meta.get("metadata") or {}).get("runs_list") or []
    item = {
        "report_title": st.session_state.get("generated_report_title") or "",
        "experiment_name": meta.get("experiment_name") or "",
        "run_count": len(runs_list) or None,
    }
    st.session_state["chat_report"] = {
        "id": str(meta.get("id") or ""),
        "short_title": _report_short_title(item),
        "title": item["report_title"],
        "text": st.session_state.get("generated_report_text") or "",
        "experiment_name": item["experiment_name"],
        "metric_key": meta.get("metric_key") or "",
    }
    _close_report_view()
    _reset_chat_to_new()  # 기존 대화가 아니라 '새 채팅'으로 시작
    st.session_state["_open_chat_dialog"] = True


_DIRECTION_LABELS = {"maximize": "높을수록 좋음", "minimize": "낮을수록 좋음"}


def _render_report_view(*, body_height: int) -> None:
    """보고서 읽기 화면. 팝업에서 쓰고, 이후 '넓게 보기' 전용 화면에서도 재사용할 수 있게 분리."""
    report_text = st.session_state.get("generated_report_text", "")
    report_meta = st.session_state.get("generated_report_meta", {}) or {}
    report_id = str(report_meta.get("id") or "")
    report_title = st.session_state.get("generated_report_title") or "제목 없음"
    if not report_text and not report_meta:
        return  # 닫힌(비워진) 보고서는 그리지 않는다

    # edit widget key는 위젯 생성 전에만 정리 가능
    if (
        st.session_state.pop("_pending_clear_report_edit_drafts", False)
        or not st.session_state.get("report_edit_mode")
    ):
        st.session_state.pop("report_edit_draft", None)
        st.session_state.pop("report_edit_title_draft", None)

    # 팝업 폭을 넓히기 위한 표식 (CSS :has 선택자에서 사용)
    st.markdown('<span class="report-dialog-marker"></span>', unsafe_allow_html=True)

    # ── 상단: 생성일 · 실험명 / 기준 지표 · 정렬 방향 · 분석 Run 수 (제목은 팝업 제목줄에 고정)
    created_at = report_meta.get("created_at") or ""
    updated_at = report_meta.get("updated_at") or created_at
    created_label = _format_ui_datetime(created_at) if created_at else "-"
    updated_label = _format_ui_datetime(updated_at) if updated_at else "-"
    line1 = [f"생성 {html.escape(created_label)}"]
    if updated_label != created_label:
        line1.append(f"수정 {html.escape(updated_label)}")
    line1.append(html.escape(str(report_meta.get("experiment_name") or "-")))
    metadata = report_meta.get("metadata") or {}
    runs_list = metadata.get("runs_list") or []
    direction = _DIRECTION_LABELS.get(metadata.get("metric_direction"), "방향 미정")
    line2 = [
        f'<span class="exp-field-label">기준 지표</span> {html.escape(_metric_short(report_meta.get("metric_key")))}',
        f'<span class="exp-field-label">정렬</span> {direction}',
        f'<span class="exp-field-label">분석 Run</span> {len(runs_list)}개'
        + (f' <span class="exp-empty">({html.escape(", ".join(map(str, runs_list)))})</span>' if runs_list else ""),
    ]
    st.markdown(
        f'<div class="exp-subline" style="margin-bottom:4px">{" · ".join(line1)}</div>'
        f'<div class="report-criteria-line">{" &nbsp;·&nbsp; ".join(line2)}</div>',
        unsafe_allow_html=True,
    )

    source = report_meta.get("source")
    if source in ("edited", "generated"):
        st.markdown('<span class="report-notice-marker"></span>', unsafe_allow_html=True)
    if source == "edited":
        st.success("보고서 수정을 저장했습니다.")
    elif source == "generated":
        st.success("보고서를 생성했습니다.")

    if report_id and st.session_state.get("_confirm_delete_report_id") == report_id:
        st.warning("이 보고서를 삭제할까요? 삭제하면 되돌릴 수 없습니다.")
        confirm_col, cancel_col, _ = st.columns([1, 1, 4])
        if confirm_col.button("삭제", key="report_delete_confirm", type="primary", use_container_width=True):
            if _delete_saved_report(report_id):
                _clear_main_report_if_matching(report_id)
                _close_report_view()
                st.session_state.pop("sidebar_saved_reports", None)
                st.session_state["_report_toast"] = "보고서를 삭제했습니다."
                st.rerun()  # 앱 전체 재실행 → 팝업 닫힘
        if cancel_col.button("취소", key="report_delete_cancel", use_container_width=True):
            st.session_state.pop("_confirm_delete_report_id", None)
            _rerun_dialog()

    # ── 본문: 이 영역만 스크롤
    if st.session_state.get("report_edit_mode") and report_id:
        st.text_input("보고서 제목", key="report_edit_title_draft")
        st.text_area("보고서 내용", key="report_edit_draft", height=body_height - 90)
        save_col, cancel_col, _ = st.columns([1, 1, 4])
        if save_col.button("저장", key="report_edit_save", type="primary", use_container_width=True):
            draft = (st.session_state.get("report_edit_draft") or "").strip()
            title_draft = (st.session_state.get("report_edit_title_draft") or "").strip()
            if not draft:
                st.warning("보고서 내용이 비어 있습니다.")
            else:
                body = _update_saved_report(report_id, report=draft, report_title=title_draft or None)
                if body is not None:
                    _apply_report_from_body(body, source="edited")
                    st.session_state.pop("sidebar_saved_reports", None)
                    if str((st.session_state.get("chat_report") or {}).get("id") or "") == report_id:
                        st.session_state["chat_report"]["text"] = body.get("report", "")
                    # 제목이 바뀌면 팝업 제목줄도 바뀌어야 하므로 앱 전체를 다시 그리고 팝업을 다시 연다
                    st.session_state["_open_report_dialog"] = True
                    st.rerun()
        if cancel_col.button("취소", key="report_edit_cancel", use_container_width=True):
            st.session_state["report_edit_mode"] = False
            st.session_state["_pending_clear_report_edit_drafts"] = True
            _rerun_dialog()
        return

    with st.container(height=body_height, border=False, key="report_dialog_body"):
        st.markdown(report_text, unsafe_allow_html=True)

    # ── 하단: 다운로드 · 이 보고서로 질문하기 · (⋯ 수정/삭제)
    safe_name = re.sub(r'[\\/:*?"<>|\s]+', "_", report_title).strip("_") or "report"
    dl_col, ask_col, _, menu_col = st.columns([2, 3, 5, 1], vertical_alignment="center")
    dl_col.download_button(
        "다운로드",
        data=f"# {report_title}\n\n{report_text}",
        file_name=f"{safe_name}.md",
        mime="text/markdown",
        icon=":material/download:",
        key="report_download_btn",
        use_container_width=True,
    )
    if ask_col.button(
        "이 보고서로 질문하기",
        icon=":material/forum:",
        key="report_ask_btn",
        type="primary",
        use_container_width=True,
    ):
        _on_ask_with_report()
        # 팝업 안 버튼은 팝업(fragment)만 다시 그리므로, 앱 전체를 재실행해야
        # 보고서 팝업이 닫히고 하단 로직에서 챗봇 팝업이 열린다
        st.rerun(scope="app")
    if report_id:
        if st.session_state.pop("_close_report_menu", False):
            st.session_state["report_menu"] = False  # 위젯 생성 전에만 바꿀 수 있음
        with menu_col.popover(
            "", icon=":material/more_horiz:", help="더보기", key="report_menu", on_change="rerun"
        ):
            if st.button("수정", key="report_edit_btn", icon=":material/edit:", use_container_width=True):
                st.session_state["report_edit_mode"] = True
                st.session_state["report_edit_draft"] = report_text
                st.session_state["report_edit_title_draft"] = st.session_state.get("generated_report_title") or ""
                st.session_state["_close_report_menu"] = True
                _rerun_dialog()
            if st.button("삭제", key="report_delete_btn", icon=":material/delete:", use_container_width=True):
                st.session_state["_confirm_delete_report_id"] = report_id
                st.session_state["_close_report_menu"] = True
                _rerun_dialog()


def show_report_dialog() -> None:
    """보고서 팝업. 제목은 팝업 제목줄(닫기 버튼과 함께 고정)에 표시한다."""
    title = st.session_state.get("generated_report_title") or "보고서"
    st.dialog(title, width="large", on_dismiss=_close_report_view)(
        lambda: _render_report_view(body_height=560)
    )()


# 브라우저에서 열 수 없는 주소(서버 자신/도커 내부 이름)
_INTERNAL_MLFLOW_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "mlflow"}


def _fetch_tracking_uri() -> str:
    try:
        response = requests.get(f"{API_BASE_URL}/mlflow/tracking-uri", timeout=5)
    except requests.RequestException:
        return ""
    if response.status_code != 200:
        return ""
    value = response.json()
    return str(value or "").strip()


def _mlflow_ui_base() -> str:
    """링크용 MLflow 주소: 연결 주소가 브라우저에서 열 수 있는 주소면 그대로, 아니면 MLFLOW_UI_URL."""
    uri = (st.session_state.get("mlflow_tracking_uri") or "").rstrip("/")
    host = urlparse(uri).hostname
    if uri.startswith(("http://", "https://")) and host and host not in _INTERNAL_MLFLOW_HOSTS:
        return uri
    return MLFLOW_UI_URL


def _on_apply_tracking_uri() -> None:
    new_uri = (st.session_state.get("mlflow_uri_input") or "").strip()
    if not new_uri:
        st.session_state["_mlflow_uri_notice"] = ("warning", "MLflow 서버 주소를 입력해주세요.")
        return
    try:
        response = requests.put(
            f"{API_BASE_URL}/mlflow/tracking-uri",
            json={"tracking_uri": new_uri},
            timeout=15,
        )
    except requests.RequestException as exc:
        st.session_state["_mlflow_uri_notice"] = ("error", f"백엔드 요청 실패: {exc}")
        return
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.session_state["_mlflow_uri_notice"] = ("error", f"연결 실패: {detail}")
        return
    applied = response.json().get("tracking_uri", new_uri)
    st.session_state["mlflow_tracking_uri"] = applied
    st.session_state["mlflow_uri_input"] = applied
    # 이전 서버의 실험/run 선택 상태는 새 서버와 맞지 않으므로 초기화
    for key in ("sidebar_experiment_select", "sidebar_metric_select", "runs_table"):
        st.session_state.pop(key, None)
    st.session_state["experiment_info"] = {}
    st.session_state["selected_runs_data"] = []
    st.session_state["selected_runs_meta"] = {}
    st.session_state["chat_resolved_context"] = {}
    st.session_state["_mlflow_uri_notice"] = ("success", f"{applied} 에 연결했습니다.")


_init_session_defaults()
st.session_state["mlflow_tracking_uri"] = _fetch_tracking_uri()
if "mlflow_uri_input" not in st.session_state:
    st.session_state["mlflow_uri_input"] = st.session_state["mlflow_tracking_uri"]
_refresh_experiment_names()

experiment_names = st.session_state["experiment_names"]

# =========================================================================
# Left Sidebar - 실험 선택 / 실행 버튼 영역
# =========================================================================
with st.sidebar:
    # ---------------- MLflow 연결 ----------------
    current_uri = st.session_state.get("mlflow_tracking_uri") or "-"
    with st.expander(f"MLflow 연결 · {current_uri}", icon=":material/link:", expanded=False):
        st.text_input(
            "MLflow 서버 주소",
            key="mlflow_uri_input",
            placeholder="http://127.0.0.1:2522",
        )
        st.button(
            "연결",
            key="mlflow_uri_apply",
            use_container_width=True,
            on_click=_on_apply_tracking_uri,
        )
        uri_notice = st.session_state.pop("_mlflow_uri_notice", None)
        if uri_notice:
            level, text = uri_notice
            getattr(st, level)(text)

    # ---------------- 실험 파트 ----------------
    st.markdown('<div class="sidebar-section-title">실험</div>', unsafe_allow_html=True)

    if st.session_state["names_fetch_error"]:
        st.warning(st.session_state["names_fetch_error"])

    selected_experiment = st.selectbox(
        "실험 이름",
        [DIRECT_INPUT_LABEL] + experiment_names,
        key="sidebar_experiment_select",
    )
    if not experiment_names:
        st.caption("조회 가능한 실험 목록이 없습니다.")

    st.session_state["selected_experiment"] = selected_experiment.strip()

    metric_keys: list[str] = []
    if selected_experiment.strip() and selected_experiment.strip() != DIRECT_INPUT_LABEL:
        metric_keys = _fetch_metric_keys(selected_experiment.strip())

    selected_metric = st.selectbox(
        "Metric key",
        [NO_SELECTION_LABEL] + metric_keys,
        key="sidebar_metric_select",
    )
    st.session_state["selected_metric"] = selected_metric.strip()

    top_k_options: list[Any] = [DEFAULT_TOP_K_LABEL] + list(range(1, 51))
    selected_top_k = st.selectbox(
        "Top K",
        top_k_options,
        index=0,
        key="sidebar_top_k_select",
    )
    st.session_state["selected_top_k"] = selected_top_k

    view_experiment_clicked = st.button(
        "실험 정보 조회",
        icon=":material/search:",
        use_container_width=True,
        key="sidebar_view_experiment_btn",
    )

    st.divider()

    # ---------------- 보고서 파트 ----------------
    st.markdown('<div class="sidebar-section-title">보고서</div>', unsafe_allow_html=True)
    can_generate = (
        bool(selected_experiment.strip())
        and selected_experiment.strip() != DIRECT_INPUT_LABEL
        and selected_metric != NO_SELECTION_LABEL
    )
    if can_generate:
        criteria = (
            f"{html.escape(_metric_short(selected_metric))} · "
            f"상위 {_report_top_k(selected_top_k)}개 Run"
        )
    else:
        criteria = '<span class="exp-empty">실험과 Metric key를 선택하세요</span>'
    st.markdown(
        f'<div class="report-criteria"><span class="exp-field-label">분석 기준</span> {criteria}</div>',
        unsafe_allow_html=True,
    )
    generate_report_clicked = st.button(
        "새 보고서 생성",
        icon=":material/add:",
        use_container_width=True,
        key="sidebar_generate_report_btn",
        type="primary",
        disabled=not can_generate,
    )
    st.write("")

    _render_saved_reports_panel(
        experiment_name=selected_experiment.strip(),
        panel_key="sidebar",
    )


# =========================================================================
# Main Screen - Title
# =========================================================================
st.title("MLflow 실험 분석 워크스페이스")
st.markdown("실험 결과를 비교하고, 보고서를 생성하고, AI에게 질문하세요.")

# =========================================================================
# Action handlers (sidebar buttons)
# =========================================================================
if view_experiment_clicked:
    exp = selected_experiment.strip()
    if not exp or exp == DIRECT_INPUT_LABEL:
        st.warning("experiment name을 선택해주세요.")
    else:
        st.session_state["experiment_info"] = _fetch_experiment_info(exp) or {}
        # 다른 실험의 run이 남지 않도록 초기화 후 조회
        st.session_state["selected_runs_data"] = []
        st.session_state["selected_runs_meta"] = {}
        if selected_metric == NO_SELECTION_LABEL:
            st.warning("Run 목록을 보려면 Metric key를 선택해주세요.")
        elif st.session_state["experiment_info"]:
            runs_data, runs_meta = _fetch_runs_for_view(exp, selected_metric, selected_top_k)
            st.session_state["selected_runs_data"] = runs_data
            st.session_state["selected_runs_meta"] = runs_meta

if generate_report_clicked:
    if not selected_experiment.strip():
        st.warning("experiment name을 입력해주세요.")
    elif selected_metric == NO_SELECTION_LABEL:
        st.warning("보고서 생성을 위해 Metric key를 선택해주세요.")
    else:
        with st.spinner("보고서를 생성하는 중입니다..."):
            _generate_report(selected_experiment.strip(), selected_metric, selected_top_k)


# =========================================================================
# Main Screen - 실험 요약 + Run 목록
# =========================================================================
experiment_info = st.session_state.get("experiment_info") or {}
if not experiment_info:
    st.caption("좌측 사이드바에서 실험과 Metric key를 선택한 뒤 **실험 정보 조회**를 누르세요.")
else:
    _render_experiment_summary(experiment_info)
    st.write("")
    _render_runs_section(
        st.session_state["selected_runs_data"],
        st.session_state["selected_runs_meta"],
    )


report_toast = st.session_state.pop("_report_toast", None)
if report_toast:
    st.toast(report_toast)


# =========================================================================
# Floating QA chatbot button (우하단)
# =========================================================================
open_chat = st.button(
    "QA 챗봇",
    key="qa_floating_chat_btn",
    help="실험 결과 QA Agent와 대화합니다.",
)

# Streamlit은 팝업을 한 번에 하나만 열 수 있다 → 보고서 팝업 우선
if st.session_state.pop("_open_report_dialog", False) and st.session_state.get("generated_report_text"):
    show_report_dialog()
elif open_chat or st.session_state.pop("_open_chat_dialog", False):
    show_chat_dialog()
