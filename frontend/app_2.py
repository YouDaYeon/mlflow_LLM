import importlib
import os
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests
import streamlit as st

from backend.app import schemas as _backend_schemas

# Streamlit은 import를 캐시하므로 schemas 변경이 UI에 안 먹을 수 있음 → 매 실행 reload
importlib.reload(_backend_schemas)
from backend.app.schemas import RunDisplayRow, format_display_datetime
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
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8011")
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
    "generated_report_text": "",
    "generated_report_title": "",
    "generated_report_meta": {},
    "report_edit_mode": False,
    "report_edit_draft": "",
    "report_edit_title_draft": "",
    "chat_session_id": None,
    "chat_resolved_context": {},  # intent 후처리된 experiment/metric/top_k
    "chat_messages": [
        {
            "role": "assistant",
            "content": CHAT_WELCOME_MESSAGE,
        }
    ],
}

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
    meta = st.session_state.get("generated_report_meta") or {}
    if str(meta.get("id") or "") == str(report_id):
        st.session_state["generated_report_text"] = ""
        st.session_state["generated_report_title"] = ""
        st.session_state["generated_report_meta"] = {}
        st.session_state["report_edit_mode"] = False
        st.session_state["_pending_clear_report_edit_drafts"] = True


def _saved_reports_table_rows(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in items:
        rows.append(
            {
                "Title": str(item.get("report_title", "-")),
                "Experiment": str(item.get("experiment_name", "-")),
                "Metric": str(item.get("metric_key", "-")),
                "Created At": _format_ui_datetime(item.get("created_at")),
                "Updated At": _format_ui_datetime(
                    item.get("updated_at") or item.get("created_at")
                ),
            }
        )
    return rows


def _report_option_label(item: dict[str, Any]) -> str:
    title = str(item.get("report_title") or "(제목 없음)")
    created = _format_ui_datetime(item.get("created_at"))
    experiment = str(item.get("experiment_name") or "-")
    short_id = str(item.get("id", ""))[:8]
    return f"{title} · {experiment} · {created} · {short_id}"


def _render_saved_reports_panel(
    *,
    experiment_name: str,
    panel_key: str,
    show_filter: bool = True,
) -> None:
    filter_current_experiment = False
    if show_filter:
        filter_current_experiment = st.checkbox(
            "현재 선택한 실험만 보기",
            value=bool(experiment_name and experiment_name != DIRECT_INPUT_LABEL),
            key=f"{panel_key}_filter_current_experiment",
        )

    exp_filter: str | None = None
    if filter_current_experiment and experiment_name and experiment_name != DIRECT_INPUT_LABEL:
        exp_filter = experiment_name.strip()

    refresh_clicked = st.button("목록 새로고침", key=f"{panel_key}_refresh_reports", use_container_width=True)

    cache_key = f"{panel_key}_saved_reports"
    cache_filter_key = f"{panel_key}_saved_reports_filter"
    if (
        refresh_clicked
        or cache_key not in st.session_state
        or st.session_state.get(cache_filter_key) != exp_filter
    ):
        st.session_state[cache_key] = _fetch_saved_reports(exp_filter)
        st.session_state[cache_filter_key] = exp_filter

    if st.session_state.pop("_pending_reset_report_select", False):
        st.session_state.pop(f"{panel_key}_selected_report", None)

    items: list[dict[str, Any]] = st.session_state.get(cache_key, [])
    if not items:
        st.caption("저장된 보고서가 없습니다.")
        return

    st.dataframe(_saved_reports_table_rows(items), use_container_width=True, hide_index=True)

    option_labels = [_report_option_label(item) for item in items]
    label_to_id = {
        label: str(item.get("id", ""))
        for label, item in zip(option_labels, items)
        if item.get("id")
    }
    selected_label = st.selectbox(
        "보고서 선택",
        options=option_labels,
        key=f"{panel_key}_selected_report",
    )
    load_clicked = st.button(
        "선택한 보고서 보기",
        key=f"{panel_key}_load_report",
        use_container_width=True,
    )
    delete_clicked = st.button(
        "보고서 삭제하기",
        key=f"{panel_key}_delete_report",
        use_container_width=True,
    )
    if load_clicked:
        report_id = label_to_id.get(selected_label, "")
        if not report_id:
            st.warning("불러올 보고서를 선택해주세요.")
            return
        body = _fetch_saved_report(report_id)
        if body is not None:
            _apply_report_from_body(body, source="loaded")
            st.success("저장된 보고서를 불러왔습니다.")
    if delete_clicked:
        report_id = label_to_id.get(selected_label, "")
        if not report_id:
            st.warning("삭제할 보고서를 선택해주세요.")
            return
        if _delete_saved_report(report_id):
            _clear_main_report_if_matching(report_id)
            st.session_state.pop(cache_key, None)
            st.success("보고서를 삭제했습니다.")
            st.rerun()


def _generate_report(exp: str, metric: str, selected_top_k: Any) -> None:
    report_top_k = 3 if selected_top_k == DEFAULT_TOP_K_LABEL else min(int(selected_top_k), 3)
    if selected_top_k == DEFAULT_TOP_K_LABEL or int(selected_top_k) > 3:
        st.info("보고서 API는 상위 run 최대 3개만 사용합니다. 상위 3개로 생성합니다.")

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


def _format_tool_calls_caption(tool_calls: Any) -> str | None:
    """metadata tool_calls를 intent 캡션과 같은 한 줄로 만든다."""
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


def _format_chat_context_caption(
    *,
    sidebar_experiment: str,
    sidebar_metric: str,
    report_title: str,
) -> str:
    resolved = st.session_state.get("chat_resolved_context") or {}
    exp = resolved.get("experiment_name") or (
        sidebar_experiment if sidebar_experiment and sidebar_experiment != DIRECT_INPUT_LABEL else None
    )
    metric = resolved.get("metric_key") or (
        sidebar_metric if sidebar_metric and sidebar_metric != NO_SELECTION_LABEL else None
    )
    top_k = resolved.get("top_k")
    report = report_title or None

    parts = [
        f"experiment `{exp or '-'}`",
        f"metric `{metric or '-'}`",
    ]
    if top_k is not None:
        parts.append(f"top_k `{top_k}`")
    parts.append(f"report `{report or '-'}`")
    return "현재 참조 문맥: " + " · ".join(parts)


def _on_new_chat_click() -> None:
    """dialog 유지: st.rerun() 없이 세션만 초기화."""
    _reset_chat_to_new()
    st.session_state["chat_session_select"] = "(현재 대화)"


def _delete_chat_session(session_id: str) -> bool:
    response = _safe_request("DELETE", f"/chat/sessions/{session_id}", timeout=10)
    if response is None:
        return False
    if response.status_code not in (200, 204):
        st.error(f"대화 삭제 실패 ({response.status_code})")
        st.code(response.text)
        return False
    return True


def _on_delete_chat_session_click() -> None:
    """선택된(또는 현재) 세션 hard delete 후 새 대화로 초기화."""
    label = st.session_state.get("chat_session_select")
    mapping = st.session_state.get("_chat_session_label_map") or {}
    target_id = mapping.get(label) if label else None
    if not target_id:
        target_id = st.session_state.get("chat_session_id")
    if not target_id:
        st.session_state["_chat_delete_notice"] = "삭제할 저장된 대화가 없습니다."
        return
    if _delete_chat_session(str(target_id)):
        _reset_chat_to_new()
        st.session_state["chat_session_select"] = "(현재 대화)"
        st.session_state["_chat_delete_notice"] = "대화를 삭제했습니다."


def _on_chat_session_select() -> None:
    """목록 변경 시 이력 로드. '(현재 대화)'는 새 대화와 동일하게 초기화."""
    label = st.session_state.get("chat_session_select")
    if not label or label == "(현재 대화)":
        _reset_chat_to_new()
        return
    mapping = st.session_state.get("_chat_session_label_map") or {}
    target_id = mapping.get(label)
    current_sid = str(st.session_state.get("chat_session_id") or "")
    if target_id and target_id != current_sid:
        _load_chat_session(target_id)


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
            tool_calls = (item.get("metadata") or {}).get("tool_calls")
            if tool_calls:
                message["tool_calls"] = tool_calls
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

    report_text = st.session_state.get("generated_report_text", "")
    report_meta = st.session_state.get("generated_report_meta", {})
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
    if chat_response is None:
        answer = "챗봇 요청 실패: 백엔드 요청에 실패했습니다."
        intent = "-"
    elif chat_response.status_code == 200:
        data = chat_response.json()
        answer = data.get("answer", "응답이 비어 있습니다.")
        intent = data.get("intent", "-")
        meta = data.get("metadata") or {}
        tool_calls = meta.get("tool_calls")
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
        intent = "-"

    assistant_message: dict[str, Any] = {"role": "assistant", "content": answer}
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls
    st.session_state["chat_messages"].append(assistant_message)
    with st.chat_message("assistant"):
        st.caption(f"intent: {intent}")
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


@st.dialog("실험 정보", width="large")
def show_experiment_info_dialog(exp_name: str) -> None:
    st.caption("창 바깥을 클릭하거나 우측 상단 X로 닫을 수 있습니다. 다시 열려면 **실험 정보 조회**를 누르세요.")
    response = _safe_request(
        "GET",
        "/mlflow/experiment",
        params={"experiment_name": exp_name},
        timeout=10,
    )
    if response is None:
        return
    if response.status_code != 200:
        st.error(f"조회 실패 ({response.status_code})")
        st.code(response.text)
        return
    data = response.json().get("experiment", {})
    st.success("실험 정보를 조회했습니다.")
    st.subheader("Experiment Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Experiment Name", data.get("experiment_name", "-"))
    c2.metric("Experiment ID", data.get("experiment_id", "-"))
    c3.metric("Lifecycle Stage", data.get("lifecycle_stage", "-"))
    st.subheader("Details")
    st.write(f"**Artifact Location**: `{data.get('artifact_location', '-')}`")
    st.write(f"**Creation Time**: `{_format_ui_datetime(data.get('creation_time'))}`")
    st.write(f"**Last Update Time**: `{_format_ui_datetime(data.get('last_update_time'))}`")
    st.write(f"**Trace Location**: `{data.get('trace_location', '-')}`")
    st.write(f"**Workspace**: `{data.get('workspace', '-')}`")
    st.subheader("Tags")
    tags = data.get("tags", {})
    if tags:
        st.json(tags)
    else:
        st.caption("등록된 태그가 없습니다.")


@st.dialog("실험 Q&A 챗봇", width="large")
def show_chat_dialog() -> None:
    current_experiment = (st.session_state.get("selected_experiment") or "").strip()
    current_metric = (st.session_state.get("selected_metric") or "").strip()
    current_report_title = (st.session_state.get("generated_report_title") or "").strip()
    st.caption(
        _format_chat_context_caption(
            sidebar_experiment=current_experiment,
            sidebar_metric=current_metric,
            report_title=current_report_title,
        )
    )

    sessions = _fetch_chat_sessions(limit=20)
    session_labels = ["(현재 대화)"]
    label_to_session_id: dict[str, str] = {}
    for item in sessions:
        sid = str(item.get("session_id") or "")
        title = (item.get("title") or "제목 없음").strip() or "제목 없음"
        updated = _format_ui_datetime(item.get("updated_at"))
        label = f"{title} · {updated}" if updated and updated != "-" else title
        if label in label_to_session_id:
            label = f"{label} ({sid[:8]})"
        session_labels.append(label)
        label_to_session_id[label] = sid

    st.session_state["_chat_session_label_map"] = label_to_session_id

    current_sid = str(st.session_state.get("chat_session_id") or "")
    default_index = 0
    if current_sid:
        for idx, label in enumerate(session_labels):
            if label_to_session_id.get(label) == current_sid:
                default_index = idx
                break

    if "chat_session_select" not in st.session_state:
        st.session_state["chat_session_select"] = session_labels[default_index]
    elif st.session_state["chat_session_select"] not in session_labels:
        st.session_state["chat_session_select"] = session_labels[default_index]

    col_sel, col_new, col_del = st.columns([3, 1, 1])
    with col_sel:
        st.selectbox(
            "대화 목록",
            session_labels,
            key="chat_session_select",
            on_change=_on_chat_session_select,
        )
    with col_new:
        st.write("")
        st.button(
            "새 대화",
            key="chat_session_new",
            use_container_width=True,
            on_click=_on_new_chat_click,
        )
    with col_del:
        st.write("")
        st.button(
            "이 대화 삭제",
            key="chat_session_delete",
            use_container_width=True,
            on_click=_on_delete_chat_session_click,
            disabled=(
                st.session_state.get("chat_session_select") == "(현재 대화)"
                and not st.session_state.get("chat_session_id")
            ),
        )

    notice = st.session_state.pop("_chat_delete_notice", None)
    if notice:
        st.caption(notice)

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_tool_calls_caption(message.get("tool_calls"))
            _render_chat_bubble(message["role"], message["content"])

    st.caption("예시 질문")
    example_cols = st.columns(2)
    selected_example_prompt: str | None = None
    for idx, question in enumerate(CHAT_EXAMPLE_QUESTIONS):
        if example_cols[idx % 2].button(question, key=f"chat_example_{idx}", use_container_width=True):
            selected_example_prompt = question

    user_prompt = st.chat_input("질문을 입력하세요", key="chat_input_dialog")
    final_prompt = user_prompt or selected_example_prompt
    if final_prompt:
        _append_and_answer_chat(final_prompt, current_experiment, current_metric)


_init_session_defaults()
_refresh_experiment_names()

experiment_names = st.session_state["experiment_names"]

# =========================================================================
# Left Sidebar - 실험 선택 / 실행 버튼 영역
# =========================================================================
with st.sidebar:
    st.header("실험 선택")

    if st.session_state["names_fetch_error"]:
        st.warning(st.session_state["names_fetch_error"])

    selected_experiment = st.selectbox(
        "조회할 experiment name 선택",
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
        "Metric key 선택",
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

    st.divider()

    st.subheader("실행")
    view_experiment_clicked = st.button(
        "실험 정보 조회",
        use_container_width=True,
        key="sidebar_view_experiment_btn",
    )
    generate_report_clicked = st.button(
        "보고서 생성하기",
        use_container_width=True,
        key="sidebar_generate_report_btn",
        type="primary",
    )

    with st.expander("저장된 보고서", expanded=True):
        _render_saved_reports_panel(
            experiment_name=selected_experiment.strip(),
            panel_key="sidebar",
        )


# =========================================================================
# Main Screen - Title
# =========================================================================
st.title("MLflow + FastAPI + Streamlit 템플릿")

# =========================================================================
# Action handlers (sidebar buttons)
# =========================================================================
if view_experiment_clicked:
    if not selected_experiment.strip():
        st.warning("experiment name을 입력해주세요.")
    elif selected_metric == NO_SELECTION_LABEL:
        exp = selected_experiment.strip()
        show_experiment_info_dialog(exp)
        st.warning("Metric key를 선택해주세요.")
    else:
        exp = selected_experiment.strip()
        show_experiment_info_dialog(exp)
        runs_data, runs_meta = _fetch_runs_for_view(exp, selected_metric, selected_top_k)

        st.session_state["selected_runs_data"] = runs_data
        st.session_state["selected_runs_meta"] = runs_meta

if generate_report_clicked:
    if not selected_experiment.strip():
        st.warning("experiment name을 입력해주세요.")
    elif selected_metric == NO_SELECTION_LABEL:
        st.warning("보고서 생성을 위해 Metric key를 선택해주세요.")
    else:
        _generate_report(selected_experiment.strip(), selected_metric, selected_top_k)


# =========================================================================
# Main Screen - Runs 화면
# =========================================================================
st.subheader("Runs")
runs_data = st.session_state["selected_runs_data"]
runs_meta = st.session_state["selected_runs_meta"]
if not runs_data:
    st.caption("실험명/Metric key/Top K를 선택한 뒤 **실험 정보 조회**를 누르면 runs가 표시됩니다.")
else:
    st.caption(
        f"experiment `{runs_meta.get('experiment_name', '-')}` · "
        f"`{runs_meta.get('metric_key', '-')}`"
        f" ({runs_meta.get('metric_direction') or '방향 미정'}) 기준 {len(runs_data)}개 "
        f"(K={runs_meta.get('top_k', '-')})"
    )
    st.dataframe(
        [RunDisplayRow.from_summary_dict(r).to_dataframe_row() for r in runs_data],
        use_container_width=True,
    )


# =========================================================================
# Main Screen - Report 화면 (Runs 정보 아래)
# 목록/선택은 사이드바 "실행" 영역만. 메인은 생성·불러오기 결과 본문만 표시.
# =========================================================================
st.divider()
st.subheader("보고서")

delete_notice = st.session_state.pop("_report_delete_notice", None)
if delete_notice:
    st.success(delete_notice)

report_text = st.session_state.get("generated_report_text", "")
report_meta = st.session_state.get("generated_report_meta", {})
if not report_text:
    st.caption(
        "좌측 사이드바에서 **보고서 생성하기** 또는 "
        "**선택한 보고서 보기** 를 사용하세요."
    )
else:
    # edit widget key는 위젯 생성 전에만 정리 가능
    if (
        st.session_state.pop("_pending_clear_report_edit_drafts", False)
        or not st.session_state.get("report_edit_mode")
    ):
        st.session_state.pop("report_edit_draft", None)
        st.session_state.pop("report_edit_title_draft", None)

    source = report_meta.get("source", "generated")
    if source == "loaded":
        st.info("저장된 보고서를 불러왔습니다.")
    elif source == "edited":
        st.success("보고서 수정을 저장했습니다.")
    else:
        st.success("보고서를 생성했습니다.")

    created_at = report_meta.get("created_at") or ""
    updated_at = report_meta.get("updated_at") or created_at
    created_label = _format_ui_datetime(created_at) if created_at else "-"
    updated_label = _format_ui_datetime(updated_at) if updated_at else "-"
    report_id = str(report_meta.get("id") or "")

    can_manage_report = bool(report_id) and not st.session_state.get("report_edit_mode")
    meta_col, edit_col, delete_col = st.columns([4, 1, 1])
    with meta_col:
        st.caption(
            f"제목: `{st.session_state.get('generated_report_title') or '-'}` · "
            f"모델: `{report_meta.get('model', '-')}` · "
            f"생성: `{created_label}` · "
            f"수정: `{updated_label}` · "
            f"비교 run 목록: {report_meta.get('metadata', {})}"
        )
    with edit_col:
        if can_manage_report and st.button("수정하기", key="report_edit_btn", use_container_width=True):
            st.session_state["report_edit_mode"] = True
            st.session_state["report_edit_draft"] = report_text
            st.session_state["report_edit_title_draft"] = (
                st.session_state.get("generated_report_title") or ""
            )
            st.rerun()
    with delete_col:
        if can_manage_report and st.button("삭제하기", key="report_delete_btn", use_container_width=True):
            if _delete_saved_report(report_id):
                _clear_main_report_if_matching(report_id)
                st.session_state.pop("sidebar_saved_reports", None)
                st.session_state["_pending_reset_report_select"] = True
                st.session_state["_report_delete_notice"] = "보고서를 삭제했습니다."
                st.rerun()

    if st.session_state.get("report_edit_mode"):
        if not report_id:
            st.warning("저장되지 않은 보고서라 수정할 수 없습니다.")
            st.session_state["report_edit_mode"] = False
        else:
            st.text_input(
                "보고서 제목",
                key="report_edit_title_draft",
            )
            st.text_area(
                "보고서 내용",
                key="report_edit_draft",
                height=420,
            )
            save_col, cancel_col = st.columns(2)
            with save_col:
                save_clicked = st.button(
                    "저장",
                    key="report_edit_save",
                    use_container_width=True,
                    type="primary",
                )
            with cancel_col:
                cancel_clicked = st.button(
                    "취소",
                    key="report_edit_cancel",
                    use_container_width=True,
                )
            if cancel_clicked:
                st.session_state["report_edit_mode"] = False
                st.session_state["_pending_clear_report_edit_drafts"] = True
                st.rerun()
            if save_clicked:
                draft = (st.session_state.get("report_edit_draft") or "").strip()
                title_draft = (st.session_state.get("report_edit_title_draft") or "").strip()
                if not draft:
                    st.warning("보고서 내용이 비어 있습니다.")
                else:
                    body = _update_saved_report(
                        report_id,
                        report=draft,
                        report_title=title_draft or None,
                    )
                    if body is not None:
                        _apply_report_from_body(body, source="edited")
                        st.rerun()
    else:
        st.markdown(report_text, unsafe_allow_html=True)


# =========================================================================
# Floating QA chatbot button (우하단)
# =========================================================================
open_chat = st.button(
    "QA 챗봇",
    key="qa_floating_chat_btn",
    help="실험 결과 QA Agent와 대화합니다.",
)

if open_chat:
    show_chat_dialog()
