import os
from typing import Any

import requests
import streamlit as st

from backend.app.schemas import RunDisplayRow, format_display_datetime
from backend.app.services.report_service import infer_metric_direction

st.title("MLflow + FastAPI + Streamlit 템플릿")
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
    </style>
    """,
    unsafe_allow_html=True,
)

# API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8010")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8888")
DEFAULT_TOP_K_LABEL = "Default(전체)"
NO_SELECTION_LABEL = "(선택 안함)"
DIRECT_INPUT_LABEL = "(직접 입력)"

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
    "chat_messages": [
        {
            "role": "assistant",
            "content": "안녕하세요. 실험 관련 질문을 입력해 주세요.",
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
    st.session_state["generated_report_text"] = body.get("report", "")
    st.session_state["generated_report_title"] = body.get("report_title", "")
    st.session_state["generated_report_meta"] = {"experiment_name": exp, "metric_key": metric}
    st.success("보고서를 생성했습니다.")
    st.caption(f"모델: `{body.get('model', '-')}` · 비교 run 목록: {body.get('metadata', {})}")
    st.markdown(body.get("report", ""))


def _render_chat_bubble(role: str, content: str) -> None:
    if role == "assistant":
        st.markdown(content)
    else:
        st.write(content)


def _append_and_answer_chat(
    user_prompt: str,
    current_experiment: str,
    current_metric: str,
) -> None:
    st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    chat_payload = {"message": user_prompt}
    # 리포트 생성 여부와 무관하게 현재 선택한 실험/metric 문맥을 전달한다.
    if current_experiment and current_experiment != DIRECT_INPUT_LABEL:
        chat_payload["experiment_name"] = current_experiment
    if current_metric and current_metric != NO_SELECTION_LABEL:
        chat_payload["metric_key"] = current_metric

    report_text = st.session_state.get("generated_report_text", "")
    report_meta = st.session_state.get("generated_report_meta", {})
    if report_text:
        chat_payload["report_text"] = report_text
        if report_meta.get("experiment_name"):
            chat_payload["experiment_name"] = report_meta.get("experiment_name")
        if report_meta.get("metric_key"):
            chat_payload["metric_key"] = report_meta.get("metric_key")

    chat_response = _safe_request("POST", "/chat/ask", json=chat_payload, timeout=30)
    if chat_response is None:
        answer = "챗봇 요청 실패: 백엔드 요청에 실패했습니다."
        route = {}
    elif chat_response.status_code == 200:
        data = chat_response.json()
        answer = data.get("answer", "응답이 비어 있습니다.")
        route = (data.get("metadata") or {}).get("route") or {}
    else:
        answer = f"챗봇 응답 실패 ({chat_response.status_code})"
        route = {}

    st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        path = route.get("path") or [route.get("handler", "-")]
        st.caption(f"route: {route.get('parser', '-')} ⇒ " + " → ".join(path))
        _render_chat_bubble("assistant", answer)


_init_session_defaults()
_refresh_experiment_names()

experiment_names = st.session_state["experiment_names"]
if st.session_state["names_fetch_error"]:
    st.warning(st.session_state["names_fetch_error"])

selected_experiment = st.selectbox(
    "조회할 experiment name 선택",
    [DIRECT_INPUT_LABEL] + experiment_names,
)
if not experiment_names:
    st.caption("조회 가능한 실험 목록이 없습니다.")

st.session_state["selected_experiment"] = selected_experiment.strip()

metric_keys = []
if selected_experiment.strip():
    metric_keys = _fetch_metric_keys(selected_experiment.strip())

selected_metric = st.selectbox("Metric key 선택", [NO_SELECTION_LABEL] + metric_keys)
st.session_state["selected_metric"] = selected_metric.strip()
top_k_options: list[Any] = [DEFAULT_TOP_K_LABEL] + list(range(1, 51))
selected_top_k = st.selectbox("Top K", top_k_options, index=0)
st.session_state["selected_top_k"] = selected_top_k

action_col1, action_col2 = st.columns(2)
view_experiment_clicked = action_col1.button("실험 정보 조회", use_container_width=True)
generate_report_clicked = action_col2.button("실험 보고서 생성", use_container_width=True)


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
    # st.write(data)
    st.write(f"**Artifact Location**: `{data.get('artifact_location', '-')}`")
    st.write(f"**Creation Time**: `{format_display_datetime(data.get('creation_time'))}`")
    st.write(f"**Last Update Time**: `{format_display_datetime(data.get('last_update_time'))}`")
    st.write(f"**Trace Location**: `{data.get('trace_location', '-')}`")
    st.write(f"**Workspace**: `{data.get('workspace', '-')}`")
    st.subheader("Tags")
    tags = data.get("tags", {})
    if tags:
        st.json(tags)
    else:
        st.caption("등록된 태그가 없습니다.")

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

st.subheader("Runs")
runs_data = st.session_state["selected_runs_data"]
runs_meta = st.session_state["selected_runs_meta"]
if not runs_data:
    st.caption("실험명/Metric key/Top K를 선택한 뒤 **실험 정보 조회**를 누르면 runs가 표시됩니다.")
else:
    st.caption(
        f"experiment `{runs_meta.get('experiment_name', '-')}` · "
        f"`{runs_meta.get('metric_key', '-')}` 기준 {len(runs_data)}개 "
        f"(K={runs_meta.get('top_k', '-')})"
    )
    st.dataframe(
        [RunDisplayRow.from_summary_dict(r).to_dataframe_row() for r in runs_data],
        use_container_width=True,
    )



st.divider()
chat_col1, chat_col2 = st.columns([4, 1])
with chat_col1:
    st.caption("버튼을 누르면 팝업 챗봇이 열립니다.")
with chat_col2:
    open_chat = st.button("Chat", use_container_width=True)

@st.dialog("실험 Q&A 챗봇", width="small")
def show_chat_dialog() -> None:
    current_experiment = (st.session_state.get("selected_experiment") or "").strip()
    current_metric = (st.session_state.get("selected_metric") or "").strip()
    current_report_title = (st.session_state.get("generated_report_title") or "").strip()
    exp_label = current_experiment if current_experiment else "-"
    metric_label = current_metric if current_metric and current_metric != NO_SELECTION_LABEL else "-"
    report_label = current_report_title if current_report_title else "-"
    st.caption(f"현재 참조 문맥: experiment `{exp_label}` · metric `{metric_label}` · report `{report_label}`")

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
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


if open_chat:
    show_chat_dialog()
