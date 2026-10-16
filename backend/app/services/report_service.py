from __future__ import annotations

import json
import re

from openai import OpenAI

from backend.app.config import settings
from backend.app.schemas import RunSummary

# metric_key 이름만으로 방향을 알 수 있는 경우. 모르면 None이라 프롬프트가 우열을 단정하지 않는다.
_MINIMIZE_RE = re.compile(
    r"rmse|mse|mae|loss|latency|inference[_\-\s]?time|infer[_\-\s]?time",
    re.IGNORECASE,
)
_MAXIMIZE_RE = re.compile(
    r"accuracy|\bf1(?:[_\-]?score)?\b|recall|precision|\bmap\d*|\bauc\b|auroc|roc[_\-]?auc",
    re.IGNORECASE,
)
_METRIC_DIRECTIONS = {"maximize", "minimize"}


def infer_metric_direction(metric_key: str) -> str | None:
    """지표 이름에서 최적화 방향을 추정한다. 확실하지 않으면 None."""
    leaf = metric_key.strip().rsplit("/", 1)[-1].rsplit(".", 1)[-1]
    if not leaf:
        return None
    minimize = _MINIMIZE_RE.search(leaf) is not None
    maximize = _MAXIMIZE_RE.search(leaf) is not None
    if minimize and not maximize:
        return "minimize"
    if maximize and not minimize:
        return "maximize"
    return None


def resolve_metric_direction(metric_key: str, metric_direction: str | None = None) -> str | None:
    """호출자가 방향을 주면 그 값을 쓰고, 없으면 지표 이름으로 추정한다."""
    if metric_direction in _METRIC_DIRECTIONS:
        return metric_direction
    return infer_metric_direction(metric_key)


def order_top_runs(top_runs: list[RunSummary], metric_direction: str | None) -> list[RunSummary]:
    """방향이 있으면 좋은 값이 앞에 오도록 정렬한다. 없으면 입력 순서를 유지한다."""
    if metric_direction == "maximize":
        return sorted(top_runs, key=lambda run: run.metric_score, reverse=True)
    if metric_direction == "minimize":
        return sorted(top_runs, key=lambda run: run.metric_score)
    return list(top_runs)


SYSTEM_PROMPT = """당신은 딥러닝 실험 분석가다.
주어진 실험 데이터(JSON)에 포함된 값만 근거로 보고서를 작성하라.
값이 없는 내용은 추측하지 말고 '근거 부족'으로 명시하라.
metric_direction이 maximize이면 값이 클수록, minimize이면 값이 작을수록 좋은 지표로 평가하라.
metric_direction이 없으면 높고 낮음을 좋고 나쁨으로 단정하지 마라.
응답은 한국어로 작성하라.
보고서 본문은 읽기 좋은 HTML 조각으로만 작성하라.
마크다운, 코드 펜스(```), 서두 설명은 출력하지 마라.
<html>, <head>, <body>, <style>, <script> 태그는 사용하지 마라.
사용할 수 있는 태그는 h1, h2, h3, p, ul, ol, li, table, thead, tbody, tr, th, td, strong이다.
표는 지표 비교처럼 숫자가 나란히 보일 때만 쓰고, 문단과 목록으로 구분해 읽기 쉽게 구성하라.
"""


def _report_title(experiment_name: str) -> str:
    return f"{experiment_name} 실험 분석 보고서"


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _ensure_report_title(report_content: str, experiment_name: str) -> str:
    title = _report_title(experiment_name)
    cleaned = _strip_code_fence(report_content)
    if not cleaned:
        return f"<h1>{title}</h1>"
    if title in cleaned:
        return cleaned
    if cleaned.lstrip().startswith("<"):
        return f"<h1>{title}</h1>\n{cleaned}"
    return f"{title}\n\n{cleaned}"


def _build_user_prompt(
    experiment_name: str,
    metric_key: str,
    top_runs: list[RunSummary],
    metric_direction: str | None = None,
) -> str:
    payload = {
        "experiment_name": experiment_name,
        "metric_key": metric_key,
        "metric_direction": metric_direction,
        "top_runs": [run.model_dump() for run in top_runs],
    }

    title = _report_title(experiment_name)

    return f"""당신은 머신러닝 실험 결과를 분석하고 기술 보고서를 작성하는 전문가다.
제공된 JSON 데이터를 근거로 실험 결과 보고서를 작성하라.

## 1. 분석 원칙

- 입력 JSON에 실제로 존재하는 정보만 사실로 기술한다.
- 제공되지 않은 데이터셋, 모델 구조, 하이퍼파라미터, 학습 환경,
  실험 목적, 성능 개선 원인을 임의로 추측하지 않는다.
- 관측된 사실, 그 사실에 대한 해석, 후속 실험을 위한 가설을 구분한다.
- 상관관계만 확인할 수 있는 경우 인과관계로 단정하지 않는다.
- 비교 실험 간 다른 조건이 통제되었는지 확인할 수 없다면,
  특정 파라미터가 성능 차이의 원인이라고 단정하지 않는다.
- metric_direction은 metric_key의 최적화 방향이다.
  "maximize"는 값이 클수록 좋은 지표이고, "minimize"는 값이 작을수록 좋은 지표이다.
  이 값이 있으면 우수 Run, 개선, 저하를 이 기준으로만 판단한다.
  null이면 수치의 크고 작음을 성능의 좋고 나쁨으로 바꾸지 않는다.
- metric_direction이 있으면 top_runs는 그 기준에서 좋은 순서이다. 앞에 있을수록 더 좋다.
  null이면 입력 순서를 성능 순위로 단정하지 않는다.
- 결측값, null, NaN, Infinity 등은 정상적인 수치로 취급하지 않는다.
  유효하지 않은 값은 별도로 표시하거나 계산에서 제외한다.
- 데이터가 부족하여 분석할 수 없는 내용은 명시적으로 한계를 밝힌다.
- 전문적이고 객관적인 한국어 기술 보고서 문체를 사용한다.
- 같은 수치를 반복하기보다 각 항목이 서로 다른 분석 목적을 갖도록 작성한다.

## 2. 보고서 작성 구조

다음 순서와 HTML 태그 구조를 준수한다.

<h1>{title}</h1>

<h2>1. 실험 개요 및 핵심 결과</h2>
<p>
실험명, 평가 지표, metric_direction, 주요 결과를 간결하게 기술한다.
metric_direction이 maximize이면 "값이 클수록 좋음", minimize이면 "값이 작을수록 좋음"으로 적는다.
그 방향에서 가장 좋은 Run을 Best로 제시한다.
metric_direction이 null이면 Best를 단정하지 않고 그 한계를 명시한다.
</p>

<h2>2. 실험 결과 비교</h2>
<p>
각 Run의 핵심 지표를 비교하고, 제공된 경우 주요 파라미터 차이를 설명한다.
</p>
<table>
  <thead>
    <tr>
      <th>Run</th>
      <th>평가 지표</th>
      <th>지표 값</th>
      <th>주요 설정</th>
      <th>비고</th>
    </tr>
  </thead>
  <tbody>
    <!-- 입력 데이터에 존재하는 Run별 결과 -->
  </tbody>
</table>

<h2>3. 실험별 강점 및 한계</h2>
<!-- 각 Run마다 h3와 ul을 작성한다.
강점은 실제 관측된 성능이나 설정상 특징을 근거로 기술한다.
한계는 확인된 성능 저하, 비교 정보의 부족 또는 검증되지 않은 부분을 기술한다.
데이터만으로 확인할 수 없는 특성은 사실처럼 서술하지 않는다. -->

<h2>4. 성능 차이 분석</h2>
<ul>
  <li>주요 Run 간 평가 지표의 차이를 수치로 설명한다.</li>
  <li>계산 가능한 경우 절대 차이와 상대 변화율을 구분한다.</li>
  <li>상대 변화율은 기준값이 0이 아니고 유효한 경우에만 계산한다.</li>
  <li>metric_direction이 maximize 또는 minimize일 때만 개선 또는 저하로 평가한다.</li>
  <li>파라미터와 성능의 관계는 관측 사실과 가설을 구분해 기술한다.</li>
  <li>실험 조건이나 반복 측정 결과가 부족하면 결론의 한계를 명시한다.</li>
</ul>

<h2>5. 후속 실험 계획</h2>
<p>
현재 결과에서 확인된 차이 또는 검증이 필요한 가설을 바탕으로
후속 실험을 제안한다. 각 제안은 기존 결과와 연결되어야 한다.
</p>
<ul>
  <li><strong>실험 목표:</strong> 검증하려는 문제 또는 가설</li>
  <li><strong>변경 변수:</strong> 조정할 하이퍼파라미터 또는 실험 조건</li>
  <li><strong>검증 방법:</strong> 비교 기준, 평가 지표, 통제할 조건</li>
  <li><strong>기대 효과:</strong> 확인하려는 성능 변화. 결과를 미리 단정하지 않는다.</li>
</ul>
<!-- 데이터에 근거한 실행 가능한 제안을 2~4개 작성한다.
기존에 확인되지 않은 최적값을 임의로 제시하지 않는다.
하이퍼파라미터 정보가 없다면 구체적인 파라미터 튜닝 대신
추가로 수집하거나 통제해야 할 정보를 제안한다. -->

<h2>6. 종합 결론</h2>
<p>
현재 제공된 실험에서 확인된 핵심 결과, 가장 유망한 접근법,
추가 검증이 필요한 사항 및 우선 진행할 후속 작업을 요약한다.
확정적인 결론을 내릴 수 없는 경우 그 이유를 함께 기술한다.
</p>

## 3. 출력 제약

- HTML 조각만 출력한다. Markdown 코드 블록이나 별도 설명은 출력하지 않는다.
- 전체 문서를 감싸는 html, head, body 태그는 사용하지 않는다.
- 허용 태그: h1, h2, h3, p, ul, ol, li, table, thead, tbody, tr, th, td, strong, em.
- 모든 태그를 올바르게 닫고 유효한 HTML 구조를 유지한다.
- 표의 열과 데이터 개수를 일치시킨다.
- 입력 데이터에 없는 수치나 실험 결과를 만들어 내지 않는다.
- 빈 값이나 유효하지 않은 값은 임의의 숫자로 대체하지 않는다.
- 보고서의 각 항목에 실질적인 분석 내용을 작성한다.
- 분석할 근거가 없는 항목은 그 한계를 간결하게 설명한다.

## 입력 데이터(JSON)

{json.dumps(payload, ensure_ascii=False, indent=2)}
"""



def generate_report(
    experiment_name: str,
    metric_key: str,
    top_runs: list[RunSummary],
    metric_direction: str | None = None,
) -> tuple[str, str, str]:
    direction = resolve_metric_direction(metric_key, metric_direction)
    ordered_runs = order_top_runs(top_runs, direction)
    prompt = _build_user_prompt(experiment_name, metric_key, ordered_runs, direction)
    client = OpenAI(base_url=settings.llama_base_url, api_key=settings.llama_api_key)

    completion = client.chat.completions.create(
        model=settings.llama_model,
        temperature=getattr(settings, "llama_temperature", 0.2),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    content = completion.choices[0].message.content or ""
    formatted = _ensure_report_title(content, experiment_name)
    report_title = _report_title(experiment_name)
    return formatted, settings.llama_model, report_title