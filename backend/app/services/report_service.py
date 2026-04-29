from __future__ import annotations

import json

from openai import OpenAI

from backend.app.config import settings
from backend.app.schemas import RunSummary


SYSTEM_PROMPT = """당신은 딥러닝 실험 분석가다.
주어진 실험 데이터(JSON)에 포함된 값만 근거로 보고서를 작성하라.
값이 없는 내용은 추측하지 말고 '근거 부족'으로 명시하라.
응답은 한국어로 작성하라.
"""


def _report_title(experiment_name: str) -> str:
    return f"{experiment_name} 실험 분석 보고서"


def _ensure_report_title(report: str, experiment_name: str) -> str:
    title = _report_title(experiment_name)
    cleaned = report.strip()
    if not cleaned:
        return title
    if title in cleaned.splitlines()[0]:
        return cleaned
    return f"{title}\n\n{cleaned}"

def _build_user_prompt(experiment_name: str, metric_key: str, top_runs: list[RunSummary]) -> str:
    payload = {
        "experiment_name": experiment_name,
        "metric_key": metric_key,
        "top_runs": [run.model_dump() for run in top_runs],
    }
    return f"""다음 상위 실험 데이터를 분석해 보고서를 작성하라.
사용하는 용어는 보고서 작성에 적합하게 사용하라.

출력 형식:
0) 제목: "{_report_title(experiment_name)}"
1) 요약(Best 실험 결과)
2) 실험별 강점/약점
3) 성능 차이 해석
4) 다음 실험 제안(하이퍼파라미터 튜닝 관점)

입력 데이터(JSON):
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def generate_report(experiment_name: str, metric_key: str, top_runs: list[RunSummary]) -> tuple[str, str, str]:
    prompt = _build_user_prompt(experiment_name, metric_key, top_runs)
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
