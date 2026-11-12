# 키워드 의도와 메타프롬프팅 비교

측정 시각: 2026-10-20T16:58:15+09:00

채팅 14건. 통과는 의도, 슬롯, 사실이 모두 맞을 때다.

- 키워드: `_heuristic_intent`가 intent만 고른다. 규칙에 없으면 `mlflow_mcp`. 인자는 비운다.
- 메타: JSON 프롬프트가 intent와 arguments를 함께 뽑는다.

## 요약

| 지표 | 키워드 | 메타프롬프팅 |
|---|---:|---:|
| 종합 통과율 | 21.4% (3/14) | 71.4% (10/14) |
| 의도 정확도 | 57.1% (8/14) | 85.7% (12/14) |
| 슬롯 정확도 | 16.7% | 83.3% |
| 사실 정확도 | 28.6% | 71.4% |
| 응답시간 중앙값 | 273 ms | 751 ms |

## 케이스

| 케이스 | 키워드 의도 | 메타 의도 | 키워드 | 메타 |
|---|---|---|---|---|
| 실험 목록 | list_experiments | list_experiments | 통과 | 통과 |
| 최근 실험 | list_recent_experiments | list_recent_experiments | 통과 | 통과 |
| 실험 정보 | get_experiment | get_experiment | 실패 | 통과 |
| run 목록 | get_runs | get_runs | 실패 | 통과 |
| metric 키 | list_metric_keys | list_metric_keys | 실패 | 통과 |
| run 지표값 | mlflow_mcp | get_run_info | 실패 | 통과 |
| run 상세 | get_runs | get_run_info | 실패 | 통과 |
| val_acc top 3 | get_runs | search_runs | 실패 | 실패 |
| 최저 val_loss | get_runs | search_runs | 실패 | 실패 |
| 학습 평가 | get_runs | mlflow_mcp | 실패 | 통과 |
| 보고서 QA | get_runs | search_runs | 실패 | 실패 |
| 없는 실험 | get_experiment | 없음 (HTTP 404) | 통과 | 실패 |
| 후속 1턴 | get_experiment | get_experiment | 실패 | 통과 |
| 후속 2턴 | list_metric_keys | list_metric_keys | 실패 | 통과 |
