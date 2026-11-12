# 실험 QA Agent 측정 결과

측정 시각: 2026-10-20T16:21:08+09:00

정답은 측정 시점의 MLflow 원본이다. 채팅 정확도는 의도, 슬롯, 사실 세 가지가 모두 맞을 때만 통과로 센다.
신규 대화의 응답시간에는 의도 분류, 조회, 첫 턴 세션 제목 생성이 포함된다.

## 요약

- 채팅 케이스: 14건
- 종합 통과율: 71.4%
- 의도 정확도: 85.7%
- 슬롯 정확도: 83.3%
- 사실 정확도: 71.4%
- 채팅 응답시간 중앙값 / p95: 833ms / 1.20s
- 같은 조회의 직접 API 중앙값: 45ms
- 자연어 경로의 중앙 오버헤드: 756ms
- 보고서 생성 시간: 10.36s (gemma-4-26B-A4B-it-UD-Q8_K_XL)
- 보고서 수치 재현율: 100.0%
- 보고서 섹션 커버리지: 83.3%

## 범주별

| 범주 | 건수 | 통과 | 의도 | 슬롯 | 사실 | 응답 중앙값 |
|---|---:|---:|---:|---:|---:|---:|
| experiment_lookup | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 907ms |
| judgment | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 1.55s |
| multiturn | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 511ms |
| report_qa | 1 | 0.0% | 0.0% | - | 0.0% | 720ms |
| robustness | 1 | 0.0% | 0.0% | - | 0.0% | - |
| run_lookup | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 850ms |
| run_search | 2 | 0.0% | 100.0% | 0.0% | 0.0% | 753ms |

## 케이스

| ID | 통과 | 의도 | 사실 | 응답시간 | 직접 API | 질문 |
|---|---|---|---|---:|---:|---|
| list_experiments | Y | list_experiments | 100.0% | 961ms | 17ms | 실험 목록 보여줘 |
| list_recent_experiments | Y | list_recent_experiments | 100.0% | 861ms | - | 최근 실험 알려줘 |
| get_experiment | Y | get_experiment | 100.0% | 907ms | 20ms | Timm Hub 실험 정보 알려줘 |
| get_runs | Y | get_runs | 100.0% | 833ms | 48ms | Timm Hub run 목록 |
| list_metric_keys | Y | list_metric_keys | 100.0% | 733ms | 42ms | Timm Hub metric key 알려줘 |
| get_run_metric | Y | get_run_info | 100.0% | 914ms | 197ms | yolo26n-hpo-20 run의 metrics/mAP50B 값은 얼마야? |
| get_run_detail | Y | get_run_info | 100.0% | 868ms | - | Deep Learning Experiment에서 burly-kit-617 run 상세 알려줘 |
| search_top_accuracy | N | search_runs | 0.0% | 776ms | 49ms | Timm Hub에서 metrics.val_acc가 높은 run top 3 보여줘 |
| search_lowest_loss | N | search_runs | 0.0% | 730ms | - | Timm Hub에서 metrics.val_loss가 가장 낮은 run 보여줘 |
| run_judgment | Y | mlflow_mcp | 100.0% | 1.55s | - | yolo26n-exp1 run 학습이 잘 된 편이야? |
| report_qa | N | search_runs | 0.0% | 720ms | - | 보고서에서 val_acc가 가장 높은 run 이름과 점수를 말해줘 |
| missing_experiment | N | None | 0.0% | 417ms | - | 없는실험XYZ 실험 정보 알려줘 |
| multiturn_context | Y | list_metric_keys | 100.0% | 450ms | - | metric key 알려줘 |
| multiturn_context_setup | Y | get_experiment | 100.0% | 572ms | - | Timm Hub 실험 정보 알려줘 |

## 실패 상세

- `search_top_accuracy` 기대 의도 `search_runs`, 실제 `search_runs`, HTTP 200. 슬롯 불일치: ['experiment_name', 'count']. 사실 불일치: ['top3_names_in_order', 'top3_scores', 'names_in_answer'].
  답변 앞부분: 조회할 실험이 없습니다.
- `search_lowest_loss` 기대 의도 `search_runs`, 실제 `search_runs`, HTTP 200. 슬롯 불일치: ['experiment_name']. 사실 불일치: ['lowest_run_first', 'lowest_loss_value'].
  답변 앞부분: 조회할 실험이 없습니다.
- `report_qa` 기대 의도 `report_qa`, 실제 `search_runs`, HTTP 200. 슬롯 불일치: -. 사실 불일치: ['best_run', 'best_score'].
  답변 앞부분: 조회할 실험이 없습니다.
- `missing_experiment` 기대 의도 `get_experiment`, 실제 `None`, HTTP 404. 슬롯 불일치: -. 사실 불일치: ['http_ok'].
  답변 앞부분: 실험을 찾을 수 없습니다: 없는실험XYZ

## 보고서

- 생성 시간: 10.36s
- 통과: 아니오
- 수치 재현율: 100.0%
