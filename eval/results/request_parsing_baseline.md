# 요청 해석 개선 — 평가 질문과 현재(개선 전) 답변

- 측정일: 2026-11-11, 모델 `Qwen3.6-35B-A3B`(chat), MLflow `127.0.0.1:2522`
- 기준선: **HEAD `3db4d47`**. 분석 문서의 "최근 실험은 항상 limit=5" 상태와 같은 코드다.
- 참고 비교: 커밋하지 않은 작업 트리(최근 실험 `limit/order_by/response_mode` 부분 수정본)도 1회 실행했다.
- 실행: `eval/compare_request_parsing.py`가 `/chat/ask`에 질문을 보내고 파서 출력, 실제 조회 함수와 인자, 답변 전문을 기록한다.
  - 원본 결과: `eval/results/request_parsing_head.json`, `request_parsing_head_r2.json`, `request_parsing_head_r3.json`, `request_parsing_worktree.json`
  - 질문 정의: `eval/request_parsing_cases.py`
- MCP는 조회 도구만 열었다. D03(삭제 요청)으로 실제 삭제가 일어나지 않게 하기 위해서다.
- 반복성: HEAD를 3회 실행했다. **파서 출력은 31개 케이스 모두 3회 동일**했다(temperature 0). LLM이 답변을 생성하는 경로(A10, A11, C01, C03, C04, D01, D03)는 문장이 매번 달랐다. 아래 표는 1회차 기준이며, 결론이 바뀐 경우에만 따로 적었다.

판정 기준: 요청의 의미 차이(대상, 개수, 기간, 필터, 정렬, 답변 범위)가 **조회와 답변에 모두** 반영됐는가.
✅ 반영 · △ 조회는 맞으나 답변 범위가 다르거나 답변 품질에 문제가 있음 · ❌ 조회 또는 결과가 틀림

## 1. 요약

| | ✅ | △ | ❌ |
|---|---|---|---|
| HEAD(기준선) | 5 | 5 | 21 |
| 작업 트리(참고, 1회) | 9 | 6 | 16 |

작업 트리 버전에서 바뀐 것: A02~A05·M3-1(최근 실험 개수·정렬·상세)이 고쳐졌다. 반면 A10은 퇴보했다(요약 요청인데 전체 상세 표 출력). B01·M1-1은 실험명이 빠지거나 run_name 칸에 들어가는 새 실패가 생겼다.

## 2. 케이스별 결과 (HEAD)

### A. 개수·답변 범위

| ID | 질문 | 기대 | 파서 출력 → 실행 | 현재 답변 | 판정 |
|---|---|---|---|---|---|
| A01 | 최근 실험 알려줘 | 최신순 목록 | `list_recent_experiments {}` → `limit=5` | 최근 실험 5개 이름 | ✅ |
| A02 | 가장 최근 실험 하나 알려줘 | 1개: ultralytics-yolo-hpo | `{top_k:1}` → **`limit=5`** | A01과 똑같은 5개 목록 | ❌ P3 |
| A03 | 최근 실험 3개만 보여줘 | 3개 | `{top_k:3}` → **`limit=5`** | 5개 목록 | ❌ P2·P3 |
| A04 | 가장 오래된 실험 뭐야? | Default(오름차순 1개) | `{ascending:true, top_k:1}` → **`limit=5`, 최신순** | 최근 실험 5개(**정반대 결과**) | ❌ P3 |
| A05 | 가장 최근 실험의 상세 정보 알려줘 | ultralytics-yolo-hpo 상세 | `get_experiment {}` → 조회 없음 | "조회할 실험이 없습니다." | ❌ P1 |
| A06 | 실험이 총 몇 개야? | 9개 | `list_experiments` | "실험 목록(9개)" + 9개 이름 전체 | △ P5 |
| A07 | ultralytics-yolo-hpo 실험의 run 이름만 나열해줘 | 이름 20개 | `get_runs` | Name/Score/Status/Date/Time 표 20행(Score 전부 `-`) | △ P5 |
| A08 | yolo26n-hpo-20 run의 metrics/mAP50B 값만 알려줘 | 0.9017 | `get_run_info {metric_key}` → `find_runs` | "`metrics/mAP50B`는 0.901733입니다." | ✅ |
| A09 | yolo26n-hpo-20 run 상세 알려줘 | 전체 상세 | `get_run_info` → `find_runs` | 기본 정보 + 파라미터·metric 표 | ✅ |
| A10 | burly-kit-617 run 핵심만 요약해줘 | 핵심 몇 줄 | `mlflow_mcp {run_name}` → run 해석 LLM | 2문장 요약(MLP, SGD, val_accuracy 77.95) | ✅ ※작업 트리에서는 전체 표 출력 ❌ |
| A11 | carefree-moth-973 run 학습 잘 된 편이야? | 결론 + 근거 metric | `mlflow_mcp` → run 해석 LLM | 결론 먼저 + 근거 인용 | △ 결론이 실행마다 다름(①"성능이 낮은 상태" ②"잘 되었다" ③"보통 수준"). 비교 기준(같은 실험의 다른 run)이 없음 |

### B. 정렬·개수·필터

| ID | 질문 | 기대 | 파서 출력 → 실행 | 현재 답변 | 판정 |
|---|---|---|---|---|---|
| B01 | Timm Hub에서 val_acc 가장 높은 run 2개 알려줘 | capricious-moth-486, mysterious-yak-903 | `search_runs {Timm Hub, val_acc, top_k:2}` → metric 존재 검사 | "**val_acc가 없습니다.**" | ❌ 실행 버그(아래 N1) |
| B02 | Hyperparameter Tuning Experiment에서 error가 가장 낮은 run 3개 | trial_21, trial_15, trial_26 | `search_runs {error, top_k:3, asc}` — **실험명 누락** | "조회할 실험이 없습니다." | ❌ P2 |
| B03 | ultralytics-yolo-hpo 실험에서 가장 최근 run 3개 | hpo-20, 19, 18 | `search_runs {top_k:3}` — 시간순 개수를 top_k로 표현 | "top_k 검색에는 metric_key 선택이 필요합니다." | ❌ P2 |
| B04 | Hyperparameter Tuning Experiment에서 실패한 run 보여줘 | trial_2, study | `search_runs` + 필터 `status:FAILED` → metric 존재 검사 | "**None가 없습니다.** … metric 키 목록을 조회해 주세요." | ❌ P7 |
| B05 | ultralytics-yolo-hpo에서 최근 30일 안에 돌린 run 몇 개야? | 20개 | `{top_k:3}`(**질문에 없는 값**) + `recent_days:30` | "top_k 검색에는 metric_key 선택이 필요합니다." | ❌ P2·P5 |
| B06 | Timm Hub에서 f1_score 높은 run 3개 | f1_score 없음 안내 | `search_runs {f1_score, top_k:3}` — **실험명 누락** | "조회할 실험이 없습니다." | ❌ P6 |

### C. 복합 요청

| ID | 질문 | 기대 | 파서 출력 → 실행 | 현재 답변 | 판정 |
|---|---|---|---|---|---|
| C01 | Timm Hub에서 val_acc 기준 최고 run과 최저 run을 비교해줘 | capricious-moth-486 vs loud-gnat-654 | `mlflow_mcp {run_name:"val_acc"}`(오분류) → MCP 6회 호출 | 정답 run 2개를 비교했으나 "capricious가 훨씬 낮음, *참고: 실제 데이터상으론 capricious가 더 높음*"처럼 앞뒤가 모순되는 문장 포함 | △ P1·P6 |
| C02 | 가장 최근 실험에서 mAP50B가 제일 높은 run 알려줘 | yolo26n-hpo-09 | `search_runs {mAP50B, top_k:1}` — "가장 최근 실험"을 표현할 필드 없음 | "조회할 실험이 없습니다." | ❌ P1·P6 |
| C03 | yolo26n-hpo-09와 yolo26n-hpo-20의 mAP50B 차이 알려줘 | 0.9062, 0.9017, 차이 0.0044 | `mlflow_mcp {run_name: hpo-09}`(두 번째 run 누락) → MCP 5회 | 두 값과 차이 -0.00444 | ✅ (MCP가 메워 줌) |
| C04 | 보고서 기준으로 최근 run 두 개를 비교해줘 (report_text 첨부) | 보고서 근거: hpo-20 vs hpo-19 | `mlflow_mcp {}` → **보고서 무시**, MCP로 임의 실험 조회 | ①MLflow Quickstart ②Deep Learning Experiment(날짜를 `2024-06-12`로 지어냄) ③yolo-hpo — 매번 다른 실험을 비교하면서 "보고서 기반"이라고 답함 | ❌ P1 |

### D. 모호·미지원

| ID | 질문 | 기대 | 파서 출력 → 실행 | 현재 답변 | 판정 |
|---|---|---|---|---|---|
| D01 | 그 run 성능 어때? (새 대화) | 대상 run 확인 질문 | `mlflow_mcp {}` | ①"지시사항에 따르면 … search_experiments() 호출"처럼 **내부 지시문과 계획을 노출** ②확인 질문 ③**빈 답변** | ❌ P6 |
| D02 | 성능 제일 좋은 run 알려줘 | 실험·metric 확인 질문 | `search_runs {}` | "조회할 실험이 없습니다." (무엇이 부족한지 말하지 않음) | ❌ P6 |
| D03 | ultralytics-yolo-hpo 실험 삭제해줘 | 조회 전용이라고 안내 | `mlflow_mcp` → MCP 조회 2회 | 삭제 기능이 없다고 안내하면서 `mlflow experiments delete -x 9` 명령을 제시. 3회차는 "삭제 기능이 포함되어 있습니다"라고 했다가 바로 "조회만 지원"이라고 하는 모순 | △ P6 |

### M. 멀티턴(조건 유지·변경·해제)

| ID | 질문 | 기대 | 파서 출력 → 실행 | 현재 답변 | 판정 |
|---|---|---|---|---|---|
| M1-1 | Timm Hub run 중 val_acc 상위 2개 보여줘 | 2개 | B01과 같음 | "val_acc가 없습니다." | ❌ N1 |
| M1-2 | 전체 run 보여줘 | Timm Hub 유지, 개수 해제 → 4개 | `search_runs {}` → 이전 턴의 `metric_key=val_acc`·`top_k=2`를 라우터가 이어받음 | "val_acc가 없습니다." | ❌ P4 |
| M1-3 | 그 중에 val_loss가 제일 낮은 건? | Timm Hub 유지, val_loss 오름차순 1개 | `{Timm Hub, val_loss, top_k:1, asc}` (해석은 정확) | "val_loss가 없습니다." | ❌ N1 |
| M2-1 | Hyperparameter Tuning Experiment에서 실패한 run 보여줘 | trial_2, study | B04와 같음 | "None가 없습니다." | ❌ P7 |
| M2-2 | 완료된 건 몇 개야? | 실험 유지, FINISHED, 33개 | 실험 이어받음 + `status:FINISHED` (해석은 정확) | "None가 없습니다." | ❌ P7·P5 |
| M3-1 | 가장 최근 실험 하나 알려줘 | 1개 | A02와 같음 | 5개 목록 | ❌ P3 |
| M3-2 | 그 실험 run 중 최근 3개만 보여줘 | hpo-20, 19, 18 | `get_runs {ultralytics-yolo-hpo, top_k:3}` → **top_k 무시** | run 20개 표 전체 | ❌ P2·P3 (실험을 이어받은 것도 5개 목록의 첫 줄을 고른 우연) |

## 3. 이번 측정에서 새로 확인한 것

분석 문서(P1~P7)에 더해 실제 채팅에서 확인된 항목이다.

- **N1. metric key 접두사 불일치(코드 버그).** `list_run_metrics()`는 `metrics.val_acc` 형태를 반환한다. 파서는 사용자 표현 그대로 `val_acc`를 넣는다. 그래서 [chat_service.py:1034](../../backend/app/services/chat_service.py#L1034)의 존재 검사가 metric을 지정한 run 검색에서 **항상 실패**한다(B01, M1-1, M1-3). `get_run_info` 쪽은 `_metric_column_name()`으로 접두사를 붙이는데, `search_runs`에는 이 처리가 없다.
- **N2. `search_runs`에서 실험명이 자주 빠진다.** B02, B06, C02와 작업 트리의 B01에서 질문에 실험명이 있어도 추출하지 않았다. 작업 트리의 M1-1은 실험명을 `run_name`에 넣었다. 프롬프트의 `search_runs` 예시에 `experiment_name`이 없는 것이 원인으로 보인다.
- **N3. 파서가 질문에 없는 값을 만든다.** B05에서 `top_k:3`이 나왔는데, 앞 예시 문장("top 5")의 영향으로 보인다.
- **N4. `report_text`가 있어도 보고서 경로로 가지 않는다.** C04는 보고서가 첨부됐는데도 MCP로 갔다. 그 결과 매번 다른 실험을 "보고서 기반"이라며 비교했다. 근거를 사칭하는 셈이라 위험도가 가장 높다.
- **N5. MCP 에이전트 답변에 내부 추론과 지시문이 노출되고 빈 답변도 나온다**(D01). 확인 질문으로 이어지는 경로가 따로 없다.

## 4. 원인 분류 (개선 범위를 정하기 위한 표)

| 원인 | 해당 케이스 | 프롬프트로 해결 | 코드 수정 필요 |
|---|---|---|---|
| 개수·정렬 필드 부족, top_k와 limit 혼용 | A02~A04, B03, B05, M3 | 필드 정의·예시 | 실행 함수가 limit·정렬을 받아야 함(`get_runs`, `search_runs`) |
| 추출한 값이 실행에서 무시됨 | A02~A04, M3-2 | — | ✔ |
| 답변 범위(개수만, 이름만, 요약, 상세) | A06, A07, A10, B05, M2-2 | 출력 범위 필드 | 범위별 답변 생성 |
| 시간으로 대상을 고르는 요청, 복합 요청 | A05, C01, C02 | 대상 선택과 작업 분리 | 연쇄 조회 실행 |
| 실험명 누락·오분류 | B02, B06, C02, (C01) | 예시 보강 | — |
| metric 접두사 | B01, M1 | — | ✔ (N1) |
| metric 없는 필터 검색 | B04, B05, M2 | — | ✔ (P7) |
| 조건 이어받기·해제 | M1-2 | 유지·변경·해제 규칙 | 라우터의 `last_meta` 이어받기 수정 |
| 보고서 근거 경로 | C04 | 근거 자료 필드 | report_text가 있으면 보고서 경로로 보내기 |
| 모호·미지원 요청 | D01~D03 | 미해결 항목·지원 범위 필드 | 확인 질문과 안내 분기 |

## 5. 다음 단계 전에 확인할 것

1. **기준선 코드.** HEAD를 기준선으로 할지, 커밋되지 않은 작업 트리(최근 실험 부분 수정본)를 기준선으로 할지. 개선 작업은 작업 트리 위에서 할 예정이다.
2. **범위.** "프롬프트 개선"만 하면 N1·P7 같은 실행 버그 때문에 B01, B04, M1, M2가 프롬프트를 고쳐도 계속 실패해 효과가 보이지 않는다. 구조화된 요청을 실행까지 연결하는 코드 수정(인자 검증, 실행 분기, 도구 인자)을 함께 할지.
3. **질문 세트.** 31개(단일 24개 + 멀티턴 3세션 7턴) 그대로 갈지, 빼거나 더할 질문이 있는지.
