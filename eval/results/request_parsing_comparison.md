# 요청 해석 개선 — 개선 전/후 비교

- 측정일: 2026-11-11, 모델 `Qwen3.6-35B-A3B`, MLflow `127.0.0.1:2522`
- 개선 전: HEAD `b9d9bca`. 개선 후: 현재 작업 트리(미커밋 변경 포함, 커밋하지 않음)
- 실행: `eval/compare_request_parsing.py`가 FastAPI `TestClient`로 `/chat/ask`를 호출한다. 같은 DB, 같은 MLflow, 같은 LLM을 쓴다.
- 평가 세트 36개: 기존 31개에 5개를 추가했다(C05, C06, M2-3, M4-1, M4-2).
- 홀드아웃 11개(`eval/request_parsing_holdout.py`): 개선을 마친 뒤 처음 만들어 두 버전에서 한 번씩만 실행했다.

함께 볼 문서

| 문서 | 내용 |
|---|---|
| [request_parsing_prompts.md](request_parsing_prompts.md) | 개선 전/후 프롬프트 전문 |
| [request_parsing_answers.md](request_parsing_answers.md) | 케이스별 해석 결과, 호출한 도구, 답변 전문(개선 전/후 나란히) |
| 커밋 `fb23f42` | 코드 변경 전체: `git diff b9d9bca fb23f42 -- backend/` |
| [request_parsing_baseline.md](request_parsing_baseline.md) | 개선 전 기준선 분석(어제 작성) |

## 1. 결과 요약

판정 기준은 기준선 문서와 같다. 요청의 의미 차이가 조회와 답변에 모두 반영됐는지를 본다.
✅ 반영 · △ 조회는 맞으나 답변 범위나 품질에 문제 · ❌ 조회 또는 결과가 틀림

| | ✅ | △ | ❌ |
|---|---|---|---|
| 평가 세트 36 — 개선 전 | 5 | 5 | 26 |
| 평가 세트 36 — 개선 후 | **35** | 1 | **0** |
| 홀드아웃 11 — 개선 전 | 5 | 2 | 4 |
| 홀드아웃 11 — 개선 후 | **11** | 0 | **0** |

| 지표(평가 세트 31개 × 3회) | 개선 전 | 개선 후 |
|---|---|---|
| 파서 출력이 3회 모두 같은 케이스 | 31/31 | 31/31 (36개 기준 36/36) |
| 코드가 만든 답변이 3회 모두 같은 케이스 | — | 33/36 (다른 3개는 LLM이 쓰는 요약·평가·보고서 답변) |
| MCP로 넘어간 비율 | 22% | 0% |
| 질문당 LLM 호출 | 1.35회 | 1.10회 |
| 응답시간 중앙값 / p90 / 최대 | 1.3초 / 11.5초 / 19.3초 | 1.8초 / 2.7초 / 4.5초 |

응답시간 중앙값은 0.4초 늘었다. 해석 프롬프트가 길어졌기 때문이다(예시 26개). 대신 MCP 경유가 없어져 느린 꼬리가 사라졌다.

## 2. 무엇을 바꿨나

### 2-1. 구조: "intent 하나 고르기" → "요청 구조 해석"

```
개선 전  질문 → intent 9개 중 1개 + 인자 6개 → (run 검색이면) 필터 파서 2차 호출 → intent별 실행 함수
개선 후  질문 + 직전 요청 구조 → 요청 구조 JSON 1회 → 검증 → 대상 확정 → 작업·답변 범위별 실행
```

요청 구조의 필드(분석 문서 4장의 항목을 그대로 옮김)

| 분석 문서 항목 | 요청 구조 필드 |
|---|---|
| 대상 | `target`: experiment, run, metric_key, report, other |
| 작업 | `action`: list, count, lookup, compare, summarize, evaluate, unsupported |
| 선택 조건 | `experiment`, `experiment_pick`(latest, oldest, latest_updated), `runs[]`, `metric`, `filters{recent_days, status, user, tag}` |
| 정렬과 개수 | `sort_by`(metric, time, update_time), `order`, `limit`(정수 또는 "all"), `pick`(both_ends) |
| 응답 요구 | `output`(names, count, value, brief, table, full), `fields`(params, metrics, status, time, id) |
| 대화 맥락 | `inherited[]`(이어받은 필드), `cleared[]`(해제한 필드) |
| 미해결 항목 | `unresolved[]`(experiment, run, metric), `unsupported_reason` |

### 2-2. 문제별 대응

| 문제 | 대응 | 위치 |
|---|---|---|
| P1 의도 기준이 섞임 | 대상(`target`)·작업(`action`)·근거(`report`)를 분리했다. "가장 최근 실험에서 mAP50B 최고 run"은 `experiment_pick` + `sort_by` + `limit`로 한 번에 표현한다. | `chat_request.py` `REQUEST_PARSING_PROMPT` |
| P2 조건 필드 부족, top_k 혼용 | `limit`(개수)과 `sort_by`(기준)를 분리했다. 기간·상태·태그는 같은 요청 안의 `filters`로 넣었다(2차 파서 제거). | 같은 곳 |
| P3 추출값을 실행에서 무시 | 실행 함수가 요청 구조를 직접 받는다. 실험·run 모두 limit·정렬·필터를 적용한다. | `_answer_experiments`, `_select_runs` |
| P4 이어받기 범위 | 직전 assistant metadata에 `request`·`resolved`를 저장하고 다음 턴 파서에 넘긴다. 대상은 지시어가 있을 때 이어받고, 조건은 "그 중에"일 때만 유지하며, "전체"·"조건 빼고"면 해제한다. 라우터가 이전 턴의 `top_k`·`ascending`을 무조건 채우던 코드는 삭제했다. | `chat_router.py`, `repository_service.get_last_assistant_context`, 프롬프트 규칙 6 |
| P5 조회와 답변 범위 미분리 | `output`·`fields`에 따라 개수만, 이름만, 값 하나, 표, 상세, 특정 항목만으로 답한다. 요약·평가는 별도 프롬프트로 분리했다. | `_answer_run_list`, `_answer_run_records`, `_run_fields_markdown` |
| P6 복합·모호·미지원 구분 | 복합 요청은 실험 선택 → run 순위 → 상세/비교로 연쇄 실행한다. 정보가 부족하면 무엇이 부족한지 묻고 선택지를 제시한다. 조회가 아닌 작업은 지원 범위를 안내한다(MCP로 보내지 않음). 보고서 질문은 보고서 경로로 보낸다. | `_clarify`, `_answer_unsupported`, `_answer_report`, `_answer_compare` |
| P7 실행 문제 | metric 없는 기간·상태 검색에서 metric 검사를 없앴다. metric 값이 없는 run은 순위에서 빼고 그 수를 알린다. 시간 정렬은 표시용 Date 문자열이 아니라 `start_time`(초 단위)을 쓴다. | `_select_runs`, `chat_service._to_epoch_seconds` |
| N1 metric 접두사 불일치 | `val_acc`, `mAP50B`, `recall`처럼 말해도 실제 컬럼(`metrics.val_acc`, `metrics.metrics/mAP50B`, `metrics.metrics/recallB`)으로 바꾼다. 후보가 여럿이면 되묻고, 없으면 사용 가능한 metric을 보여 준다. | `resolve_metric_column` |
| N2 실험명 누락·오분류 | 프롬프트 예시마다 실험명을 넣었다. 실험명이 `runs`에 들어가거나 run 이름이 `experiment`에 들어가면 코드가 바로잡는다. | `_fix_misplaced_targets` |
| N3 없는 값 생성 | 규칙 2("limit, metric, sort_by, filters를 추측하지 않는다")를 추가했다. 개수를 말하지 않아 기본값을 쓰면 그렇다고 알린다. | 프롬프트, `_answer_experiments` |
| N4 보고서 무시·근거 사칭 | `target=report`면 보고서 경로로만 보낸다. 보고서 QA 프롬프트가 "근거 없음"을 지나치게 쓰던 문제(부분 근거 규칙)도 고쳤다. | `_answer_report`, `chat_service.REPORT_QA_SYSTEM_PROMPT` |
| N5 MCP 답변의 내부 지시문 노출 | MCP 에이전트에 답변 규칙(내부 지시·계획 출력 금지, 우회 방법 안내 금지 등)을 추가했다. 평가 세트에서는 MCP로 가는 질문이 없어져 이번 측정으로는 검증하지 못했다. | `chat_service.run_with_mcp` |

값의 출처 표시: run·metric 목록 답변 첫 줄에 `적용 조건: 실험 \`Timm Hub\`(이전 대화) · 정렬 \`val_acc\` 높은 순 · 2개`처럼 조건과 출처를 보여 준다. 출처는 질문, 이전 대화, UI 선택, 가장 최근 실험, 기본값이다. 해제된 조건은 `이전 조건(기간·상태 등) 해제`로 표시한다.

### 2-3. 파일별 변경

| 파일 | 변경 |
|---|---|
| `backend/app/services/chat_request.py` | **새 파일**(1,340줄). 해석 프롬프트, 정규화, 대상 확정, 실행·답변, 요약·평가 프롬프트 |
| `backend/app/services/chat_service.py` | 대체된 코드 삭제: 기존 intent 프롬프트, 검색 필터 파서, `classify_intent`, `answer_experiment_question`, `answer_run_question` 등. 남긴 것: 공통 포맷터, 최근 실험 답변, 보고서 QA(`request_scope` 인자 추가, 프롬프트 수정), MCP(규칙 추가), `_to_epoch_seconds`(Timestamp 처리) |
| `backend/app/api/chat_router.py` | `/chat/ask`를 `parse_request` → `answer_request` → (None이면) MCP 흐름으로 교체. intent 정규식 분기와 `last_meta` 이어받기 삭제 |
| `backend/app/services/repository_service.py` | `get_last_assistant_context`가 `request`·`resolved`도 돌려준다 |
| `backend/app/services/__init__.py` | 삭제된 함수의 export를 제거했다 |
| `eval/compare_request_parsing.py` | 개선 후 파서 기록, `--cases` 옵션 |
| `eval/request_parsing_cases.py` | 5개 추가, B02 정답 수정(아래 4장) |
| `eval/request_parsing_holdout.py` | 홀드아웃 11개(새 파일) |

작업 트리에 원래 있던 미커밋 변경(`mlflow_service.py`, `schemas.py`, `frontend/app_2.py` 등)은 건드리지 않았다. 최근 실험 부분 수정(`_answer_recent_experiments`)은 새 흐름에서 그대로 재사용한다.

## 3. 케이스별 판정

답변 전문은 [request_parsing_answers.md](request_parsing_answers.md)에 있다. 아래는 핵심만 요약했다.

### A. 개수·답변 범위

| ID | 질문 | 개선 전 | 개선 후 |
|---|---|---|---|
| A01 | 최근 실험 알려줘 | ✅ 5개 목록 | ✅ 5개 목록 + "기본 5개" 안내 |
| A02 | 가장 최근 실험 하나 알려줘 | ❌ 5개 목록 | ✅ `ultralytics-yolo-hpo` 1개(ID·생성시각) |
| A03 | 최근 실험 3개만 보여줘 | ❌ 5개 | ✅ 3개 |
| A04 | 가장 오래된 실험 뭐야? | ❌ 최근 5개(정반대) | ✅ `Default` |
| A05 | 가장 최근 실험의 상세 정보 | ❌ "조회할 실험이 없습니다" | ✅ ultralytics-yolo-hpo 상세 |
| A06 | 실험이 총 몇 개야? | △ 9개 + 이름 전체 | ✅ "9개" |
| A07 | run 이름만 나열해줘 | △ 5열 표(Score 전부 `-`) | ✅ 이름 20개만 |
| A08 | metrics/mAP50B 값만 | ✅ | ✅ |
| A09 | run 상세 | ✅ | ✅ |
| A10 | burly-kit-617 핵심만 요약 | ✅ 2문장 | ✅ 3줄 요약 |
| A11 | carefree-moth-973 학습 잘 된 편이야? | △ 결론이 3회 모두 다름("낮다"/"잘 됐다"/"보통") | △ 결론 일관("같은 실험 3개 중 상위권, val_accuracy 2위, 차이 작음")·근거 순위 정확. 단 3회 중 2회 첫 문장에 "과적합 가능성"과 "일반화 양호"가 함께 나와 문장이 매끄럽지 않음 |

### B. 정렬·개수·필터

| ID | 질문 | 개선 전 | 개선 후 |
|---|---|---|---|
| B01 | Timm Hub val_acc 높은 run 2개 | ❌ "val_acc가 없습니다"(N1) | ✅ capricious-moth-486, mysterious-yak-903 |
| B02 | error 낮은 run 3개 | ❌ 실험명 누락 | ✅ trial_21, trial_28, trial_15 + "error 값 없는 run 3개 제외" |
| B03 | 가장 최근 run 3개 | ❌ "metric 필요" 오류 | ✅ hpo-20, 19, 18(시각순) |
| B04 | 실패한 run | ❌ "None가 없습니다" | ✅ trial_2, study |
| B05 | 최근 30일 run 몇 개 | ❌ 없는 top_k 생성 + 오류 | ✅ "20개" |
| B06 | f1_score 높은 run 3개 | ❌ 실험명 누락 | ✅ "f1_score 없음, 사용 가능: train_acc, train_loss, val_acc, val_loss" |

### C. 복합 요청

| ID | 질문 | 개선 전 | 개선 후 |
|---|---|---|---|
| C01 | val_acc 최고·최저 run 비교 | △ MCP 6회 호출, 모순 문장 | ✅ 최고·최저 표 + 차이 +0.0109 + "파라미터 동일" |
| C02 | 가장 최근 실험에서 mAP50B 최고 run | ❌ 실험 없음 | ✅ yolo26n-hpo-09(0.90617), 조건에 "(가장 최근 실험)" 표시 |
| C03 | hpo-09와 hpo-20 mAP50B 차이 | ✅ (MCP 5회) | ✅ 표 + 차이 +0.00444(조회 2회) |
| C04 | 보고서 기준 최근 run 두 개 비교 | ❌ 보고서 무시, 매번 다른 실험을 "보고서 기반"이라며 비교 | ✅ 3회 모두 보고서 근거로 hpo-20 vs hpo-19, "변동 범위 안" |
| C05 *(추가)* | val_accuracy 최고 run의 파라미터만 | ❌ "val_accuracy가 없습니다" | ✅ burly-kit-617 파라미터 표만 |
| C06 *(추가)* | 보고서 기준 GPU 메모리 최소 run | ❌ "조회할 실험이 없습니다" | ✅ "리포트에 근거 없음"(지어내지 않음) |

### D. 모호·미지원

| ID | 질문 | 개선 전 | 개선 후 |
|---|---|---|---|
| D01 | 그 run 성능 어때?(새 대화) | ❌ 내부 지시문 노출 / 빈 답변 | ✅ "어떤 run인가요? run 이름이나 run ID로" |
| D02 | 성능 제일 좋은 run | ❌ "조회할 실험이 없습니다" | ✅ 실험(목록 제시)과 기준 metric을 되물음 |
| D03 | 실험 삭제해줘 | △ CLI 삭제 명령 안내, 모순 문장 | ✅ "조회만 합니다" + 가능한 작업 안내(도구 호출 0회) |

### M. 멀티턴

| ID | 질문 | 개선 전 | 개선 후 |
|---|---|---|---|
| M1-1 | Timm Hub val_acc 상위 2개 | ❌ N1 | ✅ 2개 |
| M1-2 | 전체 run 보여줘 | ❌ 이전 metric·top_k가 남아 오류 | ✅ 4개 전부, "실험(이전 대화)" |
| M1-3 | 그 중에 val_loss 제일 낮은 건? | ❌ N1 | ✅ capricious-moth-486(0.1245) |
| M2-1 | 실패한 run | ❌ None 오류 | ✅ 2개 |
| M2-2 | 완료된 건 몇 개야? | ❌ None 오류(해석은 맞음) | ✅ "33개", 상태 FINISHED로 변경 |
| M2-3 *(추가)* | 상태 조건 빼고 전체는? | ❌ None 오류 | ✅ "35개", "이전 조건 해제" 표시 |
| M3-1 | 가장 최근 실험 하나 | ❌ 5개 | ✅ 1개 |
| M3-2 | 그 실험 run 중 최근 3개 | ❌ top_k 무시, 20개 표 | ✅ 3개, "실험(이전 대화)" |
| M4-1 *(추가)* | hpo-09 mAP50B 값 | ❌ "mAP50B metric이 없습니다"(N1) | ✅ 0.90617 |
| M4-2 *(추가)* | 같은 run의 recall은? | ❌ "recall metric이 없습니다" | ✅ run 이어받음, recallB 0.661325 |

### 홀드아웃(개선 후 처음 실행)

| ID | 질문 | 개선 전 | 개선 후 |
|---|---|---|---|
| H01 | Timm Hub에 run 몇 개 있어? | △ 4개 + 전체 표 | ✅ "4개" |
| H02 | mAP50-95B 상위 3개 이름만 | ❌ N1 | ✅ hpo-09, 08, 05 |
| H03 | 가장 오래된 실험 2개 | ❌ 최근 5개 | ✅ Default, OpenAI_Test_Project |
| H04 | trial_21 error 값 | ✅ | ✅ |
| H05 | Timm Hub metric 종류 | ✅ | ✅ |
| H06 | skittish-dove-494 설정값만 | △ 전체 상세 | ✅ 파라미터만 |
| H07 | ultralytics-yolo 실험 정보 | ✅ | ✅ (-hpo와 혼동 없음) |
| H08-1 | Deep Learning Experiment run 목록 | ✅ | ✅ |
| H08-2 | 거기서 val_accuracy 제일 높은 건? | ❌ N1 | ✅ burly-kit-617 |
| H09 | run 이름 바꿔줘 | ❌ "run 정보에 근거 없음" | ✅ 지원 범위 안내 |
| H10 | 두 run 비교해줘 | ❌ 빈 답변 | ✅ metric 비교 표 |

## 4. 측정 중 바로잡은 것

- **B02 정답 오류:** 기준선 문서에는 정답을 trial_21, trial_15, trial_26으로 적었다. 실제로는 trial_28(0.2457)이 trial_15(0.2465)보다 낮다. 정답을 trial_21, trial_28, trial_15로 고쳤다. 개선 전 판정(❌, 실험명 누락)은 바뀌지 않는다.
- **보고서 QA 프롬프트:** 개선 중에 보고서 경로로 보내도 C04가 "리포트에 근거 없음"으로 답하는 것을 발견했다. 기존 규칙 "근거가 없으면 반드시 '리포트에 근거 없음'"이 지나치게 보수적으로 작동했기 때문이다. 기준선에서는 보고서 경로에 도달하지 못해 드러나지 않았다. 부분 근거를 허용하도록 고쳤고, 보고서에 없는 대상(C06, GPU 메모리)은 여전히 "근거 없음"으로 답하는 것을 확인했다.

## 5. 한계와 남은 일

1. **같은 사람이 질문과 수정을 함께 만들었다.** 이를 보완하려고 프롬프트 예시에는 평가 질문의 문장과 실험·run 이름을 쓰지 않았다(resnet-ablation, brave-owl-12 같은 가상 이름 사용). 홀드아웃 11개도 따로 돌렸다. 그래도 사용자님이 직접 만든 질문으로 한 번 더 확인하는 것이 좋다.
2. **실제 서비스 경로는 확인하지 않았다.** 측정은 프로세스 안의 `TestClient`로 했다. 실행 중인 `backend` 컨테이너는 옛 코드다. 반영하려면 `docker compose up -d --build backend`가 필요하다. Streamlit 화면도 확인하지 않았다. 새 intent 값(`compare_runs`, `evaluate_run`, `summarize_run`, `clarify`, `unsupported`)은 프런트엔드에서 캡션으로만 쓰인다. 프런트엔드가 읽는 metadata 키(`experiment_name`, `metric_key`, `top_k`, `run_display_rows`, `experiment_display_row`)는 유지했다.
3. **평가하지 못한 경로가 있다.**
   - UI에서 선택한 실험·metric으로 빈 값을 채우는 경로: 평가 요청이 그 값을 보내지 않는다.
   - MCP 경로(`target=other`): 평가 세트에서 MCP로 가는 질문이 0개가 됐다.
   - 파서가 JSON을 내지 못했을 때 MCP로 넘기는 처리
4. **A11 평가 문장 품질(△):** 근거와 결론은 맞지만, 3회 중 2회 첫 문장에 "과적합 가능성"과 "일반화 양호"가 함께 나온다. 평가 프롬프트를 더 다듬을 여지가 있다.
5. **응답시간 중앙값이 0.4초 늘었다.** 해석 프롬프트의 예시 26개를 줄이면 개선될 수 있다.
6. **기존 평가 스크립트 `eval/compare_meta_prompt.py`는 삭제했다.** intent 분류가 없어져 실행할 수 없었다. 필요하면 `git show fb23f42:eval/compare_meta_prompt.py`로 복구한다.
7. **커밋하지 않았다.** 검토 후 커밋 여부를 정해 주세요.

## 6. 다시 실행하는 법

```bash
PY=~/anaconda3/envs/mlflow310/bin/python
# 개선 후(현재 작업 트리)
$PY eval/compare_request_parsing.py --label after
$PY eval/compare_request_parsing.py --label holdout_after --cases request_parsing_holdout
# 개선 전(HEAD 워크트리)
git worktree add --detach /tmp/head-wt b9d9bca
$PY eval/compare_request_parsing.py --label head --code-root /tmp/head-wt
```
