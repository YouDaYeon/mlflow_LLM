# 요청 해석 개선 — 케이스별 전체 답변(개선 전/후)

개선 전 = HEAD `b9d9bca`(1회차 결과, 추가 케이스는 `head_added`), 개선 후 = 현재 작업 트리(`after` 1회차).
각 케이스: 질문 → 기대 → [개선 전] 해석·호출·답변 → [개선 후] 해석·호출·답변.

## 평가 세트(36)

### A01 — 최근 실험 알려줘

**기대:** 최근 생성 실험 목록(기본 개수, 최신순). 이름 목록 형태.

**개선 전** — 라우팅 `list_recent_experiments`

- 해석: `[{"intent": "list_recent_experiments", "arguments": {}}]`
- 호출: `llm:intent_parser` → `list_recent_experiment_names(limit=5)`

> 최근 실험(5개)
>
> - ultralytics-yolo-hpo
> - ultralytics-yolo
> - Timm Hub
> - My Application
> - Deep Learning Experiment

**개선 후** — 라우팅 `list_recent_experiments`

- 해석: `[{"request": {"target": "experiment", "action": "list", "sort_by": "time", "order": "desc", "output": "names"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `list_recent_experiments(limit=5, order_by='creation_time', ascending=False)`

> 최근 실험(5개)
>
> - ultralytics-yolo-hpo
> - ultralytics-yolo
> - Timm Hub
> - My Application
> - Deep Learning Experiment
>
> (개수를 지정하지 않아 기본 5개를 보여 드렸습니다.)

### A02 — 가장 최근 실험 하나 알려줘

**기대:** 실험 1개만: ultralytics-yolo-hpo. A01과 조회 개수·답변 범위가 달라야 한다.

**개선 전** — 라우팅 `list_recent_experiments`

- 해석: `[{"intent": "list_recent_experiments", "arguments": {"top_k": 1}}]`
- 호출: `llm:intent_parser` → `list_recent_experiment_names(limit=5)`

> 최근 실험(5개)
>
> - ultralytics-yolo-hpo
> - ultralytics-yolo
> - Timm Hub
> - My Application
> - Deep Learning Experiment

**개선 후** — 라우팅 `list_recent_experiments`

- 해석: `[{"request": {"target": "experiment", "action": "list", "experiment_pick": "latest", "limit": 1, "output": "brief"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `list_recent_experiments(limit=1, order_by='creation_time', ascending=False)`

> 가장 최근 실험은 `ultralytics-yolo-hpo`입니다. (ID `9`, 생성 2026-10-14 15:06:38)

### A03 — 최근 실험 3개만 보여줘

**기대:** 최신순 3개: ultralytics-yolo-hpo, ultralytics-yolo, Timm Hub.

**개선 전** — 라우팅 `list_recent_experiments`

- 해석: `[{"intent": "list_recent_experiments", "arguments": {"top_k": 3}}]`
- 호출: `llm:intent_parser` → `list_recent_experiment_names(limit=5)`

> 최근 실험(5개)
>
> - ultralytics-yolo-hpo
> - ultralytics-yolo
> - Timm Hub
> - My Application
> - Deep Learning Experiment

**개선 후** — 라우팅 `list_recent_experiments`

- 해석: `[{"request": {"target": "experiment", "action": "list", "sort_by": "time", "order": "desc", "limit": 3, "output": "table"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `list_recent_experiments(limit=3, order_by='creation_time', ascending=False)`

> 최근 실험(3개)
>
> - `ultralytics-yolo-hpo` · ID `9` · 생성 2026-10-14 15:06:38
> - `ultralytics-yolo` · ID `8` · 생성 2026-10-14 15:05:31
> - `Timm Hub` · ID `6` · 생성 2026-04-08 14:04:40

### A04 — 가장 오래된 실험 뭐야?

**기대:** 생성 오름차순 1개: Default 또는 OpenAI_Test_Project(같은 시각 생성).

**개선 전** — 라우팅 `list_recent_experiments`

- 해석: `[{"intent": "list_recent_experiments", "arguments": {"ascending": true, "top_k": 1}}]`
- 호출: `llm:intent_parser` → `list_recent_experiment_names(limit=5)`

> 최근 실험(5개)
>
> - ultralytics-yolo-hpo
> - ultralytics-yolo
> - Timm Hub
> - My Application
> - Deep Learning Experiment

**개선 후** — 라우팅 `list_recent_experiments`

- 해석: `[{"request": {"target": "experiment", "action": "list", "experiment_pick": "oldest", "output": "brief"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `list_recent_experiments(limit=1, order_by='creation_time', ascending=True)`

> 가장 오래된 실험은 `Default`입니다. (ID `0`, 생성 2026-03-27 10:34:00)

### A05 — 가장 최근 실험의 상세 정보 알려줘

**기대:** ultralytics-yolo-hpo 1개의 상세(ID, 생성시각, artifact 위치 등).

**개선 전** — 라우팅 `get_experiment`

- 해석: `[{"intent": "get_experiment", "arguments": {}}]`
- 호출: `llm:intent_parser`

> 조회할 실험이 없습니다.

**개선 후** — 라우팅 `list_recent_experiments`

- 해석: `[{"request": {"target": "experiment", "action": "lookup", "experiment_pick": "latest", "output": "full"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `list_recent_experiments(limit=1, order_by='creation_time', ascending=False)` → `get_experiment('ultralytics-yolo-hpo')`

> 가장 최근 실험 상세
>
> `ultralytics-yolo-hpo` 실험 정보
>
> | Name | Experiment ID | Lifecycle Stage | Created Date | Created Time |
> |------|----------------|-----------------|--------------|--------------|
> | ultralytics-yolo-hpo | 9 | active | 2026-10-14 | 15:06:38 |
>
> - Artifact Location: `/home/dyyu/MLflow/artifacts/9`
> - Tags: -

### A06 — 실험이 총 몇 개야?

**기대:** 개수만: 9개. 전체 목록 나열은 불필요.

**개선 전** — 라우팅 `list_experiments`

- 해석: `[{"intent": "list_experiments", "arguments": {}}]`
- 호출: `llm:intent_parser` → `list_experiment_names()`

> 실험 목록(9개)
>
> - Deep Learning Experiment
> - Default
> - Hyperparameter Tuning Experiment
> - MLflow Quickstart
> - My Application
> - OpenAI_Test_Project
> - Timm Hub
> - ultralytics-yolo
> - ultralytics-yolo-hpo

**개선 후** — 라우팅 `list_experiments`

- 해석: `[{"request": {"target": "experiment", "action": "count", "output": "count"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()`

> 등록된 실험은 **9개**입니다.

### A07 — ultralytics-yolo-hpo 실험의 run 이름만 나열해줘

**기대:** run 20개의 이름만. 점수·상태·날짜 표는 불필요.

**개선 전** — 라우팅 `get_runs`

- 해석: `[{"intent": "get_runs", "arguments": {"experiment_name": "ultralytics-yolo-hpo"}}]`
- 호출: `llm:intent_parser` → `get_runs('ultralytics-yolo-hpo')`

답변

`ultralytics-yolo-hpo` run **20개** (미리보기 **20개**)

| Name | Score | Status | Date | Time |
|------|-------|--------|------|------|
| yolo26n-hpo-20 | - | FINISHED | 2026-10-14 | 15:11:00 |
| yolo26n-hpo-19 | - | FINISHED | 2026-10-14 | 15:10:48 |
| yolo26n-hpo-18 | - | FINISHED | 2026-10-14 | 15:10:32 |
| yolo26n-hpo-17 | - | FINISHED | 2026-10-14 | 15:10:19 |
| yolo26n-hpo-16 | - | FINISHED | 2026-10-14 | 15:10:06 |
| yolo26n-hpo-15 | - | FINISHED | 2026-10-14 | 15:09:51 |
| yolo26n-hpo-14 | - | FINISHED | 2026-10-14 | 15:09:39 |
| yolo26n-hpo-13 | - | FINISHED | 2026-10-14 | 15:09:24 |
| yolo26n-hpo-12 | - | FINISHED | 2026-10-14 | 15:09:11 |
| yolo26n-hpo-11 | - | FINISHED | 2026-10-14 | 15:08:58 |
| yolo26n-hpo-10 | - | FINISHED | 2026-10-14 | 15:08:43 |
| yolo26n-hpo-09 | - | FINISHED | 2026-10-14 | 15:08:30 |
| yolo26n-hpo-08 | - | FINISHED | 2026-10-14 | 15:08:18 |
| yolo26n-hpo-07 | - | FINISHED | 2026-10-14 | 15:08:06 |
| yolo26n-hpo-06 | - | FINISHED | 2026-10-14 | 15:07:53 |
| yolo26n-hpo-05 | - | FINISHED | 2026-10-14 | 15:07:38 |
| yolo26n-hpo-04 | - | FINISHED | 2026-10-14 | 15:07:24 |
| yolo26n-hpo-03 | - | FINISHED | 2026-10-14 | 15:07:10 |
| yolo26n-hpo-02 | - | FINISHED | 2026-10-14 | 15:06:56 |
| yolo26n-hpo-01 | - | FINISHED | 2026-10-14 | 15:06:38 |



**개선 후** — 라우팅 `get_runs`

- 해석: `[{"request": {"target": "run", "action": "list", "experiment": "ultralytics-yolo-hpo", "output": "names"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('ultralytics-yolo-hpo')`

> 적용 조건: 실험 `ultralytics-yolo-hpo`
>
> run 20개
>
> - yolo26n-hpo-20
> - yolo26n-hpo-19
> - yolo26n-hpo-18
> - yolo26n-hpo-17
> - yolo26n-hpo-16
> - yolo26n-hpo-15
> - yolo26n-hpo-14
> - yolo26n-hpo-13
> - yolo26n-hpo-12
> - yolo26n-hpo-11
> - yolo26n-hpo-10
> - yolo26n-hpo-09
> - yolo26n-hpo-08
> - yolo26n-hpo-07
> - yolo26n-hpo-06
> - yolo26n-hpo-05
> - yolo26n-hpo-04
> - yolo26n-hpo-03
> - yolo26n-hpo-02
> - yolo26n-hpo-01

### A08 — yolo26n-hpo-20 run의 metrics/mAP50B 값만 알려줘

**기대:** 값 하나: 0.9017. 전체 상세 표는 불필요.

**개선 전** — 라우팅 `get_run_info`

- 해석: `[{"intent": "get_run_info", "arguments": {"run_name": "yolo26n-hpo-20", "metric_key": "metrics/mAP50B"}}]`
- 호출: `llm:intent_parser` → `find_runs(run_name='yolo26n-hpo-20')`

> `yolo26n-hpo-20`의 `metrics/mAP50B`는 0.901733입니다.

**개선 후** — 라우팅 `get_run_info`

- 해석: `[{"request": {"target": "run", "action": "lookup", "runs": ["yolo26n-hpo-20"], "metric": "metrics/mAP50B", "output": "value"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `find_runs(run_name='yolo26n-hpo-20')`

> `yolo26n-hpo-20`의 `metrics/mAP50B`는 0.901733입니다.

### A09 — yolo26n-hpo-20 run 상세 알려줘

**기대:** run 전체 상세(파라미터·metric 포함).

**개선 전** — 라우팅 `get_run_info`

- 해석: `[{"intent": "get_run_info", "arguments": {"run_name": "yolo26n-hpo-20"}}]`
- 호출: `llm:intent_parser` → `find_runs(run_name='yolo26n-hpo-20')`

답변

`yolo26n-hpo-20` run 정보

- Experiment: `ultralytics-yolo-hpo` (ID `9`)
- Run name: `yolo26n-hpo-20`
- Run ID: `6c4ca958f83a46b888632cf63d737f18`
- Status: FINISHED
- User: dyyu
- Start: 2026-10-14 15:11:00
- End: 2026-10-14 15:11:07

**Parameters**

| Key | Value |
|-----|-------|
| agnostic_nms | False |
| amp | True |
| angle | 1.0 |
| augment | False |
| auto_augment | randaugment |
| batch | 16 |
| bgr | 0.0 |
| box | 7.5 |
| cache | False |
| cfg | None |
| classes | None |
| close_mosaic | 10 |
| cls | 0.5 |
| compile | False |
| conf | None |
| copy_paste | 0.0 |
| copy_paste_mode | flip |
| cos_lr | False |
| cutmix | 0.0 |
| data | /home/dyyu/anaconda3/envs/mlflow310/lib/python3.10/site-packages/ultralytics/cfg/datasets/coco8.yaml |
| degrees | 0.0 |
| deterministic | True |
| device | None |
| dfl | 1.5 |
| dnn | False |
| dropout | 0.0 |
| dynamic | False |
| embed | None |
| end2end | None |
| epochs | 5 |
| erasing | 0.4 |
| exist_ok | False |
| fliplr | 0.5 |
| flipud | 0.0 |
| format | torchscript |
| fraction | 1.0 |
| freeze | None |
| half | False |
| hsv_h | 0.015 |
| hsv_s | 0.79 |
| hsv_v | 0.472 |
| imgsz | 640 |
| int8 | False |
| iou | 0.7 |
| keras | False |
| kobj | 1.0 |
| line_width | None |
| lr0 | 0.00118 |
| lrf | 0.0119 |
| mask_ratio | 4 |
| max_det | 300 |
| mixup | 0.19 |
| mode | train |
| model | yolo26n.pt |
| momentum | 0.955 |
| mosaic | 0.62 |
| multi_scale | 0.0 |
| name | yolo26n-hpo-202 |
| nbs | 64 |
| nms | False |
| opset | None |
| optimize | False |
| optimizer | AdamW |
| overlap_mask | True |
| patience | 50 |
| perspective | 0.0 |
| plots | True |
| pose | 12.0 |
| pretrained | True |
| profile | False |
| project | runs/train |
| rect | False |
| resume | False |
| retina_masks | False |
| rle | 1.0 |
| save | True |
| save_conf | False |
| save_crop | False |
| save_dir | /home/dyyu/MLflow/runs/detect/runs/train/yolo26n-hpo-202 |
| save_frames | False |
| save_json | False |
| save_period | -1 |
| save_txt | False |
| scale | 0.635 |
| seed | 0 |
| shear | 0.0 |
| show | False |
| show_boxes | True |
| show_conf | True |
| show_labels | True |
| simplify | True |
| single_cls | False |
| source | None |
| split | val |
| stream_buffer | False |
| task | detect |
| time | None |
| tracker | botsort.yaml |
| translate | 0.195 |
| val | True |
| verbose | True |
| vid_stride | 1 |
| visualize | False |
| warmup_bias_lr | 0.1 |
| warmup_epochs | 3.0 |
| warmup_momentum | 0.8 |
| weight_decay | 0.000545 |
| workers | 8 |
| workspace | None |

**Metrics**

| Key | Value |
|-----|-------|
| lr/pg0 | 9.88934e-06 |
| lr/pg1 | 9.88934e-06 |
| lr/pg2 | 0.0960099 |
| metrics/mAP50-95B | 0.668059 |
| metrics/mAP50B | 0.901733 |
| metrics/precisionB | 0.900907 |
| metrics/recallB | 0.654259 |
| train/box_loss | 1.36043 |
| train/cls_loss | 1.29945 |
| train/dfl_loss | 0.03448 |
| val/box_loss | 2.89021 |
| val/cls_loss | 5.8718 |
| val/dfl_loss | 0.03557 |



**개선 후** — 라우팅 `get_run_info`

- 해석: `[{"request": {"target": "run", "action": "lookup", "runs": ["yolo26n-hpo-20"], "output": "full"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `find_runs(run_name='yolo26n-hpo-20')`

답변

`yolo26n-hpo-20` run 정보

- Experiment: `ultralytics-yolo-hpo` (ID `9`)
- Run name: `yolo26n-hpo-20`
- Run ID: `6c4ca958f83a46b888632cf63d737f18`
- Status: FINISHED
- User: dyyu
- Start: 2026-10-14 15:11:00
- End: 2026-10-14 15:11:07

**Parameters**

| Key | Value |
|-----|-------|
| agnostic_nms | False |
| amp | True |
| angle | 1.0 |
| augment | False |
| auto_augment | randaugment |
| batch | 16 |
| bgr | 0.0 |
| box | 7.5 |
| cache | False |
| cfg | None |
| classes | None |
| close_mosaic | 10 |
| cls | 0.5 |
| compile | False |
| conf | None |
| copy_paste | 0.0 |
| copy_paste_mode | flip |
| cos_lr | False |
| cutmix | 0.0 |
| data | /home/dyyu/anaconda3/envs/mlflow310/lib/python3.10/site-packages/ultralytics/cfg/datasets/coco8.yaml |
| degrees | 0.0 |
| deterministic | True |
| device | None |
| dfl | 1.5 |
| dnn | False |
| dropout | 0.0 |
| dynamic | False |
| embed | None |
| end2end | None |
| epochs | 5 |
| erasing | 0.4 |
| exist_ok | False |
| fliplr | 0.5 |
| flipud | 0.0 |
| format | torchscript |
| fraction | 1.0 |
| freeze | None |
| half | False |
| hsv_h | 0.015 |
| hsv_s | 0.79 |
| hsv_v | 0.472 |
| imgsz | 640 |
| int8 | False |
| iou | 0.7 |
| keras | False |
| kobj | 1.0 |
| line_width | None |
| lr0 | 0.00118 |
| lrf | 0.0119 |
| mask_ratio | 4 |
| max_det | 300 |
| mixup | 0.19 |
| mode | train |
| model | yolo26n.pt |
| momentum | 0.955 |
| mosaic | 0.62 |
| multi_scale | 0.0 |
| name | yolo26n-hpo-202 |
| nbs | 64 |
| nms | False |
| opset | None |
| optimize | False |
| optimizer | AdamW |
| overlap_mask | True |
| patience | 50 |
| perspective | 0.0 |
| plots | True |
| pose | 12.0 |
| pretrained | True |
| profile | False |
| project | runs/train |
| rect | False |
| resume | False |
| retina_masks | False |
| rle | 1.0 |
| save | True |
| save_conf | False |
| save_crop | False |
| save_dir | /home/dyyu/MLflow/runs/detect/runs/train/yolo26n-hpo-202 |
| save_frames | False |
| save_json | False |
| save_period | -1 |
| save_txt | False |
| scale | 0.635 |
| seed | 0 |
| shear | 0.0 |
| show | False |
| show_boxes | True |
| show_conf | True |
| show_labels | True |
| simplify | True |
| single_cls | False |
| source | None |
| split | val |
| stream_buffer | False |
| task | detect |
| time | None |
| tracker | botsort.yaml |
| translate | 0.195 |
| val | True |
| verbose | True |
| vid_stride | 1 |
| visualize | False |
| warmup_bias_lr | 0.1 |
| warmup_epochs | 3.0 |
| warmup_momentum | 0.8 |
| weight_decay | 0.000545 |
| workers | 8 |
| workspace | None |

**Metrics**

| Key | Value |
|-----|-------|
| lr/pg0 | 9.88934e-06 |
| lr/pg1 | 9.88934e-06 |
| lr/pg2 | 0.0960099 |
| metrics/mAP50-95B | 0.668059 |
| metrics/mAP50B | 0.901733 |
| metrics/precisionB | 0.900907 |
| metrics/recallB | 0.654259 |
| train/box_loss | 1.36043 |
| train/cls_loss | 1.29945 |
| train/dfl_loss | 0.03448 |
| val/box_loss | 2.89021 |
| val/cls_loss | 5.8718 |
| val/dfl_loss | 0.03557 |



### A10 — burly-kit-617 run 핵심만 요약해줘

**기대:** 핵심 요약 몇 줄(주요 metric, 설정). 전체 파라미터 표 반복은 불필요.

**개선 전** — 라우팅 `mlflow_mcp`

- 해석: `[{"intent": "mlflow_mcp", "arguments": {"run_name": "burly-kit-617"}}]`
- 호출: `llm:intent_parser` → `find_runs(run_name='burly-kit-617')` → `llm:run_answer`

> MLP 모델(SGD 옵티마이저, 학습률 0.001)을 배치 사이즈 64로 5에폭 훈련한 결과입니다. 검증 정확도(val_accuracy)는 **77.95%**였으며, 검증 손실(val_loss)은 **0.6467**이었습니다.

**개선 후** — 라우팅 `summarize_run`

- 해석: `[{"request": {"target": "run", "action": "summarize", "runs": ["burly-kit-617"], "output": "brief"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `find_runs(run_name='burly-kit-617')` → `llm:run_summary`

> SGD 옵티마이저와 학습률 0.001을 사용한 MLP(은닉층 [512, 512])를 5에폭 동안 학습했습니다.
> 검증 정확도는 77.95%, 검증 손실은 0.6467로 수렴된 결과를 보였습니다.
> 훈련과 검증 성능 간 차이가 크지 않아 과적합은 발생하지 않았습니다.

### A11 — carefree-moth-973 run 학습 잘 된 편이야?

**기대:** 평가: 결론 먼저 + 근거 metric(val_accuracy 77.81 등). 같은 실험 다른 run과 비교하면 더 좋음.

**개선 전** — 라우팅 `mlflow_mcp`

- 해석: `[{"intent": "mlflow_mcp", "arguments": {"run_name": "carefree-moth-973"}}]`
- 호출: `llm:intent_parser` → `find_runs(run_name='carefree-moth-973')` → `llm:run_answer`

답변

결론부터 말씀드리면, **학습이 어느 정도 진행되었으나 아직 성능이 낮은 상태**입니다. 특히 검증 데이터(val_accuracy)에서 훈련 데이터(train_accuracy)보다 성능이 낮아 과적합(Overfitting)의 초기 징후가 보이며, 전체적인 정확도 수치 자체도 높지 않습니다.

주요 근거는 다음과 같습니다:

* **정확도(Accuracy)**: 검증 정확도(`val_accuracy`)가 약 77.81%로, 훈련 정확도(`train_accuracy` 78.31%)보다 낮습니다. 이는 모델이 학습 데이터에 지나치게 맞춰지고 있을 가능성을 시사합니다.
* **손실(Loss)**: 검증 손실(`val_loss`, 0.645)이 훈련 손실(`train_loss`, 0.655)보다 약간 낮은 것은 이상할 수 있으나, 두 값 모두 여전히 높은 편입니다(1에 가까울수록 나쁨).
* **학습 단계**: 에포크가 단 5회로 매우 짧아, 모델이 충분히 학습되지 않은 상태일 가능성이 큽니다.

따라서 "잘 되었다"고 보기보다는, **에포크를 늘리거나 하이퍼파라미터(예: learning_rate)를 조정하여 더 나은 성능을 찾아야 하는 단계**로 해석됩니다.



**개선 후** — 라우팅 `evaluate_run`

- 해석: `[{"request": {"target": "run", "action": "evaluate", "runs": ["carefree-moth-973"]}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `find_runs(run_name='carefree-moth-973')` → `get_runs('Deep Learning Experiment')` → `llm:run_evaluation`

> 검증 지표 기준으로 중앙값과 동일하며, 전체 3개 중 상위권(2위)의 안정된 성능을 보입니다. 주요 근거는 `val_accuracy` 77.81 (3개 중 2위, 최고와 0.14%p 차이), `val_loss` 0.645875 (3개 중 1위, 최저값 기록)입니다. 학습 지표(`train_accuracy`)가 검증 지표보다 높으나 그 차이가 매우 작아(약 0.5%p) 과적합의 위험은 낮아 보입니다.

### B01 — Timm Hub에서 val_acc 가장 높은 run 2개 알려줘

**기대:** val_acc 내림차순 2개: capricious-moth-486(0.9569), mysterious-yak-903(0.9554).

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {"experiment_name": "Timm Hub", "metric_key": "val_acc", "top_k": 2, "ascending": false}}, {"search_filters": {"recent_days": null, "status": null, "user": null, "tag_key": null, "tag_value": null}}]`
- 호출: `llm:intent_parser` → `get_runs('Timm Hub')` → `llm:search_filter_parser` → `list_run_metrics('Timm Hub')`

> val_acc가 없습니다. Timm Hub의 metric 키 목록을 조회해 주세요.

**개선 후** — 라우팅 `search_runs`

- 해석: `[{"request": {"target": "run", "action": "list", "experiment": "Timm Hub", "metric": "val_acc", "sort_by": "metric", "order": "desc", "limit": 2, "output": "table"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('Timm Hub')`

> 적용 조건: 실험 `Timm Hub` · 정렬 `val_acc` 높은 순 · 2개
>
> run 4개 중 2개
>
> | Run | val_acc | Status | Start |
> |---|---|---|---|
> | capricious-moth-486 | 0.9569 | FINISHED | 2026-04-10 14:55:24 |
> | mysterious-yak-903 | 0.9554 | FINISHED | 2026-04-08 15:05:36 |

### B02 — Hyperparameter Tuning Experiment에서 error가 가장 낮은 run 3개 보여줘

**기대:** error 오름차순 3개: trial_21(0.2446), trial_28(0.2457), trial_15(0.2465). error 값이 없는 FAILED/study run이 앞에 오면 안 된다.

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {"metric_key": "error", "top_k": 3, "ascending": true}}]`
- 호출: `llm:intent_parser`

> 조회할 실험이 없습니다.

**개선 후** — 라우팅 `search_runs`

- 해석: `[{"request": {"target": "run", "action": "list", "experiment": "Hyperparameter Tuning Experiment", "metric": "error", "sort_by": "metric", "order": "asc", "limit": 3, "output": "table"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('Hyperparameter Tuning Experiment')`

> 적용 조건: 실험 `Hyperparameter Tuning Experiment` · 정렬 `error` 낮은 순 · 3개
>
> run 32개 중 3개
>
> | Run | error | Status | Start |
> |---|---|---|---|
> | trial_21 | 0.244573 | FINISHED | 2026-03-30 10:19:53 |
> | trial_28 | 0.245701 | FINISHED | 2026-03-30 10:23:58 |
> | trial_15 | 0.246475 | FINISHED | 2026-03-30 10:14:45 |
>
> ※ `error` 값이 없는 run 3개는 순위에서 제외했습니다.

### B03 — ultralytics-yolo-hpo 실험에서 가장 최근 run 3개 보여줘

**기대:** 시작 시각 최신순 3개: yolo26n-hpo-20, 19, 18 (모두 같은 날짜, 시각으로 정렬). metric 순위(top_k)와 혼동하지 않아야 한다.

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {"experiment_name": "ultralytics-yolo-hpo", "top_k": 3, "ascending": false}}, {"search_filters": {"recent_days": null, "status": null, "user": null, "tag_key": null, "tag_value": null}}]`
- 호출: `llm:intent_parser` → `get_runs('ultralytics-yolo-hpo')` → `llm:search_filter_parser`

> top_k 검색에는 metric_key 선택이 필요합니다. metric을 선택한 뒤 다시 시도해 주세요.

**개선 후** — 라우팅 `search_runs`

- 해석: `[{"request": {"target": "run", "action": "list", "experiment": "ultralytics-yolo-hpo", "sort_by": "time", "order": "desc", "limit": 3, "output": "table"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('ultralytics-yolo-hpo')`

> 적용 조건: 실험 `ultralytics-yolo-hpo` · 정렬 시작 시각 최신순 · 3개
>
> run 20개 중 3개
>
> | Run | Status | Start |
> |---|---|---|
> | yolo26n-hpo-20 | FINISHED | 2026-10-14 15:11:00 |
> | yolo26n-hpo-19 | FINISHED | 2026-10-14 15:10:48 |
> | yolo26n-hpo-18 | FINISHED | 2026-10-14 15:10:32 |

### B04 — Hyperparameter Tuning Experiment에서 실패한 run 보여줘

**기대:** status=FAILED 2개: trial_2, study. metric 없이도 검색되어야 한다.

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {"experiment_name": "Hyperparameter Tuning Experiment"}}, {"search_filters": {"recent_days": null, "status": "FAILED", "user": null, "tag_key": null, "tag_value": null}}]`
- 호출: `llm:intent_parser` → `get_runs('Hyperparameter Tuning Experiment')` → `llm:search_filter_parser` → `list_run_metrics('Hyperparameter Tuning Experiment')`

> None가 없습니다. Hyperparameter Tuning Experiment의 metric 키 목록을 조회해 주세요.

**개선 후** — 라우팅 `search_runs`

- 해석: `[{"request": {"target": "run", "action": "list", "experiment": "Hyperparameter Tuning Experiment", "filters": {"status": "FAILED"}, "output": "table"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('Hyperparameter Tuning Experiment')`

> 적용 조건: 실험 `Hyperparameter Tuning Experiment` · 상태 FAILED
>
> run 2개
>
> | Run | Status | Start |
> |---|---|---|
> | trial_2 | FAILED | 2026-04-10 14:38:58 |
> | study | FAILED | 2026-04-10 14:37:54 |

### B05 — ultralytics-yolo-hpo에서 최근 30일 안에 돌린 run 몇 개야?

**기대:** 기간 필터(30일) 적용 후 개수만: 20개.

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {"experiment_name": "ultralytics-yolo-hpo", "top_k": 3}}, {"search_filters": {"recent_days": 30, "status": null, "user": null, "tag_key": null, "tag_value": null}}]`
- 호출: `llm:intent_parser` → `get_runs('ultralytics-yolo-hpo')` → `llm:search_filter_parser`

> top_k 검색에는 metric_key 선택이 필요합니다. metric을 선택한 뒤 다시 시도해 주세요.

**개선 후** — 라우팅 `search_runs`

- 해석: `[{"request": {"target": "run", "action": "count", "experiment": "ultralytics-yolo-hpo", "filters": {"recent_days": 30}, "output": "count"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('ultralytics-yolo-hpo')`

> 적용 조건: 실험 `ultralytics-yolo-hpo` · 기간 최근 30일
>
> 조건에 맞는 run은 **20개**입니다.

### B06 — Timm Hub에서 f1_score 높은 run 3개 보여줘

**기대:** f1_score metric이 없다고 안내하고 사용 가능한 metric(val_acc 등)을 제시. 다른 metric으로 몰래 대체 금지.

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {"metric_key": "f1_score", "top_k": 3, "ascending": false}}]`
- 호출: `llm:intent_parser`

> 조회할 실험이 없습니다.

**개선 후** — 라우팅 `clarify`

- 해석: `[{"request": {"target": "run", "action": "list", "experiment": "Timm Hub", "metric": "f1_score", "sort_by": "metric", "order": "desc", "limit": 3, "output": "table"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('Timm Hub')`

> `Timm Hub`에 `f1_score` metric이 없습니다. 사용할 수 있는 metric: `train_acc`, `train_loss`, `val_acc`, `val_loss`

### C01 — Timm Hub에서 val_acc 기준 최고 run과 최저 run을 비교해줘

**기대:** 최고 capricious-moth-486(0.9569) vs 최저 loud-gnat-654(0.9460) 비교.

**개선 전** — 라우팅 `mlflow_mcp`

- 해석: `[{"intent": "mlflow_mcp", "arguments": {"run_name": "val_acc", "metric_key": "val_acc"}}]`
- 호출: `llm:intent_parser` → `mcp:search_experiments('view')` → `mcp:list_runs('experiment_id', 'view')` → `mcp:describe_run('run_id')` → `mcp:describe_run('run_id')` → `mcp:describe_run('run_id')` → `mcp:describe_run('run_id')`

답변

Timm Hub 실험에서 `val_acc`를 기준으로 가장 높았던 run과 낮았던 run을 비교해 드립니다.

### 🏆 최고 Performance Run: **capricious-moth-486**

* **val_accuracy:** 95.69% (0.9569)
* **train_loss:** 0.1636
* **val_loss:** 0.1245

### 📉 최저 Performance Run: **loud-gnat-654**

* **val_accuracy:** 94.60% (0.946)
* **train_loss:** 0.5266
* **val_loss:** 0.1634

---

### 📊 비교 분석

두 run 모두 동일한 하이퍼파라미터(`lr: 0.0001`, `optimizer: AdamW`, `epochs: 2`)와 모델(`resnet50-oxford-iiit-pet`)을 사용했음에도 성능 차이가 발생했습니다.

1. **최단 시간 실행:** 최고 성능을 기록한 `capricious-moth-486`은 2026년 4월 10일 14:55, 최저 성능인 `loud-gnat-654`는 같은 날 14:54에 로그되었습니다. 즉, 실험 시간을 기준으로 **최저 run이 최고 run보다 약 1분 먼저 종료**되었습니다.
2. **성능 차이:** 검증 정확도(val_acc)에서 약 **1.09%** (`capricious-moth-486`이 `loud-gnat-654`에 비해 훨씬 낮음, *참고: 실제 데이터상으론 capricious가 더 높음*)의 차이를 보이며, 학습 손실(train_loss)에서도 큰 격차가 있음을 알 수 있습니다. (최고 run은 0.163, 최저 run은 0.526으로 **최대 run이 최소 run보다 train_loss가 훨씬 낮았**)

따라서 실행 순서나 데이터 처리의 미세한 차이(예: 난수 시드 등)로 인해 성능 편차가 발생했을 가능성이 있습니다.



**개선 후** — 라우팅 `compare_runs`

- 해석: `[{"request": {"target": "run", "action": "compare", "experiment": "Timm Hub", "metric": "val_acc", "sort_by": "metric", "order": "desc", "pick": "both_ends"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('Timm Hub')`

> 적용 조건: 실험 `Timm Hub` · 정렬 `val_acc` 높은 순 · 최고·최저 run
>
> | 항목 | capricious-moth-486 (최고) | loud-gnat-654 (최저) |
> |---|---|---|
> | val_acc | 0.9569 | 0.946 |
> | train_acc | 0.94704 | 0.83844 |
> | train_loss | 0.163607 | 0.526638 |
> | val_loss | 0.124486 | 0.163361 |
>
> `val_acc` 차이: `capricious-moth-486` − `loud-gnat-654` = +0.0109 (`capricious-moth-486`가 더 높음)
>
> ※ 비교한 run의 파라미터는 모두 같습니다.

### C02 — 가장 최근 실험에서 mAP50B가 제일 높은 run 알려줘

**기대:** 최근 실험(ultralytics-yolo-hpo) 선택 → mAP50B 최고 1개: yolo26n-hpo-09(0.9062).

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {"metric_key": "mAP50B", "top_k": 1, "ascending": false}}]`
- 호출: `llm:intent_parser`

> 조회할 실험이 없습니다.

**개선 후** — 라우팅 `search_runs`

- 해석: `[{"request": {"target": "run", "action": "list", "experiment_pick": "latest", "metric": "mAP50B", "sort_by": "metric", "order": "desc", "limit": 1, "output": "brief"}}]`
- 호출: `llm:request_parser` → `list_recent_experiments(limit=1, order_by='creation_time', ascending=False)` → `get_runs('ultralytics-yolo-hpo')`

> 적용 조건: 실험 `ultralytics-yolo-hpo`(가장 최근 실험) · 정렬 `metrics/mAP50B` 높은 순 · 1개
>
> `yolo26n-hpo-09` — `metrics/mAP50B` 0.90617 (FINISHED, 2026-10-14 15:08:30)

### C03 — yolo26n-hpo-09와 yolo26n-hpo-20의 mAP50B 차이 알려줘

**기대:** 두 run 값(0.9062, 0.9017)과 차이(약 0.0044).

**개선 전** — 라우팅 `mlflow_mcp`

- 해석: `[{"intent": "mlflow_mcp", "arguments": {"run_name": "yolo26n-hpo-09", "metric_key": "metrics/mAP50B"}}]`
- 호출: `llm:intent_parser` → `mcp:search_experiments()` → `mcp:list_runs('experiment_id')` → `mcp:list_runs('experiment_id')` → `mcp:describe_run('run_id')` → `mcp:describe_run('run_id')`

> 각 실험의 mAP50B (metrics/mAP50B)는 다음과 같습니다:
>
> * **yolo26n-hpo-09**: 0.90617
> * **yolo26n-hpo-20**: 0.90173
>
> 두 모델의 mAP50B 차이는 약 **-0.00444** (또는 약 -0.44%)입니다. 즉, `yolo26n-hpo-09`가 `yolo26n-hpo-20`보다 mAP50B 성능이 더 높습니다.

**개선 후** — 라우팅 `compare_runs`

- 해석: `[{"request": {"target": "run", "action": "compare", "runs": ["yolo26n-hpo-09", "yolo26n-hpo-20"], "metric": "mAP50B", "output": "value"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `find_runs(run_name='yolo26n-hpo-09')` → `find_runs(run_name='yolo26n-hpo-20')`

> | 항목 | yolo26n-hpo-09 | yolo26n-hpo-20 |
> |---|---|---|
> | metrics/mAP50B | 0.90617 | 0.901733 |
>
> `metrics/mAP50B` 차이: `yolo26n-hpo-09` − `yolo26n-hpo-20` = +0.00443711 (`yolo26n-hpo-09`가 더 높음)

### C04 — 보고서 기준으로 최근 run 두 개를 비교해줘

**기대:** 보고서 근거로 yolo26n-hpo-20 vs yolo26n-hpo-19 비교(0.9017 vs 0.9013, 차이는 변동 범위 안).

**개선 전** — 라우팅 `mlflow_mcp`

- 해석: `[{"intent": "mlflow_mcp", "arguments": {}}]`
- 호출: `llm:intent_parser` → `mcp:search_experiments()` → `mcp:list_runs('experiment_id')` → `mcp:list_runs('experiment_id')` → `mcp:describe_run('run_id')` → `mcp:describe_run('run_id')`

> 보고서 기반으로 'MLflow Quickstart' 실험의 최근 두 개 run을 비교해 드리겠습니다.
>
> 두 run은 다음과 같은 유사한 결과를 보입니다:
>
> * **인트리구드 스쿼드 781 (intrigued-squid-781):**
>   * 생성 시간: 2026년 4월 10일
>   * 평가 정확도: 1.0
>   * 상태: 완료(FINISHED)
> * **언퀄티드 간트 8 (unequaled-gnat-8):**
>   * 생성 시간: 2026년 4월 8일
>   * 평가 정확도: 1.0
>   * 상태: 완료(FINISHED)
>
> **비교 요약:**
> 두 실험은 동일한 코드를 기반으로 훈련되었으며, 둘 다 **100%(1.0)**의 높은 평가 정확도를 달성했습니다. 생성된 모델 ID는 다르지만(각각 m-eab2a8b12dde...과 m-6068a901f121...), 성능 차이는 보이지 않습니다.

**개선 후** — 라우팅 `report_qa`

- 해석: `[{"request": {"target": "report", "action": "compare", "limit": 2, "output": "table"}}]`
- 호출: `llm:request_parser` → `llm:report_answer`

> 최근 run 두 개는 yolo26n-hpo-20 (mAP50B 0.9017)과 yolo26n-hpo-19 (mAP50B 0.9013)입니다. yolo26n-hpo-20이 mAP50B 기준으로 0.0005 높지만, 이는 run 간 변동 범위 내에 있습니다.

### C05 — Deep Learning Experiment에서 val_accuracy 제일 높은 run의 파라미터만 알려줘

**기대:** val_accuracy 최고 run(burly-kit-617, 77.95) 선택 → 파라미터만: batch_size 64, epochs 5, hidden_units [512, 512], learning_rate 0.001, model_type MLP, optimizer SGD.

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {"experiment_name": "Deep Learning Experiment", "metric_key": "val_accuracy", "top_k": 1}}, {"search_filters": {"recent_days": null, "status": null, "user": null, "tag_key": null, "tag_value": null}}]`
- 호출: `llm:intent_parser` → `get_runs('Deep Learning Experiment')` → `llm:search_filter_parser` → `list_run_metrics('Deep Learning Experiment')`

> val_accuracy가 없습니다. Deep Learning Experiment의 metric 키 목록을 조회해 주세요.

**개선 후** — 라우팅 `get_run_info`

- 해석: `[{"request": {"target": "run", "action": "lookup", "experiment": "Deep Learning Experiment", "metric": "val_accuracy", "sort_by": "metric", "order": "desc", "limit": 1, "fields": ["params"]}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('Deep Learning Experiment')`

> 적용 조건: 실험 `Deep Learning Experiment` · 정렬 `val_accuracy` 높은 순 · 1개
>
> `burly-kit-617` (Deep Learning Experiment)
>
> | Parameter | Value |
> |---|---|
> | batch_size | 64 |
> | epochs | 5 |
> | hidden_units | [512, 512] |
> | learning_rate | 0.001 |
> | model_type | MLP |
> | optimizer | SGD |

### C06 — 보고서 기준으로 GPU 메모리 사용량이 가장 적은 run 알려줘

**기대:** 보고서에 GPU 메모리 정보가 없으므로 '리포트에 근거 없음'. MLflow 조회나 수치 생성 금지.

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {}}]`
- 호출: `llm:intent_parser`

> 조회할 실험이 없습니다.

**개선 후** — 라우팅 `report_qa`

- 해석: `[{"request": {"target": "report", "action": "list", "metric": "GPU 메모리 사용량", "sort_by": "metric", "order": "asc", "limit": 1, "output": "brief"}}]`
- 호출: `llm:request_parser` → `llm:report_answer`

> 리포트에 근거 없음

### D01 — 그 run 성능 어때?

**기대:** 새 대화라 대상 run이 없다. 어떤 run인지 확인 질문.

**개선 전** — 라우팅 `mlflow_mcp`

- 해석: `[{"intent": "mlflow_mcp", "arguments": {}}]`
- 호출: `llm:intent_parser`

> 사용자가 "그 run"이라고 했으므로 먼저 어떤 실험(experiment)과 런(run)을 참조하는지 파악해야 합니다. 지시사항에 따르면 'run name'은 사람이 읽는 이름(예: carefree-moth-973)이며, 사용자가 이것을 run ID라고 불러도 experiment_id로 쓰면 안 된다고 명시되어 있습니다. 
>
> 먼저 사용 가능한 실험 목록을 확인한 후, 각 실험의 런들을 나열해야 합니다. 그런 다음 해당 성능 데이터를 평가할 수 있습니다.
>
> 질문이 매우 일반적이므로 먼저 모든 실험을 리스팅 해보겠습니다:
>
> search_experiments() 호출 - 기본 설정으로 조회

**개선 후** — 라우팅 `clarify`

- 해석: `[{"request": {"target": "run", "action": "evaluate", "unresolved": ["run"]}}]`
- 호출: `llm:request_parser`

> 요청(run 평가)을 처리하려면 다음을 알려주세요.
>
> - 어떤 run인가요? run 이름이나 32자리 run ID로 알려주세요.

### D02 — 성능 제일 좋은 run 알려줘

**기대:** 실험·기준 metric이 없다. 확인 질문, 또는 가정을 명시하고 답.

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {}}]`
- 호출: `llm:intent_parser`

> 조회할 실험이 없습니다.

**개선 후** — 라우팅 `clarify`

- 해석: `[{"request": {"target": "run", "action": "list", "unresolved": ["experiment", "metric"]}}]`
- 호출: `llm:request_parser` → `list_experiment_names()`

> 요청(run 조회)을 처리하려면 다음을 알려주세요.
>
> - 어느 실험에서 찾을까요? 실험: `Deep Learning Experiment`, `Default`, `Hyperparameter Tuning Experiment`, `MLflow Quickstart`, `My Application`, `OpenAI_Test_Project`, `Timm Hub`, `ultralytics-yolo`, `ultralytics-yolo-hpo`
> - 어떤 metric 기준으로 볼까요?

### D03 — ultralytics-yolo-hpo 실험 삭제해줘

**기대:** 지원 범위 밖(조회 전용)이라고 안내. 실제 삭제 시도 금지.

**개선 전** — 라우팅 `mlflow_mcp`

- 해석: `[{"intent": "mlflow_mcp", "arguments": {"experiment_name": "ultralytics-yolo-hpo"}}]`
- 호출: `llm:intent_parser` → `mcp:search_experiments('view')` → `mcp:get_experiment('experiment_id', 'output')`

> 죄송하지만, 현재 사용 가능한 도구에는 실험(experiment)을 삭제하는 기능이 포함되어 있지 않습니다. 
>
> MLflow에서 실험을 삭제하려면 일반적으로 다음 명령어를 사용할 수 있습니다:
>
> ```bash
> mlflow experiments delete -x 9
> ```
>
> 또는 프론트엔드 UI를 통해 수동으로 삭제할 수도 있습니다.

**개선 후** — 라우팅 `unsupported`

- 해석: `[{"request": {"target": "experiment", "action": "unsupported", "experiment": "ultralytics-yolo-hpo", "unsupported_reason": "실험 삭제"}}]`
- 호출: `llm:request_parser`

> 요청하신 작업(실험 삭제)은 이 챗봇에서 할 수 없습니다. 이 챗봇은 MLflow를 조회만 합니다.
> 할 수 있는 것: 실험·run 목록과 상세 조회, 조건 검색, run 비교·요약·평가, 첨부 보고서 질의응답

### M1-1 (세션 M1) — Timm Hub run 중 val_acc 상위 2개 보여줘

**기대:** val_acc 내림차순 2개.

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {"experiment_name": "Timm Hub", "metric_key": "val_acc", "top_k": 2, "ascending": false}}, {"search_filters": {"recent_days": null, "status": null, "user": null, "tag_key": null, "tag_value": null}}]`
- 호출: `llm:intent_parser` → `get_runs('Timm Hub')` → `llm:search_filter_parser` → `list_run_metrics('Timm Hub')`

> val_acc가 없습니다. Timm Hub의 metric 키 목록을 조회해 주세요.

**개선 후** — 라우팅 `search_runs`

- 해석: `[{"request": {"target": "run", "action": "list", "experiment": "Timm Hub", "metric": "val_acc", "sort_by": "metric", "order": "desc", "limit": 2, "output": "table"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('Timm Hub')`

> 적용 조건: 실험 `Timm Hub` · 정렬 `val_acc` 높은 순 · 2개
>
> run 4개 중 2개
>
> | Run | val_acc | Status | Start |
> |---|---|---|---|
> | capricious-moth-486 | 0.9569 | FINISHED | 2026-04-10 14:55:24 |
> | mysterious-yak-903 | 0.9554 | FINISHED | 2026-04-08 15:05:36 |

### M1-2 (세션 M1) — 전체 run 보여줘

**기대:** 실험(Timm Hub)은 유지, 개수 제한(2) 해제 → run 4개 전부.

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {}}, {"search_filters": {"recent_days": null, "status": null, "user": null, "tag_key": null, "tag_value": null}}]`
- 호출: `llm:intent_parser` → `get_runs('Timm Hub')` → `llm:search_filter_parser` → `list_run_metrics('Timm Hub')`

> val_acc가 없습니다. Timm Hub의 metric 키 목록을 조회해 주세요.

**개선 후** — 라우팅 `get_runs`

- 해석: `[{"request": {"target": "run", "action": "list", "experiment": "Timm Hub", "all": true, "output": "table", "inherited": ["experiment"], "cleared": ["metric", "sort_by", "order", "limit"]}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('Timm Hub')`

> 적용 조건: 실험 `Timm Hub`(이전 대화) · 이전 metric·정렬·개수 해제
>
> run 4개
>
> | Run | Status | Start |
> |---|---|---|
> | capricious-moth-486 | FINISHED | 2026-04-10 14:55:24 |
> | loud-gnat-654 | FINISHED | 2026-04-10 14:54:29 |
> | mysterious-yak-903 | FINISHED | 2026-04-08 15:05:36 |
> | skittish-dove-494 | FINISHED | 2026-04-08 15:04:38 |

### M1-3 (세션 M1) — 그 중에 val_loss가 제일 낮은 건?

**기대:** Timm Hub 유지, metric을 val_loss로 변경, 오름차순 1개.

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {"experiment_name": "Timm Hub", "metric_key": "val_loss", "top_k": 1, "ascending": true}}, {"search_filters": {"recent_days": null, "status": null, "user": null, "tag_key": null, "tag_value": null}}]`
- 호출: `llm:intent_parser` → `get_runs('Timm Hub')` → `llm:search_filter_parser` → `list_run_metrics('Timm Hub')`

> val_loss가 없습니다. Timm Hub의 metric 키 목록을 조회해 주세요.

**개선 후** — 라우팅 `search_runs`

- 해석: `[{"request": {"target": "run", "action": "list", "experiment": "Timm Hub", "metric": "val_loss", "sort_by": "metric", "order": "asc", "limit": 1, "output": "brief", "inherited": ["experiment"]}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('Timm Hub')`

> 적용 조건: 실험 `Timm Hub`(이전 대화) · 정렬 `val_loss` 낮은 순 · 1개
>
> `capricious-moth-486` — `val_loss` 0.124486 (FINISHED, 2026-04-10 14:55:24)

### M2-1 (세션 M2) — Hyperparameter Tuning Experiment에서 실패한 run 보여줘

**기대:** FAILED 2개: trial_2, study.

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {"experiment_name": "Hyperparameter Tuning Experiment"}}, {"search_filters": {"recent_days": null, "status": "FAILED", "user": null, "tag_key": null, "tag_value": null}}]`
- 호출: `llm:intent_parser` → `get_runs('Hyperparameter Tuning Experiment')` → `llm:search_filter_parser` → `list_run_metrics('Hyperparameter Tuning Experiment')`

> None가 없습니다. Hyperparameter Tuning Experiment의 metric 키 목록을 조회해 주세요.

**개선 후** — 라우팅 `search_runs`

- 해석: `[{"request": {"target": "run", "action": "list", "experiment": "Hyperparameter Tuning Experiment", "filters": {"status": "FAILED"}, "output": "table"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('Hyperparameter Tuning Experiment')`

> 적용 조건: 실험 `Hyperparameter Tuning Experiment` · 상태 FAILED
>
> run 2개
>
> | Run | Status | Start |
> |---|---|---|
> | trial_2 | FAILED | 2026-04-10 14:38:58 |
> | study | FAILED | 2026-04-10 14:37:54 |

### M2-2 (세션 M2) — 완료된 건 몇 개야?

**기대:** 실험 유지, 상태를 FINISHED로 변경, 개수만: 33개.

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {"experiment_name": "Hyperparameter Tuning Experiment"}}, {"search_filters": {"recent_days": null, "status": "FINISHED", "user": null, "tag_key": null, "tag_value": null}}]`
- 호출: `llm:intent_parser` → `get_runs('Hyperparameter Tuning Experiment')` → `llm:search_filter_parser` → `list_run_metrics('Hyperparameter Tuning Experiment')`

> None가 없습니다. Hyperparameter Tuning Experiment의 metric 키 목록을 조회해 주세요.

**개선 후** — 라우팅 `search_runs`

- 해석: `[{"request": {"target": "run", "action": "count", "experiment": "Hyperparameter Tuning Experiment", "filters": {"status": "FINISHED"}, "inherited": ["experiment"]}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('Hyperparameter Tuning Experiment')`

> 적용 조건: 실험 `Hyperparameter Tuning Experiment`(이전 대화) · 상태 FINISHED
>
> 조건에 맞는 run은 **33개**입니다.

### M2-3 (세션 M2) — 상태 조건 빼고 전체는 몇 개야?

**기대:** 실험 유지, 상태 조건 해제, 개수만: 35개.

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {"experiment_name": "Hyperparameter Tuning Experiment"}}, {"search_filters": {"recent_days": null, "status": null, "user": null, "tag_key": null, "tag_value": null}}]`
- 호출: `llm:intent_parser` → `get_runs('Hyperparameter Tuning Experiment')` → `llm:search_filter_parser` → `list_run_metrics('Hyperparameter Tuning Experiment')`

> None가 없습니다. Hyperparameter Tuning Experiment의 metric 키 목록을 조회해 주세요.

**개선 후** — 라우팅 `get_runs`

- 해석: `[{"request": {"target": "run", "action": "count", "experiment": "Hyperparameter Tuning Experiment", "cleared": ["filters"]}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('Hyperparameter Tuning Experiment')`

> 적용 조건: 실험 `Hyperparameter Tuning Experiment` · 이전 조건(기간·상태 등) 해제
>
> 조건에 맞는 run은 **35개**입니다.

### M3-1 (세션 M3) — 가장 최근 실험 하나 알려줘

**기대:** ultralytics-yolo-hpo 1개.

**개선 전** — 라우팅 `list_recent_experiments`

- 해석: `[{"intent": "list_recent_experiments", "arguments": {"top_k": 1}}]`
- 호출: `llm:intent_parser` → `list_recent_experiment_names(limit=5)`

> 최근 실험(5개)
>
> - ultralytics-yolo-hpo
> - ultralytics-yolo
> - Timm Hub
> - My Application
> - Deep Learning Experiment

**개선 후** — 라우팅 `list_recent_experiments`

- 해석: `[{"request": {"target": "experiment", "action": "list", "experiment_pick": "latest", "limit": 1, "output": "brief"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `list_recent_experiments(limit=1, order_by='creation_time', ascending=False)`

> 가장 최근 실험은 `ultralytics-yolo-hpo`입니다. (ID `9`, 생성 2026-10-14 15:06:38)

### M3-2 (세션 M3) — 그 실험 run 중 최근 3개만 보여줘

**기대:** 실험 ultralytics-yolo-hpo 이어받음 → 시작 최신순 3개: yolo26n-hpo-20, 19, 18.

**개선 전** — 라우팅 `get_runs`

- 해석: `[{"intent": "get_runs", "arguments": {"experiment_name": "ultralytics-yolo-hpo", "top_k": 3, "ascending": false}}]`
- 호출: `llm:intent_parser` → `get_runs('ultralytics-yolo-hpo')`

답변

`ultralytics-yolo-hpo` run **20개** (미리보기 **20개**)

| Name | Score | Status | Date | Time |
|------|-------|--------|------|------|
| yolo26n-hpo-20 | - | FINISHED | 2026-10-14 | 15:11:00 |
| yolo26n-hpo-19 | - | FINISHED | 2026-10-14 | 15:10:48 |
| yolo26n-hpo-18 | - | FINISHED | 2026-10-14 | 15:10:32 |
| yolo26n-hpo-17 | - | FINISHED | 2026-10-14 | 15:10:19 |
| yolo26n-hpo-16 | - | FINISHED | 2026-10-14 | 15:10:06 |
| yolo26n-hpo-15 | - | FINISHED | 2026-10-14 | 15:09:51 |
| yolo26n-hpo-14 | - | FINISHED | 2026-10-14 | 15:09:39 |
| yolo26n-hpo-13 | - | FINISHED | 2026-10-14 | 15:09:24 |
| yolo26n-hpo-12 | - | FINISHED | 2026-10-14 | 15:09:11 |
| yolo26n-hpo-11 | - | FINISHED | 2026-10-14 | 15:08:58 |
| yolo26n-hpo-10 | - | FINISHED | 2026-10-14 | 15:08:43 |
| yolo26n-hpo-09 | - | FINISHED | 2026-10-14 | 15:08:30 |
| yolo26n-hpo-08 | - | FINISHED | 2026-10-14 | 15:08:18 |
| yolo26n-hpo-07 | - | FINISHED | 2026-10-14 | 15:08:06 |
| yolo26n-hpo-06 | - | FINISHED | 2026-10-14 | 15:07:53 |
| yolo26n-hpo-05 | - | FINISHED | 2026-10-14 | 15:07:38 |
| yolo26n-hpo-04 | - | FINISHED | 2026-10-14 | 15:07:24 |
| yolo26n-hpo-03 | - | FINISHED | 2026-10-14 | 15:07:10 |
| yolo26n-hpo-02 | - | FINISHED | 2026-10-14 | 15:06:56 |
| yolo26n-hpo-01 | - | FINISHED | 2026-10-14 | 15:06:38 |



**개선 후** — 라우팅 `search_runs`

- 해석: `[{"request": {"target": "run", "action": "list", "experiment": "ultralytics-yolo-hpo", "sort_by": "time", "order": "desc", "limit": 3, "output": "table", "inherited": ["experiment"]}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('ultralytics-yolo-hpo')`

> 적용 조건: 실험 `ultralytics-yolo-hpo`(이전 대화) · 정렬 시작 시각 최신순 · 3개
>
> run 20개 중 3개
>
> | Run | Status | Start |
> |---|---|---|
> | yolo26n-hpo-20 | FINISHED | 2026-10-14 15:11:00 |
> | yolo26n-hpo-19 | FINISHED | 2026-10-14 15:10:48 |
> | yolo26n-hpo-18 | FINISHED | 2026-10-14 15:10:32 |

### M4-1 (세션 M4) — yolo26n-hpo-09 run의 mAP50B 값 알려줘

**기대:** 값 하나: 0.9062 (metric 이름을 접두사 없이 말함).

**개선 전** — 라우팅 `get_run_info`

- 해석: `[{"intent": "get_run_info", "arguments": {"run_name": "yolo26n-hpo-09", "metric_key": "mAP50B"}}]`
- 호출: `llm:intent_parser` → `find_runs(run_name='yolo26n-hpo-09')`

> `yolo26n-hpo-09` run에 `mAP50B` metric이 없습니다.

**개선 후** — 라우팅 `get_run_info`

- 해석: `[{"request": {"target": "run", "action": "lookup", "runs": ["yolo26n-hpo-09"], "metric": "mAP50B", "output": "value"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `find_runs(run_name='yolo26n-hpo-09')`

> `yolo26n-hpo-09`의 `metrics/mAP50B`는 0.90617입니다.

### M4-2 (세션 M4) — 같은 run의 recall은?

**기대:** run yolo26n-hpo-09 이어받음, metric recall → metrics/recallB 값 하나: 0.6613.

**개선 전** — 라우팅 `get_run_info`

- 해석: `[{"intent": "get_run_info", "arguments": {"run_name": "yolo26n-hpo-09", "metric_key": "recall"}}]`
- 호출: `llm:intent_parser` → `find_runs(run_name='yolo26n-hpo-09', experiment_name='ultralytics-yolo-hpo')`

> `yolo26n-hpo-09` run에 `recall` metric이 없습니다.

**개선 후** — 라우팅 `get_run_info`

- 해석: `[{"request": {"target": "run", "action": "lookup", "runs": ["yolo26n-hpo-09"], "metric": "recall", "output": "value", "inherited": ["runs"]}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `find_runs(run_name='yolo26n-hpo-09')`

> 대상 run: `yolo26n-hpo-09`(이전 대화)
>
> `yolo26n-hpo-09`의 `metrics/recallB`는 0.661325입니다.

## 홀드아웃(11)

### H01 — Timm Hub에 run 몇 개 있어?

**기대:** 개수만: 4개.

**개선 전** — 라우팅 `get_runs`

- 해석: `[{"intent": "get_runs", "arguments": {"experiment_name": "Timm Hub"}}]`
- 호출: `llm:intent_parser` → `get_runs('Timm Hub')`

> `Timm Hub` run **4개** (미리보기 **4개**)
>
> | Name | Score | Status | Date | Time |
> |------|-------|--------|------|------|
> | capricious-moth-486 | - | FINISHED | 2026-04-10 | 14:55:24 |
> | loud-gnat-654 | - | FINISHED | 2026-04-10 | 14:54:29 |
> | mysterious-yak-903 | - | FINISHED | 2026-04-08 | 15:05:36 |
> | skittish-dove-494 | - | FINISHED | 2026-04-08 | 15:04:38 |

**개선 후** — 라우팅 `get_runs`

- 해석: `[{"request": {"target": "run", "action": "count", "experiment": "Timm Hub", "output": "count"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('Timm Hub')`

> 적용 조건: 실험 `Timm Hub`
>
> 조건에 맞는 run은 **4개**입니다.

### H02 — ultralytics-yolo-hpo에서 mAP50-95B 상위 3개 run 이름만 알려줘

**기대:** yolo26n-hpo-09, yolo26n-hpo-08, yolo26n-hpo-05 (이름만).

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {"experiment_name": "ultralytics-yolo-hpo", "metric_key": "metrics/mAP50-95B", "top_k": 3, "ascending": false}}, {"search_filters": {"recent_days": null, "status": null, "user": null, "tag_key": null, "tag_value": null}}]`
- 호출: `llm:intent_parser` → `get_runs('ultralytics-yolo-hpo')` → `llm:search_filter_parser` → `list_run_metrics('ultralytics-yolo-hpo')`

> metrics/mAP50-95B가 없습니다. ultralytics-yolo-hpo의 metric 키 목록을 조회해 주세요.

**개선 후** — 라우팅 `search_runs`

- 해석: `[{"request": {"target": "run", "action": "list", "experiment": "ultralytics-yolo-hpo", "metric": "mAP50-95B", "sort_by": "metric", "order": "desc", "limit": 3, "output": "names"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('ultralytics-yolo-hpo')`

> 적용 조건: 실험 `ultralytics-yolo-hpo` · 정렬 `metrics/mAP50-95B` 높은 순 · 3개
>
> run 20개 중 3개
>
> - yolo26n-hpo-09
> - yolo26n-hpo-08
> - yolo26n-hpo-05

### H03 — 가장 오래된 실험 2개 알려줘

**기대:** 생성 오름차순 2개: Default, OpenAI_Test_Project(같은 시각).

**개선 전** — 라우팅 `list_recent_experiments`

- 해석: `[{"intent": "list_recent_experiments", "arguments": {"top_k": 2, "ascending": true}}]`
- 호출: `llm:intent_parser` → `list_recent_experiment_names(limit=5)`

> 최근 실험(5개)
>
> - ultralytics-yolo-hpo
> - ultralytics-yolo
> - Timm Hub
> - My Application
> - Deep Learning Experiment

**개선 후** — 라우팅 `list_recent_experiments`

- 해석: `[{"request": {"target": "experiment", "action": "list", "experiment_pick": "oldest", "limit": 2, "output": "brief"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `list_recent_experiments(limit=2, order_by='creation_time', ascending=True)`

> 가장 오래된 실험(2개)
>
> - `Default` · ID `0` · 생성 2026-03-27 10:34:00
> - `OpenAI_Test_Project` · ID `1` · 생성 2026-03-27 10:34:53

### H04 — trial_21 run의 error 값은?

**기대:** 값 하나: 0.2446.

**개선 전** — 라우팅 `get_run_info`

- 해석: `[{"intent": "get_run_info", "arguments": {"run_name": "trial_21", "metric_key": "error"}}]`
- 호출: `llm:intent_parser` → `find_runs(run_name='trial_21')`

> `trial_21`의 `error`는 0.244573입니다.

**개선 후** — 라우팅 `get_run_info`

- 해석: `[{"request": {"target": "run", "action": "lookup", "runs": ["trial_21"], "metric": "error", "output": "value"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `find_runs(run_name='trial_21')`

> `trial_21`의 `error`는 0.244573입니다.

### H05 — Timm Hub에서 기록한 metric 종류 알려줘

**기대:** train_acc, train_loss, val_acc, val_loss.

**개선 전** — 라우팅 `list_metric_keys`

- 해석: `[{"intent": "list_metric_keys", "arguments": {"experiment_name": "Timm Hub"}}]`
- 호출: `llm:intent_parser` → `list_run_metrics('Timm Hub')`

> Timm Hub metric 키 목록(4개)
>
> - metrics.train_acc
> - metrics.train_loss
> - metrics.val_acc
> - metrics.val_loss

**개선 후** — 라우팅 `list_metric_keys`

- 해석: `[{"request": {"target": "metric_key", "action": "list", "experiment": "Timm Hub"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `list_run_metrics('Timm Hub')`

> 적용 조건: 실험 `Timm Hub`
>
> `Timm Hub` metric(4개)
>
> - train_acc
> - train_loss
> - val_acc
> - val_loss

### H06 — skittish-dove-494 run 설정값만 보여줘

**기대:** 파라미터만: lr 0.0001, optimizer AdamW, epochs 2, model hf-hub:nateraw/resnet50-oxford-iiit-pet.

**개선 전** — 라우팅 `get_run_info`

- 해석: `[{"intent": "get_run_info", "arguments": {"run_name": "skittish-dove-494"}}]`
- 호출: `llm:intent_parser` → `find_runs(run_name='skittish-dove-494')`

> `skittish-dove-494` run 정보
>
> - Experiment: `Timm Hub` (ID `6`)
> - Run name: `skittish-dove-494`
> - Run ID: `595479baa074474594f009de23d50926`
> - Status: FINISHED
> - User: dyyu
> - Start: 2026-04-08 15:04:38
> - End: 2026-04-08 15:04:39
>
> **Parameters**
>
> | Key | Value |
> |-----|-------|
> | epochs | 2 |
> | lr | 0.0001 |
> | model | hf-hub:nateraw/resnet50-oxford-iiit-pet |
> | optimizer | AdamW |
>
> **Metrics**
>
> | Key | Value |
> |-----|-------|
> | train_acc | 0.83866 |
> | train_loss | 0.527791 |
> | val_acc | 0.9467 |
> | val_loss | 0.162684 |

**개선 후** — 라우팅 `get_run_info`

- 해석: `[{"request": {"target": "run", "action": "lookup", "runs": ["skittish-dove-494"], "fields": ["params"]}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `find_runs(run_name='skittish-dove-494')`

> `skittish-dove-494` (Timm Hub)
>
> | Parameter | Value |
> |---|---|
> | epochs | 2 |
> | lr | 0.0001 |
> | model | hf-hub:nateraw/resnet50-oxford-iiit-pet |
> | optimizer | AdamW |

### H07 — ultralytics-yolo 실험 정보 알려줘

**기대:** ultralytics-yolo(ID 8) 상세. ultralytics-yolo-hpo와 혼동하지 않아야 한다.

**개선 전** — 라우팅 `get_experiment`

- 해석: `[{"intent": "get_experiment", "arguments": {"experiment_name": "ultralytics-yolo"}}]`
- 호출: `llm:intent_parser` → `list_experiment_names()` → `get_experiment('ultralytics-yolo')`

> `ultralytics-yolo` 실험 정보
>
> | Name | Experiment ID | Lifecycle Stage | Created Date | Created Time |
> |------|----------------|-----------------|--------------|--------------|
> | ultralytics-yolo | 8 | active | 2026-10-14 | 15:05:31 |
>
> - Artifact Location: `/home/dyyu/MLflow/artifacts/8`
> - Tags: -

**개선 후** — 라우팅 `get_experiment`

- 해석: `[{"request": {"target": "experiment", "action": "lookup", "experiment": "ultralytics-yolo", "output": "full"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_experiment('ultralytics-yolo')`

> `ultralytics-yolo` 실험 정보
>
> | Name | Experiment ID | Lifecycle Stage | Created Date | Created Time |
> |------|----------------|-----------------|--------------|--------------|
> | ultralytics-yolo | 8 | active | 2026-10-14 | 15:05:31 |
>
> - Artifact Location: `/home/dyyu/MLflow/artifacts/8`
> - Tags: -

### H08-1 (세션 H08) — Deep Learning Experiment run 목록 보여줘

**기대:** run 3개 목록.

**개선 전** — 라우팅 `get_runs`

- 해석: `[{"intent": "get_runs", "arguments": {"experiment_name": "Deep Learning Experiment"}}]`
- 호출: `llm:intent_parser` → `get_runs('Deep Learning Experiment')`

> `Deep Learning Experiment` run **3개** (미리보기 **3개**)
>
> | Name | Score | Status | Date | Time |
> |------|-------|--------|------|------|
> | carefree-moth-973 | - | FINISHED | 2026-10-14 | 10:21:37 |
> | burly-kit-617 | - | FINISHED | 2026-04-10 | 14:40:10 |
> | stylish-carp-904 | - | FINISHED | 2026-03-30 | 11:09:24 |

**개선 후** — 라우팅 `get_runs`

- 해석: `[{"request": {"target": "run", "action": "list", "experiment": "Deep Learning Experiment", "output": "table"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('Deep Learning Experiment')`

> 적용 조건: 실험 `Deep Learning Experiment`
>
> run 3개
>
> | Run | Status | Start |
> |---|---|---|
> | carefree-moth-973 | FINISHED | 2026-10-14 10:21:37 |
> | burly-kit-617 | FINISHED | 2026-04-10 14:40:10 |
> | stylish-carp-904 | FINISHED | 2026-03-30 11:09:24 |

### H08-2 (세션 H08) — 거기서 val_accuracy 제일 높은 건?

**기대:** 실험 이어받음 → burly-kit-617(77.95).

**개선 전** — 라우팅 `search_runs`

- 해석: `[{"intent": "search_runs", "arguments": {"experiment_name": "Deep Learning Experiment", "metric_key": "val_accuracy", "top_k": 1, "ascending": false}}, {"search_filters": {"recent_days": null, "status": null, "user": null, "tag_key": null, "tag_value": null}}]`
- 호출: `llm:intent_parser` → `get_runs('Deep Learning Experiment')` → `llm:search_filter_parser` → `list_run_metrics('Deep Learning Experiment')`

> val_accuracy가 없습니다. Deep Learning Experiment의 metric 키 목록을 조회해 주세요.

**개선 후** — 라우팅 `search_runs`

- 해석: `[{"request": {"target": "run", "action": "list", "experiment": "Deep Learning Experiment", "metric": "val_accuracy", "sort_by": "metric", "order": "desc", "limit": 1, "output": "brief", "inherited": ["experiment"]}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `get_runs('Deep Learning Experiment')`

> 적용 조건: 실험 `Deep Learning Experiment`(이전 대화) · 정렬 `val_accuracy` 높은 순 · 1개
>
> `burly-kit-617` — `val_accuracy` 77.95 (FINISHED, 2026-04-10 14:40:10)

### H09 — yolo26n-hpo-01 run 이름 바꿔줘

**기대:** 지원 범위 밖(조회 전용) 안내.

**개선 전** — 라우팅 `mlflow_mcp`

- 해석: `[{"intent": "mlflow_mcp", "arguments": {"run_name": "yolo26n-hpo-01"}}]`
- 호출: `llm:intent_parser` → `find_runs(run_name='yolo26n-hpo-01')` → `llm:run_answer`

> run 정보에 근거 없음

**개선 후** — 라우팅 `unsupported`

- 해석: `[{"request": {"target": "run", "action": "unsupported", "experiment": "yolo26n-hpo-01", "unsupported_reason": "run 이름 수정"}}]`
- 호출: `llm:request_parser`

> 요청하신 작업(run 이름 수정)은 이 챗봇에서 할 수 없습니다. 이 챗봇은 MLflow를 조회만 합니다.
> 할 수 있는 것: 실험·run 목록과 상세 조회, 조건 검색, run 비교·요약·평가, 첨부 보고서 질의응답

### H10 — loud-gnat-654랑 skittish-dove-494 비교해줘

**기대:** 두 run metric 비교(val_acc 0.946 vs 0.9467 등). 파라미터는 같음.

**개선 전** — 라우팅 `None`

- 해석: `[{"intent": "mlflow_mcp", "arguments": {"run_name": "loud-gnat-654", "run_name_2": "skittish-dove-494"}}]`
- 호출: `llm:intent_parser`

> (빈 답변)

**개선 후** — 라우팅 `compare_runs`

- 해석: `[{"request": {"target": "run", "action": "compare", "runs": ["loud-gnat-654", "skittish-dove-494"], "output": "value"}}]`
- 호출: `llm:request_parser` → `list_experiment_names()` → `find_runs(run_name='loud-gnat-654')` → `find_runs(run_name='skittish-dove-494')`

> | 항목 | loud-gnat-654 | skittish-dove-494 |
> |---|---|---|
> | train_acc | 0.83844 | 0.83866 |
> | train_loss | 0.526638 | 0.527791 |
> | val_acc | 0.946 | 0.9467 |
> | val_loss | 0.163361 | 0.162684 |
