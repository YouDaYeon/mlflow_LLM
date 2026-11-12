"""요청 해석 개선 평가용 채팅 질문 세트.

각 케이스는 질문의 의미 차이(대상, 개수, 기간, 필터, 정렬, 답변 범위)가
조회와 답변에 반영되는지를 본다. 문장이 달라졌는지는 보지 않는다.

session 값이 같은 케이스는 같은 대화 세션에서 turn 순서대로 실행한다.
expected는 사람이 채점할 때 쓰는 기대 동작이다. 정답 값은 2026-11-11 MLflow 데이터 기준이다.
"""

from __future__ import annotations

# 보고서 근거 질문용 짧은 보고서. 실제 생성 보고서 대신 사실을 고정한다.
REPORT_FIXTURE = (
    "# ultralytics-yolo-hpo 실험 보고서\n"
    "## 요약\n"
    "- 대상 실험: ultralytics-yolo-hpo (run 20개, 모두 FINISHED)\n"
    "- 기준 metric: metrics/mAP50B\n"
    "- 최고 run: yolo26n-hpo-09 (mAP50B 0.9062, lr0=0.0072)\n"
    "- 최근 run 두 개: yolo26n-hpo-20 (mAP50B 0.9017, lr0=0.0100), "
    "yolo26n-hpo-19 (mAP50B 0.9013, lr0=0.0095)\n"
    "## 분석\n"
    "- 학습률이 0.009 이상인 run은 mAP50B가 0.902 안팎에서 정체했다.\n"
    "- yolo26n-hpo-20은 yolo26n-hpo-19보다 mAP50B가 0.0005 높지만 차이는 run 간 변동 범위 안이다.\n"
    "## 다음 액션\n"
    "- lr0를 0.006~0.008 구간에서 다시 탐색한다.\n"
)

# problem: 사용자 분석 문서의 P1~P7
# expected: 조회 조건(무엇을 몇 개, 어떤 기준으로)과 답변 범위(무엇을 얼마나 보여줄지)
CASES: list[dict] = [
    # ── A. 개수와 답변 범위 ───────────────────────────────
    {
        "id": "A01",
        "group": "개수·답변 범위",
        "problem": ["P3", "P5"],
        "message": "최근 실험 알려줘",
        "expected": "최근 생성 실험 목록(기본 개수, 최신순). 이름 목록 형태.",
    },
    {
        "id": "A02",
        "group": "개수·답변 범위",
        "problem": ["P3", "P5"],
        "message": "가장 최근 실험 하나 알려줘",
        "expected": "실험 1개만: ultralytics-yolo-hpo. A01과 조회 개수·답변 범위가 달라야 한다.",
    },
    {
        "id": "A03",
        "group": "개수·답변 범위",
        "problem": ["P2", "P3"],
        "message": "최근 실험 3개만 보여줘",
        "expected": "최신순 3개: ultralytics-yolo-hpo, ultralytics-yolo, Timm Hub.",
    },
    {
        "id": "A04",
        "group": "개수·답변 범위",
        "problem": ["P2", "P3"],
        "message": "가장 오래된 실험 뭐야?",
        "expected": "생성 오름차순 1개: Default 또는 OpenAI_Test_Project(같은 시각 생성).",
    },
    {
        "id": "A05",
        "group": "개수·답변 범위",
        "problem": ["P5"],
        "message": "가장 최근 실험의 상세 정보 알려줘",
        "expected": "ultralytics-yolo-hpo 1개의 상세(ID, 생성시각, artifact 위치 등).",
    },
    {
        "id": "A06",
        "group": "개수·답변 범위",
        "problem": ["P5"],
        "message": "실험이 총 몇 개야?",
        "expected": "개수만: 9개. 전체 목록 나열은 불필요.",
    },
    {
        "id": "A07",
        "group": "개수·답변 범위",
        "problem": ["P5"],
        "message": "ultralytics-yolo-hpo 실험의 run 이름만 나열해줘",
        "expected": "run 20개의 이름만. 점수·상태·날짜 표는 불필요.",
    },
    {
        "id": "A08",
        "group": "개수·답변 범위",
        "problem": ["P5"],
        "message": "yolo26n-hpo-20 run의 metrics/mAP50B 값만 알려줘",
        "expected": "값 하나: 0.9017. 전체 상세 표는 불필요.",
    },
    {
        "id": "A09",
        "group": "개수·답변 범위",
        "problem": ["P5"],
        "message": "yolo26n-hpo-20 run 상세 알려줘",
        "expected": "run 전체 상세(파라미터·metric 포함).",
    },
    {
        "id": "A10",
        "group": "개수·답변 범위",
        "problem": ["P5"],
        "message": "burly-kit-617 run 핵심만 요약해줘",
        "expected": "핵심 요약 몇 줄(주요 metric, 설정). 전체 파라미터 표 반복은 불필요.",
    },
    {
        "id": "A11",
        "group": "개수·답변 범위",
        "problem": ["P5"],
        "message": "carefree-moth-973 run 학습 잘 된 편이야?",
        "expected": "평가: 결론 먼저 + 근거 metric(val_accuracy 77.81 등). 같은 실험 다른 run과 비교하면 더 좋음.",
    },
    # ── B. 정렬·개수·필터 ────────────────────────────────
    {
        "id": "B01",
        "group": "정렬·개수·필터",
        "problem": ["P2"],
        "message": "Timm Hub에서 val_acc 가장 높은 run 2개 알려줘",
        "expected": "val_acc 내림차순 2개: capricious-moth-486(0.9569), mysterious-yak-903(0.9554).",
    },
    {
        "id": "B02",
        "group": "정렬·개수·필터",
        "problem": ["P2", "P7"],
        "message": "Hyperparameter Tuning Experiment에서 error가 가장 낮은 run 3개 보여줘",
        "expected": "error 오름차순 3개: trial_21(0.2446), trial_28(0.2457), trial_15(0.2465). "
        "error 값이 없는 FAILED/study run이 앞에 오면 안 된다.",
    },
    {
        "id": "B03",
        "group": "정렬·개수·필터",
        "problem": ["P2", "P7"],
        "message": "ultralytics-yolo-hpo 실험에서 가장 최근 run 3개 보여줘",
        "expected": "시작 시각 최신순 3개: yolo26n-hpo-20, 19, 18 (모두 같은 날짜, 시각으로 정렬). "
        "metric 순위(top_k)와 혼동하지 않아야 한다.",
    },
    {
        "id": "B04",
        "group": "정렬·개수·필터",
        "problem": ["P2", "P7"],
        "message": "Hyperparameter Tuning Experiment에서 실패한 run 보여줘",
        "expected": "status=FAILED 2개: trial_2, study. metric 없이도 검색되어야 한다.",
    },
    {
        "id": "B05",
        "group": "정렬·개수·필터",
        "problem": ["P5", "P7"],
        "message": "ultralytics-yolo-hpo에서 최근 30일 안에 돌린 run 몇 개야?",
        "expected": "기간 필터(30일) 적용 후 개수만: 20개.",
    },
    {
        "id": "B06",
        "group": "정렬·개수·필터",
        "problem": ["P6"],
        "message": "Timm Hub에서 f1_score 높은 run 3개 보여줘",
        "expected": "f1_score metric이 없다고 안내하고 사용 가능한 metric(val_acc 등)을 제시. 다른 metric으로 몰래 대체 금지.",
    },
    # ── C. 복합 요청 ─────────────────────────────────────
    {
        "id": "C01",
        "group": "복합 요청",
        "problem": ["P1", "P6"],
        "message": "Timm Hub에서 val_acc 기준 최고 run과 최저 run을 비교해줘",
        "expected": "최고 capricious-moth-486(0.9569) vs 최저 loud-gnat-654(0.9460) 비교.",
    },
    {
        "id": "C02",
        "group": "복합 요청",
        "problem": ["P1", "P6"],
        "message": "가장 최근 실험에서 mAP50B가 제일 높은 run 알려줘",
        "expected": "최근 실험(ultralytics-yolo-hpo) 선택 → mAP50B 최고 1개: yolo26n-hpo-09(0.9062).",
    },
    {
        "id": "C03",
        "group": "복합 요청",
        "problem": ["P1", "P6"],
        "message": "yolo26n-hpo-09와 yolo26n-hpo-20의 mAP50B 차이 알려줘",
        "expected": "두 run 값(0.9062, 0.9017)과 차이(약 0.0044).",
    },
    {
        "id": "C04",
        "group": "복합 요청",
        "problem": ["P1"],
        "message": "보고서 기준으로 최근 run 두 개를 비교해줘",
        "report_text": REPORT_FIXTURE,
        "expected": "보고서 근거로 yolo26n-hpo-20 vs yolo26n-hpo-19 비교(0.9017 vs 0.9013, 차이는 변동 범위 안).",
    },
    {
        "id": "C05",
        "group": "복합 요청",
        "problem": ["P1", "P5"],
        "added": True,
        "message": "Deep Learning Experiment에서 val_accuracy 제일 높은 run의 파라미터만 알려줘",
        "expected": "val_accuracy 최고 run(burly-kit-617, 77.95) 선택 → 파라미터만: batch_size 64, epochs 5, "
        "hidden_units [512, 512], learning_rate 0.001, model_type MLP, optimizer SGD.",
    },
    {
        "id": "C06",
        "group": "복합 요청",
        "problem": ["P1", "P6"],
        "added": True,
        "message": "보고서 기준으로 GPU 메모리 사용량이 가장 적은 run 알려줘",
        "report_text": REPORT_FIXTURE,
        "expected": "보고서에 GPU 메모리 정보가 없으므로 '리포트에 근거 없음'. MLflow 조회나 수치 생성 금지.",
    },
    # ── D. 모호·미지원 요청 ──────────────────────────────
    {
        "id": "D01",
        "group": "모호·미지원",
        "problem": ["P6"],
        "message": "그 run 성능 어때?",
        "expected": "새 대화라 대상 run이 없다. 어떤 run인지 확인 질문.",
    },
    {
        "id": "D02",
        "group": "모호·미지원",
        "problem": ["P6"],
        "message": "성능 제일 좋은 run 알려줘",
        "expected": "실험·기준 metric이 없다. 확인 질문, 또는 가정을 명시하고 답.",
    },
    {
        "id": "D03",
        "group": "모호·미지원",
        "problem": ["P6"],
        "message": "ultralytics-yolo-hpo 실험 삭제해줘",
        "expected": "지원 범위 밖(조회 전용)이라고 안내. 실제 삭제 시도 금지.",
    },
    # ── M. 멀티턴: 조건 유지·변경·해제 ────────────────────
    {
        "id": "M1-1",
        "group": "멀티턴",
        "problem": ["P4"],
        "session": "M1",
        "message": "Timm Hub run 중 val_acc 상위 2개 보여줘",
        "expected": "val_acc 내림차순 2개.",
    },
    {
        "id": "M1-2",
        "group": "멀티턴",
        "problem": ["P4"],
        "session": "M1",
        "message": "전체 run 보여줘",
        "expected": "실험(Timm Hub)은 유지, 개수 제한(2) 해제 → run 4개 전부.",
    },
    {
        "id": "M1-3",
        "group": "멀티턴",
        "problem": ["P4"],
        "session": "M1",
        "message": "그 중에 val_loss가 제일 낮은 건?",
        "expected": "Timm Hub 유지, metric을 val_loss로 변경, 오름차순 1개.",
    },
    {
        "id": "M2-1",
        "group": "멀티턴",
        "problem": ["P4"],
        "session": "M2",
        "message": "Hyperparameter Tuning Experiment에서 실패한 run 보여줘",
        "expected": "FAILED 2개: trial_2, study.",
    },
    {
        "id": "M2-2",
        "group": "멀티턴",
        "problem": ["P4"],
        "session": "M2",
        "message": "완료된 건 몇 개야?",
        "expected": "실험 유지, 상태를 FINISHED로 변경, 개수만: 33개.",
    },
    {
        "id": "M2-3",
        "group": "멀티턴",
        "problem": ["P4"],
        "session": "M2",
        "added": True,
        "message": "상태 조건 빼고 전체는 몇 개야?",
        "expected": "실험 유지, 상태 조건 해제, 개수만: 35개.",
    },
    {
        "id": "M3-1",
        "group": "멀티턴",
        "problem": ["P4"],
        "session": "M3",
        "message": "가장 최근 실험 하나 알려줘",
        "expected": "ultralytics-yolo-hpo 1개.",
    },
    {
        "id": "M3-2",
        "group": "멀티턴",
        "problem": ["P4"],
        "session": "M3",
        "message": "그 실험 run 중 최근 3개만 보여줘",
        "expected": "실험 ultralytics-yolo-hpo 이어받음 → 시작 최신순 3개: yolo26n-hpo-20, 19, 18.",
    },
    {
        "id": "M4-1",
        "group": "멀티턴",
        "problem": ["P4", "P5"],
        "session": "M4",
        "added": True,
        "message": "yolo26n-hpo-09 run의 mAP50B 값 알려줘",
        "expected": "값 하나: 0.9062 (metric 이름을 접두사 없이 말함).",
    },
    {
        "id": "M4-2",
        "group": "멀티턴",
        "problem": ["P4", "P5"],
        "session": "M4",
        "added": True,
        "message": "같은 run의 recall은?",
        "expected": "run yolo26n-hpo-09 이어받음, metric recall → metrics/recallB 값 하나: 0.6613.",
    },
]
