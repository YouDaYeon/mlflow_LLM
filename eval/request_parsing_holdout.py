"""요청 해석 평가 — 홀드아웃 질문.

개선 작업 중에는 보지 않고, 개선이 끝난 뒤 한 번만 실행해 평가 세트 과적합 여부를 본다.
정답 값은 2026-11-11 MLflow 데이터 기준이다.
"""

from __future__ import annotations

CASES: list[dict] = [
    {
        "id": "H01",
        "group": "홀드아웃",
        "problem": ["P5"],
        "message": "Timm Hub에 run 몇 개 있어?",
        "expected": "개수만: 4개.",
    },
    {
        "id": "H02",
        "group": "홀드아웃",
        "problem": ["P2", "P5"],
        "message": "ultralytics-yolo-hpo에서 mAP50-95B 상위 3개 run 이름만 알려줘",
        "expected": "yolo26n-hpo-09, yolo26n-hpo-08, yolo26n-hpo-05 (이름만).",
    },
    {
        "id": "H03",
        "group": "홀드아웃",
        "problem": ["P2", "P3"],
        "message": "가장 오래된 실험 2개 알려줘",
        "expected": "생성 오름차순 2개: Default, OpenAI_Test_Project(같은 시각).",
    },
    {
        "id": "H04",
        "group": "홀드아웃",
        "problem": ["P5"],
        "message": "trial_21 run의 error 값은?",
        "expected": "값 하나: 0.2446.",
    },
    {
        "id": "H05",
        "group": "홀드아웃",
        "problem": ["P5"],
        "message": "Timm Hub에서 기록한 metric 종류 알려줘",
        "expected": "train_acc, train_loss, val_acc, val_loss.",
    },
    {
        "id": "H06",
        "group": "홀드아웃",
        "problem": ["P5"],
        "message": "skittish-dove-494 run 설정값만 보여줘",
        "expected": "파라미터만: lr 0.0001, optimizer AdamW, epochs 2, model hf-hub:nateraw/resnet50-oxford-iiit-pet.",
    },
    {
        "id": "H07",
        "group": "홀드아웃",
        "problem": ["P1"],
        "message": "ultralytics-yolo 실험 정보 알려줘",
        "expected": "ultralytics-yolo(ID 8) 상세. ultralytics-yolo-hpo와 혼동하지 않아야 한다.",
    },
    {
        "id": "H08-1",
        "group": "홀드아웃",
        "problem": ["P4"],
        "session": "H08",
        "message": "Deep Learning Experiment run 목록 보여줘",
        "expected": "run 3개 목록.",
    },
    {
        "id": "H08-2",
        "group": "홀드아웃",
        "problem": ["P4"],
        "session": "H08",
        "message": "거기서 val_accuracy 제일 높은 건?",
        "expected": "실험 이어받음 → burly-kit-617(77.95).",
    },
    {
        "id": "H09",
        "group": "홀드아웃",
        "problem": ["P6"],
        "message": "yolo26n-hpo-01 run 이름 바꿔줘",
        "expected": "지원 범위 밖(조회 전용) 안내.",
    },
    {
        "id": "H10",
        "group": "홀드아웃",
        "problem": ["P1"],
        "message": "loud-gnat-654랑 skittish-dove-494 비교해줘",
        "expected": "두 run metric 비교(val_acc 0.946 vs 0.9467 등). 파라미터는 같음.",
    },
]
