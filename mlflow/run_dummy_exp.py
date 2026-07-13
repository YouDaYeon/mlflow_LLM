"""DistilBERT로 짧은 더미 실험을 돌려 MLflow에 run 로그를 남긴다."""

from __future__ import annotations

import os
import random
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MODEL_NAME = "distilbert-base-uncased"
EXPERIMENT_NAME = "dummy-distilbert"
N_STEPS = 5
TEXTS = [
    "this movie was great",
    "terrible experience overall",
    "absolutely loved it",
    "would not recommend",
    "pretty average film",
    "best one I have seen",
]
LABELS = [1, 0, 1, 0, 0, 1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def run_one(learning_rate: float, batch_size: int, seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    with mlflow.start_run(run_name=f"lr{learning_rate}_bs{batch_size}"):
        mlflow.set_tags(
            {
                "model": MODEL_NAME,
                "task": "sequence-classification",
                "purpose": "dummy-log",
            }
        )
        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_steps": N_STEPS,
                "seed": seed,
                "device": str(device),
            }
        )

        model.train()
        for step in range(1, N_STEPS + 1):
            idx = random.sample(range(len(TEXTS)), k=min(batch_size, len(TEXTS)))
            batch_texts = [TEXTS[i] for i in idx]
            batch_labels = torch.tensor([LABELS[i] for i in idx], device=device)

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(**encoded, labels=batch_labels)
            loss = outputs.loss
            preds = outputs.logits.argmax(dim=-1)
            acc = (preds == batch_labels).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mlflow.log_metrics(
                {"train_loss": float(loss.item()), "train_acc": acc},
                step=step,
            )
            print(
                f"[lr={learning_rate} bs={batch_size}] "
                f"step={step} loss={loss.item():.4f} acc={acc:.2f}"
            )

        # 마지막 스텝 기준 요약 metric
        with torch.no_grad():
            encoded_all = tokenizer(
                TEXTS, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            labels_all = torch.tensor(LABELS, device=device)
            logits = model(**encoded_all).logits
            final_loss = F.cross_entropy(logits, labels_all).item()
            final_acc = (logits.argmax(dim=-1) == labels_all).float().mean().item()

        mlflow.log_metrics({"final_loss": final_loss, "final_acc": final_acc})


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("MLFLOW_TRACKING_URI가 .env에 없습니다.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"tracking_uri={tracking_uri}")
    print(f"experiment={EXPERIMENT_NAME}, model={MODEL_NAME}")

    # 하이퍼파라미터만 바꿔 run 3개 생성
    configs = [
        {"learning_rate": 5e-5, "batch_size": 2, "seed": 0},
        {"learning_rate": 2e-5, "batch_size": 2, "seed": 1},
        {"learning_rate": 1e-5, "batch_size": 4, "seed": 2},
    ]
    for cfg in configs:
        run_one(**cfg)

    print("done.")


if __name__ == "__main__":
    main()
