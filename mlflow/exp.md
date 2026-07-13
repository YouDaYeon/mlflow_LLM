# 가상 MLflow 실험 로그 생성

## 목적
`MLFLOW_TRACKING_URI`로 연결된 MLflow에 간단한 실험 run을 몇 개 올려서, UI에서 볼 수 있는 가상 로그를 만든다.

## 진행 순서

1. `.env`의 `MLFLOW_TRACKING_URI`가 올바른지 확인한다.
2. `mlflow/run_dummy_exp.py`에 DistilBERT(`distilbert-base-uncased`) 짧은 실험 스크립트를 만든다.
3. 스크립트에서 experiment를 하나 정하고, learning rate / batch size를 바꿔 run 여러 개에 params / metrics / tags를 남긴다.
4. 스크립트를 실행해서 tracking 서버에 로그를 보낸다.
5. MLflow UI에서 experiment / run이 보이는지 확인한다.
