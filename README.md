# mlflow_LLM_dy

MLflow 실험 데이터를 조회하고, LLM 기반 보고서 생성 및 Q&A를 제공하는 프로젝트입니다.

## 구성

- `backend`: FastAPI API 서버
- `frontend`: Streamlit UI
- `docker-compose.yml`: Docker 통합 실행 설정

## Docker 실행

### 1) 사전 준비

- Docker Engine + Docker Compose 사용 가능 환경
- (선택) 외부 MLflow Tracking Server / LLM endpoint 접근 가능 네트워크

### 2) 환경변수 설정 (선택)

기본값으로도 실행되지만, 운영/개발 환경에 맞게 값을 바꾸려면:

```bash
cp .env.example .env
```

필요 시 `.env`에서 아래 값을 수정합니다.

- `MLFLOW_TRACKING_URI`
- `LLAMA_BASE_URL`
- `LLAMA_API_KEY`
- `LLAMA_MODEL`
- `LLAMA_TEMPERATURE`
- `API_BASE_URL` (기본값: `http://backend:8010`)

### 3) 빌드 및 실행

프로젝트 루트에서 실행:

```bash
docker compose build
docker compose up -d
```

### 4) 동작 확인

- Backend health check: `http://localhost:8010/health`
- Frontend (Streamlit): `http://localhost:8501`

### 5) 종료

```bash
docker compose down
```

## API 요약

- MLflow API
  - `GET /mlflow/names`
  - `GET /mlflow/experiment`
  - `GET /mlflow/runs`
  - `GET /mlflow/top-runs`
  - `GET /mlflow/metric-keys`
- Report API
  - `POST /report/`
- Chat API
  - `POST /chat/ask`

