# 온보딩 가이드

MLflow 실험 데이터를 조회하고, LLM으로 분석 보고서를 만들고, 멀티턴 Q&A 챗봇으로 후속 질문을 할 수 있는 프로젝트입니다.

이 문서만 읽어도 **전체 구조·데이터 흐름·핵심 기능**을 파악할 수 있도록 작성했습니다.  
세부 실행은 `README.md`, 멀티턴 설계 메모는 `multi_trun.md`를 참고하세요.

---

## 1. 한눈에 보기

```text
[Streamlit UI]  frontend/app_2.py
       │  HTTP (API_BASE_URL)
       ▼
[FastAPI]       backend/app/
       ├── /mlflow/*   → 원격 MLflow Tracking Server
       ├── /report/*   → LLM(보고서용) + PostgreSQL(reports)
       └── /chat/*     → LLM(챗/의도분류) + MLflow/MCP + PostgreSQL(chat_*)
```

| 구성 | 역할 | 기본 포트 |
|------|------|-----------|
| Frontend | Streamlit UI | `8501` |
| Backend | FastAPI | Docker `8010` / 로컬 `8011` |
| PostgreSQL | 보고서·채팅 이력 | 호스트 `5435` → 컨테이너 `5432` |
| pgAdmin | DB GUI | `5050` |
| MLflow Tracking | 실험/런 원본 데이터 | `.env`의 `MLFLOW_TRACKING_URI` |
| LLM | 보고서 생성 / 챗·intent | `LLAMA_*` · `CHAT_*` |

**역할 분리**

- MLflow Tracking = 실험 원본 (이 프로젝트 DB에 런을 복제하지 않음)
- PostgreSQL = 앱이 만든 **보고서** + **챗 세션/메시지**
- `LLAMA_*` ≈ 보고서 생성, `CHAT_*` ≈ intent 분류·report Q&A 등 챗 경로

---

## 2. 디렉터리 구조

```text
mlflow_LLM_dy/
├── backend/app/
│   ├── main.py              # FastAPI 진입, 기동 시 init_db()
│   ├── config.py            # Settings (.env)
│   ├── schemas.py           # 요청/응답 모델
│   ├── api/                 # mlflow_router, report_router, chat_router
│   ├── services/            # mlflow / report / chat / repository
│   └── db/                  # models, session, init_db
├── frontend/
│   ├── app_2.py             # ★ 메인 UI
│   └── app.py               # 구버전 (참고용)
├── docker-compose.yml
├── run.sh                   # 로컬 실행 메모
├── .env / .env.example
├── README.md
└── multi_trun.md            # 멀티턴·세션 영속화 상세 메모
```

**코드를 볼 때 우선순위**

1. `frontend/app_2.py` — 사용자 시나리오 전부
2. `backend/app/api/*_router.py` — API 경계
3. `backend/app/services/*` — 실제 비즈니스 로직
4. `backend/app/db/models.py` — 저장 스키마

---

## 3. 실행 방법

### Docker (전체 스택)

```bash
cp .env .env   # 값 채우기
docker compose up -d --build
```

- UI: `http://localhost:8501`
- API health: `http://localhost:8010/health`
- pgAdmin: `http://localhost:5050`

### 로컬 개발 (DB만 Docker, 앱은 호스트)

1. Postgres(+pgAdmin) 기동
2. `.env`의 `DATABASE_URL`은 `127.0.0.1:5435/...` 사용  
   `API_BASE_URL=http://localhost:8011` (프론트 → 로컬 백엔드)
3. 백엔드 / 프론트:

```bash
conda activate mlflow310   # 또는 프로젝트 환경
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8011 --reload
python -m streamlit run frontend/app_2.py
```

기동 시 `init_db()`가 테이블을 없으면 생성합니다 (`reports`, `chat_sessions`, `chat_messages`).

---

## 4. 핵심 기능과 흐름

### 4.1 MLflow 실험 조회

**목적:** Tracking Server의 실험/런/메트릭을 UI에 보여 준다.

```text
사이드바에서 실험·metric·top-k 선택
  → GET /mlflow/names | /metric-keys | /top-runs | /experiment ...
  → 메인에 runs 테이블 / 실험 정보 다이얼로그
```

| 계층 | 파일 |
|------|------|
| API | `api/mlflow_router.py` |
| 로직 | `services/mlflow_service.py` |

---

### 4.2 보고서 생성 · 조회 · 수정 · 삭제

**목적:** top runs(최대 3개)를 바탕으로 한국어 분석 보고서를 만들고 DB에 보관한다.

```text
[생성] POST /report/
       → report_service (LLAMA_*) 로 본문 생성
       → reports 테이블 저장
       → 메인 화면에 표시

[목록/보기] GET /report/ , GET /report/{id}
[수정]      PATCH /report/{id}   (본문·제목, updated_at 갱신)
[삭제]      DELETE /report/{id}
```

UI (`app_2.py`)

- 사이드바: 생성, 저장된 보고서 목록·보기·삭제
- 메인: 보고서 본문, **수정하기** (저장 시 수정 시각 반영)

| 계층 | 파일 |
|------|------|
| API | `api/report_router.py` |
| 로직 | `services/report_service.py`, `repository_service.py` |
| 모델 | `db/models.py` → `Report` |

---

### 4.3 Q&A 챗봇 (멀티턴 + 세션 영속화)

**목적:** 실험/보고서 맥락으로 질문하고, 대화를 이어가며 DB에 이력을 남긴다.

#### 요청 한 번의 흐름 (`POST /chat/ask`)

```text
1. session_id 없으면 세션 생성 / 있으면 검증
2. DB에서 최근 N턴 history 로드 (없으면 프론트 history fallback)
3. classify_intent (CHAT_*, history 포함)
4. intent 후처리
   intent args → UI payload → 직전 assistant metadata
   로 experiment_name / metric_key / top_k / ascending 보완
5. 분기
   · 실험 조회 intent → mlflow_service 직접 호출 (결정론적)
   · report_qa        → 보고서 텍스트 + history 로 LLM 답변
   · 그 외            → MLflow MCP agent
6. user/assistant 메시지를 DB에 append
7. 첫 턴이면 LLM으로 세션 title 생성
8. ChatResponse(answer, intent, metadata, session_id)
```

#### UI에서의 세션

- 대화 목록에서 불러오기 / `(현재 대화)`·**새 대화**로 초기화 / 삭제
- 참조 문맥(caption): 사이드바·보고서 + intent 후처리된 experiment/metric/top_k
- dialog 안에서는 `st.rerun()`을 쓰면 창이 닫히므로, 목록 전환은 `on_change`/`on_click`으로 state만 갱신

| 계층 | 파일 |
|------|------|
| API | `api/chat_router.py` |
| 로직 | `services/chat_service.py`, `repository_service.py`, `mlflow_service.py`(MCP) |
| 모델 | `ChatSession`, `ChatMessageRow` |

---

## 5. 데이터 모델 (PostgreSQL)

DB 이름 기본값: `mlflow_llm` (`POSTGRES_DB` / `DATABASE_URL` path).

| 테이블 | 역할 |
|--------|------|
| `reports` | 보고서 본문·제목·실험/metric·model·top_runs·metadata·created_at·**updated_at** |
| `chat_sessions` | 대화 방 (title, created_at, updated_at) |
| `chat_messages` | 턴 (role, content, intent, metadata, session FK CASCADE) |

MLflow 런/메트릭은 **저장하지 않습니다.** 필요할 때마다 Tracking Server를 조회합니다.

---

## 6. API 맵 (요약)

### MLflow — `/mlflow`

`names`, `experiment`, `runs`, `top-runs`, `metric-keys`, `mcp-health`, …

### Report — `/report`

| Method | Path | 설명 |
|--------|------|------|
| POST | `/` | 생성·저장 |
| GET | `/` | 목록 |
| GET | `/{id}` | 단건 |
| PATCH | `/{id}` | 수정 |
| DELETE | `/{id}` | 삭제 |

### Chat — `/chat`

| Method | Path | 설명 |
|--------|------|------|
| POST | `/ask` | 질문 (세션·history·영속화) |
| POST | `/sessions` | 빈 세션 생성 |
| GET | `/sessions` | 세션 목록 |
| GET | `/sessions/{id}/messages` | 메시지 목록 |
| DELETE | `/sessions/{id}` | 세션 삭제 |

---

## 7. 설정 (`.env`)

| 변수 | 의미 |
|------|------|
| `MLFLOW_TRACKING_URI` | MLflow Tracking Server |
| `DATABASE_URL` | PostgreSQL (로컬은 `:5435`, Docker backend는 `postgres:5432`) |
| `LLAMA_*` | 보고서 생성 LLM |
| `CHAT_*` | 챗/intent LLM (`CHAT_HISTORY_LIMIT` 포함) |
| `API_BASE_URL` | 프론트가 호출할 백엔드 (Docker: `http://backend:8010`, 로컬: `http://localhost:8011`) |
| `PGADMIN_*` | pgAdmin 로그인 |

모델 id는 LLM 서버의 `/v1/models` 실제 id와 맞춰야 합니다. `base_url`은 보통 `.../v1`까지 포함합니다.

---

## 8. UI 화면 구성 (`app_2.py`)

| 영역 | 내용 |
|------|------|
| 사이드바 | 실험/metric/top-k 선택, 실험 정보, 보고서 생성, 저장 보고서 목록·보기·삭제 |
| 메인 | runs 테이블, 보고서 본문(수정/수정시각) |
| 플로팅 | QA 챗봇 dialog (세션 목록·멀티턴·참조 문맥) |

프론트는 상태를 `st.session_state`에 두고, 영속 데이터는 전부 백엔드 API → PostgreSQL로 갑니다.

---

## 9. 새 기능 넣을 때

1. **API 계약**부터: `schemas.py` + 해당 `*_router.py`
2. **로직**은 `services/` (가능하면 라우터는 얇게)
3. **저장**이 필요하면 `db/models.py` + `repository_service.py` (+ `init_db` 컬럼 보강)
4. **UI**는 `frontend/app_2.py`에서 해당 API 호출
5. 챗 맥락이 필요하면 `/chat/ask`의 history·metadata·session 흐름을 먼저 따라가 보기

---

## 10. 관련 문서

| 문서 | 내용 |
|------|------|
| `README.md` | 아키텍처 그림, Docker 실행, API 요약 |
| `multi_trun.md` | 멀티턴·세션 영속화 구현 단계·검증 시나리오 |
| `.env.example` | 환경변수 템플릿 |
