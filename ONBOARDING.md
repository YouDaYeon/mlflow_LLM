# 온보딩 가이드

MLflow 실험 조회 → LLM 보고서 생성/저장 → Q&A 챗봇을 제공하는 프로젝트입니다.

**스택:** FastAPI (backend) · Streamlit (frontend) · PostgreSQL · MLflow Tracking · OpenAI-compatible LLM

---

## 1. 프로젝트 구조

```
mlflow_LLM_dy/
├── backend/app/          # FastAPI
│   ├── main.py           # 앱 진입점, DB init, 라우터 마운트
│   ├── config.py         # .env 설정
│   ├── schemas.py        # 요청/응답 모델
│   ├── api/              # mlflow / report / chat 라우터
│   ├── services/         # 비즈니스 로직
│   └── db/               # SQLAlchemy 모델·세션
├── frontend/
│   ├── app_2.py          # 메인 UI (Docker·run.sh 기준)
│   └── app.py            # 구버전 UI
├── mlflow/               # 더미 실험 헬퍼
├── docker-compose.yml    # postgres, pgadmin, backend, frontend
├── run.sh                # 로컬 실행 참고
└── .env                  # MLFLOW / DB / LLM / API 설정
```

| 진입점 | 설명 |
|--------|------|
| `docker compose up` | 전체 스택 (backend `:8010`, frontend `:8501`, postgres `:5435`) |
| `run.sh` | 로컬 uvicorn + Streamlit 참고 (backend 보통 `:8011`) |
| `GET /health` | 백엔드 헬스체크 |

---

## 2. 구현된 기능

### 2.1 MLflow 조회

- Tracking URI의 실험/런/메트릭 조회
- API: `GET /mlflow/names`, `/experiment`, `/runs`, `/top-runs`, `/metric-keys` 등
- UI: 사이드바에서 실험·메트릭·top-k 선택 후 런 테이블 표시

### 2.2 보고서 생성·저장

1. UI에서 top runs(최대 3개) 선택
2. `POST /report/` → LLM이 한국어 분석 보고서 생성
3. PostgreSQL `reports` 테이블에 저장
4. UI에서 생성 결과 표시 + 저장된 보고서 목록 조회 (`GET /report/`)

### 2.3 Q&A 챗봇 (멀티턴 + 세션 영속화)

- API: `POST /chat/ask` (`message`, `history`, `session_id` optional + report/experiment 문맥)
- 흐름:
  1. `classify_intent`로 의도 분류 (history 반영)
  2. 실험 조회 intent → MLflow 서비스 직접 호출
  3. `report_qa` → 보고서 텍스트 + history 기반 LLM 답변
  4. 그 외 → MLflow MCP agent fallback
- DB: `chat_sessions` / `chat_messages`에 이력 저장, 목록에서 재개 가능 (Step 6)

### 2.4 DB 연결

| 항목 | 내용 |
|------|------|
| DB | PostgreSQL (`DATABASE_URL`, SQLAlchemy + psycopg) |
| 용도 | 보고서 + **채팅 세션/메시지** (MLflow 런은 Tracking Server) |
| 테이블 | `reports`, `chat_sessions`, `chat_messages` |
| 초기화 | 앱 기동 시 `init_db()` → `create_all` |

MLflow 데이터는 원격 Tracking Server(`MLFLOW_TRACKING_URI`)에 존재합니다.

### 2.5 주요 설정 변수 (이름만)

`MLFLOW_TRACKING_URI`, `DATABASE_URL`, `LLAMA_*`, `CHAT_*`, `API_BASE_URL`

---

## 3. 추가 기능: 챗봇 멀티턴

### 현황

| 계층 | 상태 |
|------|------|
| Frontend | `history` 전송 + `chat_session_id` / 목록에서 이력 재개 |
| Backend `/chat/ask` | `history`·`session_id` 수신, DB recent N턴 우선 |
| DB | `chat_sessions` / `chat_messages` 저장 (Step 6 구현) |

목표: 후속 질문이 이전 맥락을 이해하고, 저장·조회·재개가 가능하게 만들기.

### 권장 접근 (진행 순서)

**1~5단계:** 프론트 history → intent/LLM 멀티턴  
**6단계:** DB 세션 영속화 (구현됨)  
**선택(미구현):** report 연결, Alembic, URL 복원 등  
**Step 6.6:** 실험 intent 후처리 · 세션 삭제 · LLM title (구현됨)

---

### 구현 단계

#### Step 1. 스키마 확장

`backend/app/schemas.py`의 `ChatRequest`에 대화 이력 필드 추가.

```python
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)  # 직전 턴들
    report_text: str | None = None
    experiment_name: str | None = None
    metric_key: str | None = None
```

- `history`에는 **현재 `message` 제외**한 이전 턴만 넣는 것을 권장 (중복 방지).
- 토큰 한도를 위해 최근 N턴(예: 6~10)만 전송.

#### Step 2. Frontend에서 history 전달

`frontend/app_2.py`의 `_append_and_answer_chat` 수정:

1. `st.session_state["chat_messages"]`에서 시스템 인사말 제외한 직전 메시지 추출
2. 최근 6개만 슬라이스
3. `chat_payload["history"]`에 `{role, content}` 리스트로 포함
4. 기존처럼 `message`에 현재 질문만 유지

#### Step 3. Backend 라우터에서 history 전달

`chat_router.py` → 각 answer 함수에 `history` 인자 추가.

- `classify_intent(message, history=...)`
- `answer_report_question(..., history=...)`
- `run_with_mcp(..., history=...)`
- 실험 intent는 1차에서는 history 없이 유지해도 됨 (결정론적 조회)

#### Step 4. Intent 분류에 맥락 반영

`classify_intent` 프롬프트에 최근 대화를 짧게 포함.

- 예: “이전 실험이 X였고, top_k=10을 봤다” → 후속 “metric key 알려줘”에서 `experiment_name` 추론
- 출력 JSON 스키마는 기존 유지 (`intent` + args)
- history가 비어 있으면 현재와 동일하게 동작하도록 fallback

#### Step 5. LLM 답변 경로에 messages 배열 사용

`report_qa` / MCP(또는 chat completion) 호출 시:

```text
system → (선택) history turns → user(current message)
```

- `report_text`는 system 또는 별도 context 메시지로 유지
- MCP agent 경로: runner/input에 이전 턴을 이어 붙이거나, 단일 user 메시지에 “이전 대화 요약 + 현재 질문” 형태로 합치는 방식 중 택1
  - **간단:** 요약/히스토리 텍스트를 현재 prompt에 concat
  - **권장:** OpenAI messages 형식 그대로 전달

#### Step 6. 세션 영속화 — PostgreSQL에 채팅 이력 저장·조회·재개

같은 PostgreSQL(`DATABASE_URL`)에 `reports`와 별도 chat 테이블을 두고,
대화 저장 → 목록 조회 → 선택 시 이력 복원 → 이어서 질문할 수 있게 한다.

##### 6.1 목표 (구현 범위)

| 동작 | 구현 |
|------|------|
| 채팅 시작 | `session_id` 없으면 `/chat/ask`에서 세션 자동 생성 → 응답·UI에 보관 |
| 메시지 전송 | 응답 후 user/assistant를 DB에 append |
| 세션 목록 | `GET /chat/sessions` → UI selectbox |
| 이력 재개 | 목록에서 선택 시 `GET .../messages`로 로드 후 같은 세션으로 이어서 채팅 |
| 새 대화 | UI에서 세션/메시지 state 초기화 (`session_id=null`로 다음 ask 시 새 세션) |

##### 6.2 테이블 (구현)

`backend/app/db/models.py` — `reports` 스키마는 변경하지 않음.

**`chat_sessions`**

| 컬럼 | 설명 |
|------|------|
| `id` | UUID PK (`session_id`) |
| `title` | 목록용. 첫 user 메시지 truncate로 자동 설정 |
| `created_at` / `updated_at` | 생성·마지막 메시지 시각 (`updated_at` 내림차순 목록) |

**`chat_messages`** (`ChatMessageRow`, Pydantic `ChatMessage`와 이름 분리)

| 컬럼 | 설명 |
|------|------|
| `id` | UUID PK |
| `session_id` | FK → `chat_sessions.id` ON DELETE CASCADE |
| `role` / `content` | `user` \| `assistant` |
| `intent` | assistant 턴 분류 결과 (nullable) |
| `metadata` | 응답 metadata JSONB |
| `created_at` | 시간순 정렬 |

- 인덱스: `(session_id, created_at)`
- 기동 시 `init_db()` → `create_all` (모델 import 필수)

##### 6.3 Repository

`repository_service.py`에 구현:

- `create_chat_session` / `get_chat_session` / `list_chat_sessions`
- `append_chat_message` (메시지 insert + `updated_at`·title 갱신)
- `list_chat_messages` / `get_recent_chat_history` (LLM용 최근 N턴)

##### 6.4 API · 스키마

| Method | Path | 역할 |
|--------|------|------|
| `POST` | `/chat/sessions` | 빈 세션 생성 |
| `GET` | `/chat/sessions` | 목록 (`updated_at` desc) |
| `GET` | `/chat/sessions/{session_id}/messages` | 메시지 전체 |
| `POST` | `/chat/ask` | `session_id` optional + 응답에 `session_id` + DB append |

`/chat/ask` 순서:

```text
1. session_id 없으면 create / 있으면 검증
2. DB recent N턴 load (없으면 payload history fallback)
3. classify → answer
4. append user / assistant
5. ChatResponse(..., session_id)
```

- DB history가 있으면 **DB 우선**, 없으면 프론트 `history`
- `CHAT_HISTORY_LIMIT` / `settings.chat_history_limit` (기본 6)

##### 6.5 Frontend (`app_2.py`)

- `st.session_state["chat_session_id"]` + `chat_messages`
- ask payload에 `session_id` 포함, 응답 `session_id` 저장
- 챗봇 dialog: **저장된 대화** selectbox 선택 시 자동 로드 (`on_change`)
- **새 대화** 버튼 (`on_click`)으로 state 초기화
- dialog 유지: `st.rerun()` 사용하지 않음 (호출 시 dialog가 닫힘)

##### 6.6 추가 기능 ✅ 구현됨

| 항목 | 구현 |
|------|------|
| 실험 intent 후처리 | intent args → payload → 직전 assistant `metadata` 순으로 `experiment_name` / `metric_key` / `top_k` / `ascending` 채움. 응답 metadata에도 기록해 다음 턴 복원 |
| 세션 삭제 API | `DELETE /chat/sessions/{session_id}` hard delete (messages CASCADE). UI **삭제** 버튼 |
| LLM 세션 title | 첫 턴 완료 후 `generate_session_title`로 짧은 제목 생성·저장 (실패 시 첫 질문 truncate) |

##### 6.7 검증

| 시나리오 | 기대 |
|----------|------|
| 2턴 대화 후 DB | session 1 + messages 4 |
| 목록에서 다른 대화 선택 | dialog 유지한 채 메시지 복원, 후속 질문 가능 |
| `session_id` 없이 ask | 새 세션 + 응답에 `session_id` |
| 새 대화 | 인사말로 리셋, 다음 질문이 새 세션 |

**관련 파일:** `db/models.py`, `db/init_db.py`, `repository_service.py`, `schemas.py`, `api/chat_router.py`, `config.py` (`chat_history_limit`), `frontend/app_2.py`

#### Step 7. 멀티턴 검증 시나리오

| 시나리오 | 기대 |
|----------|------|
| “실험 목록 보여줘” → “첫 번째 실험 metric key 알려줘” | 이전 목록 맥락으로 experiment 추론 |
| 보고서 생성 후 “강점만 다시 요약” → “약점도” | report_qa가 같은 report_text + history로 답변 |
| history 없이 단일 질문 | 기존 싱글턴과 동일 |
| 긴 대화 (N턴 초과) | 최근 N턴만 LLM에 사용, 응답 정상 |
| (Step 6) 목록에서 이전 대화 재개 | UI·DB 이력 일치, 이어서 질문 가능 |

---

### 작업 순서 요약

```
1. ChatRequest에 history 추가
2. app_2.py에서 최근 N턴 history 전송
3. chat_router → chat_service로 history 전달
4. classify_intent에 history 반영
5. report_qa / MCP LLM 호출을 multi-message로 변경
6. session DB 영속화 (구현됨)
   - chat_sessions / chat_messages
   - repository + /chat/sessions·messages + /ask session_id
   - UI: 목록 선택 로드 / 새 대화 / dialog 유지
7. Step 7 시나리오로 수동 테스트
```

### 건드릴 핵심 파일

- `backend/app/schemas.py`
- `backend/app/api/chat_router.py`
- `backend/app/services/chat_service.py`
- `frontend/app_2.py`
- (Step 6) `backend/app/db/models.py`, `init_db.py`, `repository_service.py`, `config.py`

---

## 4. 빠른 참고

- 메인 UI: `frontend/app_2.py`
- 챗 진입: `POST /chat/ask` → `chat_service.py`
- 보고서: `report_service.py` + `repository_service.py`
- 실행: `README.md` / `docker-compose.yml`
