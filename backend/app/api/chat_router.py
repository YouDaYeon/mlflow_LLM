from fastapi import APIRouter
from backend.app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/ask", response_model=ChatResponse)
def chat_with_experiment(payload: ChatRequest) -> ChatResponse:
    # 다른 라우트 기동에 영향 없도록 chat 의존성은 실제 호출 시 로드한다.
    from backend.app.services.chat_service import classify_intent, answer_report_question, answer_experiment_question

    intent = classify_intent(payload.message)

    experiment_intents = [
        "list_experiments",
        "list_recent_experiments",
        "list_metric_keys",
        "filter_runs_by_metric",
        "search_runs",
    ]
    
    if intent in experiment_intents:
        answer, intent, metadata = answer_experiment_question(
            intent=intent,
            experiment_name=payload.experiment_name,
            message=payload.message,
            metric_key=payload.metric_key,
        )
    else:
        answer, intent, metadata = answer_report_question(
            message=payload.message,
            report_text=payload.report_text,
            experiment_name=payload.experiment_name,
            metric_key=payload.metric_key,
        )   

    return ChatResponse(answer=answer, intent=intent, metadata=metadata)
