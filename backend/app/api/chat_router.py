from fastapi import APIRouter
from backend.app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/ask", response_model=ChatResponse)
async def chat_with_experiment(payload: ChatRequest) -> ChatResponse:
    # 다른 라우트 기동에 영향 없도록 chat 의존성은 실제 호출 시 로드한다.
    from backend.app.services.chat_service import (
        classify_intent,
        answer_report_question,
        answer_experiment_question,
        run_with_mcp,
    )

    intent = classify_intent(payload.message)

    # intent = "mlflow_mcp"

    experiment_intents = [
        "list_experiments",
        "list_recent_experiments",
        "get_experiment",
        "get_runs",
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
    elif intent == "report_qa":
        answer, intent, metadata = answer_report_question(
            intent=intent,
            message=payload.message,
            report_text=payload.report_text,
            experiment_name=payload.experiment_name,
            metric_key=payload.metric_key,
        )
    else:
        answer, intent, metadata = await run_with_mcp(
            message=payload.message
        )

    return ChatResponse(answer=answer, intent=intent, metadata=metadata)
