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

    intent, intent_args = classify_intent(payload.message)

    if intent_args is not None:
        experiment_name = intent_args.get("experiment_name") if intent_args.get("experiment_name") is not None else payload.experiment_name
        metric_key = intent_args.get("metric_key") if intent_args.get("metric_key") is not None else payload.metric_key
        top_k = intent_args.get("top_k") or None
        ascending = intent_args.get("ascending") or False


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
            message=payload.message,
            experiment_name=experiment_name,
            metric_key=metric_key,
            top_k=top_k,
            ascending=ascending,
        )
    elif intent == "report_qa":
        answer, intent, metadata = answer_report_question(
            intent=intent,
            message=payload.message,
            report_text=payload.report_text,
            experiment_name=experiment_name,
            metric_key=metric_key,
        )
    else:
        answer, intent, metadata = await run_with_mcp(
            message=payload.message
        )

    return ChatResponse(answer=answer, intent=intent, metadata=metadata)
