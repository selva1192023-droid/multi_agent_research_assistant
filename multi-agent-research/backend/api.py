from fastapi import FastAPI
from pydantic import BaseModel

from backend.orchestrator import build_graph


app = FastAPI(
    title="Multi-Agent AI Research Assistant",
    description="AI Research Assistant with RAG and Multi-Agent Architecture",
    version="1.0"
)


research_app = build_graph()


class ResearchRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Multi-Agent AI Research Assistant API is running"
    }


@app.post("/research")
def research(request: ResearchRequest):

    result = research_app.invoke({
        "question": request.question,
        "research_plan": "",
        "rag_context": "",
        "analysis": "",
        "fact_check": "",
        "final_report": ""
    })

    return {
        "question": request.question,
        "research_plan": result["research_plan"],
        "analysis": result["analysis"],
        "fact_check": result["fact_check"],
        "final_report": result["final_report"]
    }