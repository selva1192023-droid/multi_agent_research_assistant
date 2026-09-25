from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from backend.research_agent import research_agent
from backend.rag_agent import rag_agent
from backend.analysis_agent import analysis_agent
from backend.fact_checker import fact_checker
from backend.synthesizer import synthesizer


class ResearchState(TypedDict):
    question: str
    research_plan: str
    rag_context: str
    analysis: str
    fact_check: str
    final_report: str


def orchestrator_node(state: ResearchState):

    print("\n[Orchestrator] Processing research request...")

    return state


def research_node(state: ResearchState):

    print("\n[Research Agent] Creating research plan...")

    result = research_agent(
        state["question"]
    )

    return {
        "research_plan": result
    }


def rag_node(state: ResearchState):

    print("\n[RAG Agent] Retrieving relevant information...")

    result = rag_agent(
        state["question"]
    )

    return {
        "rag_context": result["context"]
    }


def analysis_node(state: ResearchState):

    print("\n[Analysis Agent] Analyzing research information...")

    information = f"""
Research Plan:

{state["research_plan"]}

Retrieved Information:

{state["rag_context"]}
"""

    result = analysis_agent(
        information
    )

    return {
        "analysis": result
    }


def fact_check_node(state: ResearchState):

    print("\n[Fact Checker] Checking claims...")

    result = fact_checker(
        state["analysis"],
        state["rag_context"]
    )

    return {
        "fact_check": result
    }


def synthesizer_node(state: ResearchState):

    print("\n[Synthesizer] Creating final research report...")

    result = synthesizer(
        state["research_plan"],
        state["rag_context"],
        state["analysis"],
        state["fact_check"]
    )

    return {
        "final_report": result
    }


def build_graph():

    graph = StateGraph(ResearchState)

    graph.add_node(
        "orchestrator",
        orchestrator_node
    )

    graph.add_node(
        "research_agent",
        research_node
    )

    graph.add_node(
        "rag_agent",
        rag_node
    )

    graph.add_node(
        "analysis_agent",
        analysis_node
    )

    graph.add_node(
        "fact_checker",
        fact_check_node
    )

    graph.add_node(
        "synthesizer",
        synthesizer_node
    )

    graph.add_edge(
        START,
        "orchestrator"
    )

    graph.add_edge(
        "orchestrator",
        "research_agent"
    )

    graph.add_edge(
        "research_agent",
        "rag_agent"
    )

    graph.add_edge(
        "rag_agent",
        "analysis_agent"
    )

    graph.add_edge(
        "analysis_agent",
        "fact_checker"
    )

    graph.add_edge(
        "fact_checker",
        "synthesizer"
    )

    graph.add_edge(
        "synthesizer",
        END
    )

    return graph.compile()


if __name__ == "__main__":

    app = build_graph()

    question = input(
        "Enter research question: "
    )

    result = app.invoke({
        "question": question,
        "research_plan": "",
        "rag_context": "",
        "analysis": "",
        "fact_check": "",
        "final_report": ""
    })

    print("\n" + "=" * 60)
    print("FINAL RESEARCH REPORT")
    print("=" * 60)

    print(result["final_report"])