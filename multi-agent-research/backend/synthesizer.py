from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


def synthesizer(
    research_plan,
    context,
    analysis,
    fact_check
):

    # No evidence available
    if not context.strip():

        return (
            "The information is not available "
            "in the provided documents."
        )

    prompt = f"""
You are the Synthesizer in a Multi-Agent AI Research Assistant.

Create a final research report using ONLY the information
provided by the other agents.

Research Plan:
{research_plan}

Retrieved Evidence:
{context}

Analysis:
{analysis}

Fact Check:
{fact_check}

Create the report with these sections:

1. Research Topic
2. Key Findings
3. Evidence
4. Analysis
5. Fact-Checked Findings
6. Conclusion

Important rules:

- Use only the retrieved evidence.
- Do not introduce outside knowledge.
- Do not invent facts.
- Do not treat the research plan as evidence.
- Do not include unsupported claims as established facts.
- If evidence is insufficient, clearly state that the
  information is not available in the provided documents.
"""

    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":

    research_plan = input(
        "Enter research plan: "
    )

    context = input(
        "\nEnter evidence: "
    )

    analysis = input(
        "\nEnter analysis: "
    )

    fact_check = input(
        "\nEnter fact check: "
    )

    result = synthesizer(
        research_plan,
        context,
        analysis,
        fact_check
    )

    print("\nFinal Research Report:\n")
    print(result)