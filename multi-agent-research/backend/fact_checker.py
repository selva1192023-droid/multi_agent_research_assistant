from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


def fact_checker(analysis, context):

    # No evidence available
    if not context.strip():

        return (
            "No claims available for fact checking "
            "because no relevant evidence was retrieved."
        )

    # No meaningful analysis available
    if not analysis.strip():

        return (
            "No claims available for fact checking "
            "because no analysis was produced."
        )

    prompt = f"""
You are a Fact Checker in a Multi-Agent AI Research Assistant.

Your job is to check whether the claims in the analysis
are supported by the provided evidence.

Evidence:
{context}

Analysis:
{analysis}

For each important claim:

1. Identify the claim.
2. Determine whether it is supported by the evidence.
3. Mark it as SUPPORTED or UNSUPPORTED.
4. Briefly explain why.

Rules:

- Use only the provided evidence.
- Do not add new facts.
- Do not use outside knowledge.
- Do not treat the research plan as evidence.
- If a claim is not supported by the evidence, mark it UNSUPPORTED.
"""

    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":

    context = input("Enter evidence: ")

    analysis = input("\nEnter analysis: ")

    result = fact_checker(
        analysis,
        context
    )

    print("\nFact Checker Output:\n")
    print(result)