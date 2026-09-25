from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


def analysis_agent(information):

    # Check whether useful evidence was retrieved
    if not information.strip():
        return (
            "No relevant evidence is available "
            "in the provided documents."
        )

    if "Retrieved Information:\n\n" in information:

        retrieved_information = information.split(
            "Retrieved Information:\n\n",
            1
        )[1].strip()

        if not retrieved_information:
            return (
                "No relevant evidence is available "
                "in the provided documents."
            )

    prompt = f"""
You are an Analysis Agent in a Multi-Agent AI Research Assistant.

Analyze ONLY the information provided below.

{information}

Rules:
- Use only the provided research plan and retrieved information.
- Do not introduce outside knowledge.
- Do not invent facts.
- Do not assume that a research plan is evidence.
- If the retrieved information is empty or does not contain
  evidence relevant to the question, clearly state:
  "No relevant evidence is available in the provided documents."
- Do not make conclusions that are not supported by the retrieved information.

Provide:

1. Key Findings
2. Important Patterns
3. Relationships between Findings
4. Advantages and Limitations
5. Important Observations
"""

    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":

    information = input(
        "Enter research information: "
    )

    result = analysis_agent(
        information
    )

    print("\nAnalysis Output:\n")
    print(result)