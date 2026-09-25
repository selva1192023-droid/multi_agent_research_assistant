from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


def research_agent(question):

    prompt = f"""
You are a Research Agent in a Multi-Agent AI Research Assistant.

Your job is to create a structured research plan for the user's question.

Rules:
- Do not invent facts.
- Do not assume the meaning of abbreviations.
- Do not expand abbreviations unless the meaning is given in the question.
- Do not answer the question.
- Stay focused on the exact research question.
- The RAG Agent will provide evidence later.

Identify:

1. Main Topic
2. Key Concepts
3. Important Aspects to Investigate
4. Questions That Should Be Answered

Research Question:
{question}

Return only the structured research plan.
"""

    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":

    question = input("Enter research question: ")

    result = research_agent(question)

    print("\nResearch Agent Output:\n")
    print(result)