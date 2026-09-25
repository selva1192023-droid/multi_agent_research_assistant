from backend.rag.loader import load_documents
from backend.rag.chunker import split_text
from backend.rag.embeddings import EmbeddingModel
from backend.rag.vectorstore import VectorStore

from langchain_ollama import ChatOllama


DOCUMENT_PATH = "data/documents"

# Minimum similarity score required to accept retrieved context
SIMILARITY_THRESHOLD = 0.55


def rag_agent(question):

    documents = load_documents(DOCUMENT_PATH)

    chunks = []

    for document in documents:

        document_chunks = split_text(
            document["text"]
        )

        for chunk in document_chunks:

            chunks.append({
                "text": chunk,
                "source": document["source"]
            })

    embedding_model = EmbeddingModel()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(texts)

    dimension = embeddings.shape[1]

    vector_store = VectorStore(dimension)

    vector_store.add(
        embeddings,
        chunks
    )

    query_embedding = embedding_model.encode(
        [question]
    )[0]

    results = vector_store.search(
        query_embedding,
        top_k=3
    )

    # Keep only sufficiently relevant results
    relevant_results = [
        result
        for result in results
        if result["score"] >= SIMILARITY_THRESHOLD
    ]

    # No sufficiently relevant information found
    if not relevant_results:

        return {
            "answer": (
                "The information is not available "
                "in the provided documents."
            ),
            "context": ""
        }

    context = "\n\n".join(
        result["text"]
        for result in relevant_results
    )

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    prompt = f"""
You are a RAG Agent.

Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Rules:
- Use only the provided context.
- Do not invent information.
- If the answer is not present in the context,
  say "The information is not available in the provided documents."
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "context": context
    }


if __name__ == "__main__":

    question = input("Ask a question: ")

    result = rag_agent(question)

    print("\nRetrieved Context:\n")
    print(result["context"])

    print("\nFinal Answer:\n")
    print(result["answer"])