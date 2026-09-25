import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):

        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    def add(self, embeddings, documents):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(embeddings)

        self.documents.extend(documents)

    def search(self, query_embedding, top_k=3):

        query_embedding = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index != -1:

                results.append({
                    "score": float(score),
                    "text": self.documents[index]["text"],
                    "source": self.documents[index]["source"]
                })

        return results