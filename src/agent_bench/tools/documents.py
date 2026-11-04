
import re

from rank_bm25 import BM25Okapi
from langchain_core.tools import tool


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "in", "on", "of", "for", "to", "and", "or",
    "what", "which", "who", "when", "where",
    "did", "does", "do", "has", "have", "had",
    "by", "with", "from", "that", "this"
}


def tokenize(text):
    words = re.findall(
        r"[a-z0-9]+",
        str(text).lower()
    )

    return [
        word for word in words
        if word not in STOPWORDS
    ]


class DocumentEnvironment:

    def __init__(self, documents):

        self.documents = documents or []

        self.corpus = [
            tokenize(
                doc["title"] + " " + doc["content"]
            )
            for doc in self.documents
        ]

        self.bm25 = (
            BM25Okapi([tokens or ["empty"] for tokens in self.corpus])
            if self.corpus else None
        )

    def search(self, query, top_k=3):

        if self.bm25 is None:
            return "No documents available for retrieval."

        query_tokens = tokenize(query)

        if not query_tokens:
            return "No relevant evidence found."

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices:

            if scores[index] <= 0:
                continue

            doc = self.documents[index]

            results.append(
                f"Title: {doc['title']}\n"
                f"Content: {doc['content']}"
            )

            if len(results) >= top_k:
                break

        if not results:
            return "No relevant evidence found."

        return "\n\n".join(results)

    def read(self, title):

        for doc in self.documents:

            if doc["title"].lower() == title.lower():

                return (
                    f"Title: {doc['title']}\n"
                    f"Content: {doc['content']}"
                )

        return "Document not found."


def create_tools(environment):

    @tool
    def search_documents(query: str) -> str:
        """Search the document collection for relevant evidence."""
        return environment.search(query)

    @tool
    def read_document(title: str) -> str:
        """Read a document by its exact title."""
        return environment.read(title)

    return [search_documents, read_document]
