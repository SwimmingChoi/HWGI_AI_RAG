from langchain.schema import Document, BaseRetriever
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from pydantic import Field

class BM25Retriever(BaseRetriever):
    docs: List[Document] = Field(default_factory=list)
    tokenized_corpus: List[List[str]] = Field(default_factory=list)
    bm25: BM25Okapi = None

    def __init__(self, docs: List[Document]):
        super().__init__(docs=docs)
        object.__setattr__(self, 'tokenized_corpus', [doc.page_content.split() for doc in docs])
        object.__setattr__(self, 'bm25', BM25Okapi(self.tokenized_corpus))

    def _get_relevant_documents(self, query: str, **kwargs):
        """
        쿼리에 대해 관련 문서를 반환
        """
        top_k = kwargs.get("k", 3)
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.docs[i] for i in top_k_indices]

    def as_retriever(self, search_kwargs: Dict[str, Any] = None):
        search_kwargs = search_kwargs or {}

        class BM25Wrapper(BaseRetriever):
            parent: BM25Retriever = Field()
            tags: List[str] = Field(default_factory=list)
            metadata: Dict[str, Any] = Field(default_factory=dict)

            def __init__(self, parent: BM25Retriever):
                object.__setattr__(self, "parent", parent)
                object.__setattr__(self, "tags", [])
                object.__setattr__(self, "metadata", {})

            def _get_relevant_documents(self, query: str):
                return self.parent._get_relevant_documents(query, **search_kwargs)

        return BM25Wrapper(self)