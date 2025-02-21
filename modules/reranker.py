from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.schema import BaseRetriever


class Reranker:
    def __init__(self, base_retriever: BaseRetriever, reranker_model: str = "Dongjin-kr/ko-reranker", reranker_top_k: int = 20):
        """
        Reranker 초기화
        :param base_retriever: BaseRetriever를 구현한 객체
        :param reranker_model: Reranker 모델 이름
        :param reranker_top_k: 상위 n개의 문서만 선택
        """
        if not isinstance(base_retriever, BaseRetriever):
            raise TypeError("base_retriever는 BaseRetriever의 인스턴스여야 합니다.")
        
        self.model = HuggingFaceCrossEncoder(model_name=reranker_model)
        self.compressor = CrossEncoderReranker(model=self.model, top_n=reranker_top_k)
        
        # retriever를 명확히 BaseRetriever로 변환
        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever
        )
    
    def get_retriever(self):
        """
        Reranker가 적용된 Retriever 반환
        """
        return self.retriever
