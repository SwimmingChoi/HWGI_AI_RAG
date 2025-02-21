from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Union, Optional, Dict
from tqdm import tqdm
import torch
import os
import logging
class CustomEmbeddings(Embeddings):
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        embedding_config: dict,
        model_type: str = "sts",
        faiss_index_path: str = "faiss_embedding_large.bin",
        batch_size: int = 4
    ):
        """
        :param model_path: SentenceTransformer 모델 경로
        :param device: torch device
        :param embedding_config: embedding api 설정
        :param model_type: 'sts' 또는 'api'
        :param faiss_index_path: FAISS 인덱스 저장 경로
        """
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_type = model_type.lower()
        self.faiss_index_path = os.path.join(current_dir, "documents", faiss_index_path)
        self.batch_size = batch_size
        if self.model_type not in ["sts", "api"]:
            raise ValueError("model_type must be either 'sts' or 'api'")
        if self.model_type == "sts":
            self.model = SentenceTransformer(
                model_name_or_path = embedding_config['dense_model']['model_name'],
                cache_folder=os.path.join(current_dir, model_path)
            ).to(device)
        else:
            self.model = OpenAIEmbeddings(
                api_key=embedding_config['openai']['api_key'],
                model=embedding_config['openai']['embedding_model'],
            )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.model_type == "sts":
            embeddings = []
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding documents"):
                batch_texts = texts[i:i+self.batch_size]
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        show_progress_bar=False,
                        convert_to_tensor=True
                        )
                    embeddings.extend(batch_embeddings.cpu().numpy())
                torch.cuda.empty_cache()
            return embeddings
        else:
            return self.model.embed_documents(texts)
    def embed_query(self, text: str) -> List[float]:
        if self.model_type == "sts":
            with torch.no_grad():
                embedding = self.model.encode(text, convert_to_tensor=True)
                return embedding.cpu().numpy()
        else:
            return self.model.embed_query(text)
    def create_or_load_vectorstore(self, documents: Optional[List] = None, metadata: Optional[Dict] = None) -> FAISS:
        """
        FAISS 벡터 DB를 생성하거나 로드합니다.
        Args:
            documents: 문서 리스트 (새로 생성할 경우에만 필요)
            metadata: 문서 메타데이터 딕셔너리
        Returns:
            FAISS 벡터스토어
        """
        try:
            if os.path.exists(self.faiss_index_path):
                logging.info("로컬에 저장된 FAISS 벡터 DB를 불러옵니다...")
                vectorstore = FAISS.load_local(
                    self.faiss_index_path,
                    self,
                    allow_dangerous_deserialization=True
                )
                logging.info(f"{self.faiss_index_path} FAISS 벡터 DB 로드 완료.")
            else:
                if documents is None:
                    raise ValueError("문서가 제공되지 않았습니다.")
                logging.info("FAISS 벡터 DB를 새로 생성합니다...")
                batch_size = self.batch_size
                total_docs = len(documents)
                vectorstore = None
                for i in tqdm(range(0, total_docs, batch_size), desc="Creating vector DB"):
                    batch_docs = documents[i:i + batch_size]
                    # 각 문서에 메타데이터 추가
                    if metadata:
                        for doc in batch_docs:
                            doc_idx = documents.index(doc)
                            if doc_idx in metadata:
                                doc.metadata.update(metadata[doc_idx])
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(
                            documents=batch_docs,
                            embedding=self
                        )
                    else:
                        temp_vectorstore = FAISS.from_documents(
                            documents=batch_docs,
                            embedding=self
                        )
                        vectorstore.merge_from(temp_vectorstore)
                    torch.cuda.empty_cache()
                vectorstore.save_local(self.faiss_index_path)
                logging.info(f"FAISS 벡터 DB 저장 완료: {self.faiss_index_path}")
            return vectorstore
        except Exception as e:
            logging.error(f"벡터 DB 처리 중 오류 발생: {str(e)}")
            raise
    def similarity_search_with_score(
        self,
        vectorstore,
        query: Union[str, List[str]],
        k: int = 4,
        filter_metadata: Optional[Dict] = None
    ) -> List[List[tuple]]:
        """
        메타데이터 필터링을 포함한 유사도 검색
        Args:
            vectorstore: FAISS 벡터스토어
            query: 단일 문자열 또는 쿼리 문자열 리스트
            k: 각 쿼리당 반환할 결과 수
            filter_metadata: 메타데이터 필터 조건
        """
        try:
            queries = [query] if isinstance(query, str) else query
            results = []
            for i in tqdm(range(0, len(queries), self.batch_size), desc="Similarity search"):
                batch_queries = queries[i:i + self.batch_size]
                batch_results = []
                for q in batch_queries:
                    docs_and_scores = vectorstore.similarity_search_with_score(
                        q,
                        k=k,
                        filter=filter_metadata
                    )
                    batch_results.append(docs_and_scores)
                results.extend(batch_results)
                torch.cuda.empty_cache()
            return results[0] if isinstance(query, str) else results
        except Exception as e:
            logging.error(f"유사도 검색 중 오류 발생: {str(e)}")
            raise