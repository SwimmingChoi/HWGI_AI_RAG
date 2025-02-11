from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Union, Optional
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
        faiss_index_path: str = "faiss_embedding_large.bin"
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
        print(self.model_type)
        self.faiss_index_path = os.path.join(current_dir, "documents", faiss_index_path)
        
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
            return self.model.encode(texts)
        else:
            return self.model.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        if self.model_type == "sts":
            return self.model.encode(text)
        else:
            return self.model.embed_query(text)
            
    def create_or_load_vectorstore(self, documents: Optional[List] = None) -> FAISS:
        """
        FAISS 벡터 DB를 생성하거나 로드합니다.
        
        :param documents: 문서 리스트 (새로 생성할 경우에만 필요)
        :return: FAISS 벡터스토어
        """
        try:
            if os.path.exists(self.faiss_index_path):
                # 로컬 DB가 존재하면 로드
                logging.info("로컬에 저장된 FAISS 벡터 DB를 불러옵니다...")
                vectorstore = FAISS.load_local(
                    self.faiss_index_path,
                    self,
                    allow_dangerous_deserialization=True
                    ) # allow_dangerous_deserialization=True)
                logging.info(f"{self.faiss_index_path} FAISS 벡터 DB 로드 완료.")
            else:
                # 로컬 DB가 없으면 새로 생성
                if documents is None:
                    raise ValueError("문서가 제공되지 않았습니다. 새로운 벡터 DB를 생성하려면 documents가 필요합니다.")
                
                logging.info("FAISS 벡터 DB를 새로 생성합니다...")
                vectorstore = FAISS.from_documents(
                    documents=documents,
                    embedding=self
                )
                # 생성된 벡터 DB를 로컬에 저장
                vectorstore.save_local(self.faiss_index_path)
                logging.info(f"FAISS 벡터 DB 저장 완료: {self.faiss_index_path}")
                
            return vectorstore
            
        except Exception as e:
            logging.error(f"벡터 DB 처리 중 오류 발생: {str(e)}")
            raise