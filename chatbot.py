# rag_main.py 에서 구현된 RAG 기능을 확장하여 챗봇 형태로 출력
from modules.config_loader import load_config
from modules.document_processor import DocumentProcessor
from modules.dense_retrieval import CustomEmbeddings
from modules.sparse_retrieval import BM25Retriever
from modules.qa_chain import QAChain
from modules.file_saver import ExcelSaver, JsonSaver
from modules.faq import FAQ

from langchain.retrievers import EnsembleRetriever

import pandas as pd
import torch
import argparse
import os

def setup_retriever():

    parser = argparse.ArgumentParser(description='Enter user name for the config script')
    parser.add_argument('--config', type=str, required=False, default='config')
    args = parser.parse_args()

    # 설정 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(current_dir, "config", f"{args.config}.yaml"))


        # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 문서 처리
    doc_processor = DocumentProcessor()
    docs, metadata = doc_processor.load_and_process_documents(
        file_path=config['dataset']['document'],
        mode=config['dataset']['document_type'],
        do_chunking=config['preprocessing']['chunking'],
        chunk_size=config['preprocessing']['chunk_size'],
        chunk_overlap=config['preprocessing']['chunk_overlap']
    )

    # --- Retriever 생성
    retrieval_type = config['retrieval']['type']
    if retrieval_type == 'sparse':
        retriever = BM25Retriever(docs)

    elif retrieval_type == 'dense':
        faiss_index_path = config['dataset']['document'].replace(
            '.json',
            f"_{config['dense_model']['model_type']}_" +  # sts 또는 api 구분
            ('chunk.bin' if config['preprocessing']['chunking'] else 'full.bin')
        )
        embedding_service = CustomEmbeddings(
            model_type=config['dense_model']['model_type'],
            model_path=config['dense_model']['model_cache'],
            device=device,
            embedding_config=config,
            faiss_index_path=faiss_index_path
        )
        retriever = embedding_service.create_or_load_vectorstore(
            documents=docs,
            metadata=metadata
        )
        
    elif retrieval_type == 'hybrid':
        retrieversp = BM25Retriever(docs) #tokenizer=config['retrieval']['tokenizer']
        faiss_index_path = config['dataset']['document'].replace(
            '.json',
            f"_{config['dense_model']['model_type']}_" +
            ('chunk.bin' if config['preprocessing']['chunking'] else 'full.bin')
        )
        embedding_service = CustomEmbeddings(
            model_type=config['dense_model']['model_type'],
            model_path=config['dense_model']['model_cache'],
            device=device,
            embedding_config=config,
            faiss_index_path=faiss_index_path,
            #index_type=config['dense_model']['index_type']
        )
        retrieverde = embedding_service.create_or_load_vectorstore(docs)
        retrieverde = retrieverde.as_retriever(search_kwargs={
            "k": config['retrieval']['top_k'],
            "search_type": config['retrieval']['search_type'],
        })
        retriever = EnsembleRetriever(
            retrievers=[retrieversp, retrieverde],
            weights=[0.4, 0.6]
        )

    else:
        raise ValueError("Unsupported retrieval type")

    # QAChain 설정
    qa_chain = QAChain(
        openai_config=config['openai'],
        retriever=retriever,
        search_type=retrieval_type,
        query_translation_type=config['query_translation']['type'],
        top_k=config['retrieval']['top_k'],
        use_reranker=config['retrieval']['reranker'],
        reranker_model=config['retrieval']['reranker_model'],
        reranker_top_k=config['retrieval']['reranker_top_k']
    )

    return qa_chain

def main():
    qa_chain = setup_retriever()
    while True:
        question = input("질문을 입력하세요: ")
        answer = qa_chain.multi_step_qa(question)
        print(answer['answer'])

if __name__ == "__main__":
    main()