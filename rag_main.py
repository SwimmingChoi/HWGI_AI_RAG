from modules.config_loader import load_config
from modules.document_processor import DocumentProcessor
from modules.dense_retrieval import CustomEmbeddings
from modules.sparse_retrieval import BM25Retriever
from modules.qa_chain import QAChain
from modules.file_saver import ExcelSaver, JsonSaver

from langchain.retrievers import EnsembleRetriever

import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from time import time
import json

from utils.logger import setup_logger
import torch
import logging
import os

import argparse

def main():
    
    total_start_time = time()
    
    parser = argparse.ArgumentParser(description='Enter user name for the config script')
    parser.add_argument('--config', type=str, required=False, default='config')
    args = parser.parse_args()
    
    # 설정 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(current_dir, "config", f"{args.config}.yaml"))
    logger = setup_logger() # DEFAULT_LOG_PATH=config['log']['log_path']

    df = pd.read_excel(os.path.join(current_dir, "documents", config['dataset']['qna']))
    reference_questions = df[['number', 'question']].dropna().set_index('number')['question'].to_dict()
    reference_answers = df.set_index('number')['answers'].dropna().to_dict()
    reference_pages = df.set_index('number')['answer_pages'].dropna().to_dict()
    # reference_context = df.set_index('number')['ground truth page content'].dropna().to_dict()
    # reference_info = df.set_index('number')['document'].dropna().to_dict()
    logger.info(f"'answer_pages' 컬럼에서 {len(reference_pages)}개의 정답페이지를 가져왔습니다.")
    
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
        retriever = embedding_service.create_or_load_vectorstore(docs)
        
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
            "search_type": config['retrieval']['search_type']
        })
        retriever = EnsembleRetriever(
            retrievers=[retrieversp, retrieverde],
            weights=[0.5, 0.5]
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

    today_date = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d')
    qna_name = os.path.splitext(config['dataset']['qna'])[0]
    # document_name = os.path.splitext(config['dataset']['document'])[0]
    chunking = f"chunking_{config['preprocessing']['chunk_size']}_{config['preprocessing']['chunk_overlap']}" if config['preprocessing']['chunking'] else "full_page"
    reranker = f"rerank_{config['retrieval']['reranker_top_k']}" if config['retrieval']['reranker'] else "no_reranker"
    query_translation = f"{config['query_translation']['type']}"

    # 출력 파일 경로 설정
    output_file = os.path.join(
        config['output']['save_path'], 
        f"result_{today_date}_q_{qna_name}.json"
    )


    save_results = []

    # 기존 결과 로드
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            save_results = {int(item['number']): item for item in json.load(f)}
        logger.info("기존 결과 파일을 불러왔습니다.")
    else:
        save_results = {}
        logger.info("새로운 결과 파일을 생성합니다.")


    # 새로운 키 정의
    retrieval_model = embedding_service.model_type if retrieval_type == 'dense' else (
        'bm25' if retrieval_type == 'sparse' else f'{embedding_service.model_type}+bm25'
        )
    new_key = (
        f"{retrieval_type}_"
        f"{retrieval_model}_"
        f"{query_translation}_"
        f"{chunking}_"
        f"{reranker}"
    )
    
    # call json_saver
    json_saver = JsonSaver(output_file)
    
    # 결과 처리
    for num, question in reference_questions.items():
        start_time = time()
        if num in save_results:
            # 기존 데이터에서 new_key 확인
            if new_key in save_results[num]['llm response']:
                logger.info(f"중복된 키 '{new_key}'가 질문 번호 {num}에 이미 존재합니다. 추가하지 않습니다.")
                continue
            else:
                result = qa_chain.multi_step_qa(question, reset_memory=True)
                end_time = time()
                execution_time = end_time - start_time
                logger.info(f"질문 번호 {num}의 소요 시간: {execution_time}초")
                save_results[num]['llm response'][new_key] = json_saver.save_response(
                    result, execution_time
                    )
                logger.info(f"'{new_key}'가 질문 번호 {num}에 추가되었습니다.")
        else:
            # 새로운 질문 추가
            result = qa_chain.multi_step_qa(question, reset_memory=True)
            end_time = time()
            execution_time = end_time - start_time
            logger.info(f"질문 번호 {num}의 소요 시간: {execution_time}초")
            if 'context_labeling' in config['dataset']['qna']:
                save_results[num] = {
                    "number": num,
                    "question": question,
                    "ground truth answer": reference_answers.get(num, ""),
                    "ground truth answer pages": reference_pages.get(num, []),
                    "llm response": {
                    new_key: json_saver.save_response(
                        result, execution_time
                        )
                    }
                }
            else:
                save_results[num] = {
                    "number": num,
                    "question": question,
                    "ground truth answer": reference_answers.get(num, ""),
                    "ground truth answer pages": reference_pages.get(num, []),
                    "llm response": {
                    new_key: json_saver.save_response(
                        result, execution_time
                        )
                    }
                }
            logger.info(f"새 질문 번호 {num}의 결과를 저장했습니다.")
    
    total_end_time = time()
    total_execution_time = total_end_time - total_start_time
    logger.info(f"총 소요 시간: {total_execution_time}초")

    # 결과 저장
    json_saver.save_to_json(save_results)
    
    logger.info(f"\n모든 질문에 대한 결과가 json 파일로 저장되었습니다: {output_file}")

    # Excel 저장
    excel_saver = ExcelSaver(output_file)
    excel_saver.save_to_excel()


if __name__ == "__main__":
    main()
