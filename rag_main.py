from modules.config_loader import load_config
from modules.document_processor import DocumentProcessor
from modules.dense_retrieval import CustomEmbeddings
from modules.sparse_retrieval import BM25Retriever
from modules.qa_chain import QAChain
import pandas as pd
# from eval import evaluate_results
from datetime import datetime
from zoneinfo import ZoneInfo
import json

from utils.logger import setup_logger
import torch
import logging
import os

import argparse

def main():
    
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
    reference_context = df.set_index('number')['ground truth page content'].dropna().to_dict()
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
    else:
        raise ValueError("Unsupported retrieval type")
    
    # QAChain 설정
    qa_chain = QAChain(
        openai_config=config['openai'],
        retriever=retriever,
        search_type=retrieval_type,
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
    new_key = (
        f"{retrieval_type}_"
        f"{embedding_service.model_type if retrieval_type == 'dense' else 'bm25'}_"
        f"{chunking}_"
        f"{reranker}"
    )

    # 결과 처리
    for num, question in reference_questions.items():
        if num in save_results:
            # 기존 데이터에서 new_key 확인
            if new_key in save_results[num]['llm response']:
                logger.info(f"중복된 키 '{new_key}'가 질문 번호 {num}에 이미 존재합니다. 추가하지 않습니다.")
                continue
            else:
                # 새로운 키 추가
                result = qa_chain.ask_question(question, reset_memory=True)
                save_results[num]['llm response'][new_key] = {
                    "answer": result['answer'],
                    "pages": result['context_pages'],
                    "page contents": result['context_pages_content']
                }
                logger.info(f"'{new_key}'가 질문 번호 {num}에 추가되었습니다.")
        else:
            # 새로운 질문 추가
            result = qa_chain.ask_question(question, reset_memory=True)
            
            if 'context_labeling' in config['dataset']['qna']:
                save_results[num] = {
                    "number": num,
                    "question": question,
                    "ground truth answer": reference_answers.get(num, ""),
                    "ground truth answer pages": reference_pages.get(num, []),
                    'ground truth answer context': reference_context.get(num, ""),
                    "llm response": {
                    new_key: {
                        "answer": result['answer'],
                        "pages": result['context_pages'],
                        "page contents": result['context_pages_content']
                    }
                },
                
            }
            else:
                save_results[num] = {
                    "number": num,
                    "question": question,
                    "ground truth answer": reference_answers.get(num, ""),
                    "ground truth answer pages": reference_pages.get(num, []),
                    "llm response": {
                    new_key: {
                        "answer": result['answer'],
                        "pages": result['context_pages'],
                        "page contents": result['context_pages_content']
                    }
                },
                # "실제QA": result['실제QA'],
                # "document": result['document']
            }
            logger.info(f"새 질문 번호 {num}의 결과를 저장했습니다.")

    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(list(save_results.values()), json_file, ensure_ascii=False, indent=4)
    logger.info(f"\n모든 질문에 대한 결과가 json 파일로 저장되었습니다: {output_file}")

    # JSON 파일 읽기
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 필요한 데이터 추출
    excel_data = []
    prev_number = None

    # methods 자동 추출 (첫 번째 항목의 llm response에서 키들을 가져옴)
    methods = list(data[0]['llm response'].keys())

    for item in data:
        # 각 method별로 행 생성
        for i, method in enumerate(methods):
            if 'context_labeling' in config['dataset']['qna']:
                row = {
                    'number': item['number'] if item['number'] != prev_number else '',
                    'question': item['question'] if item['number'] != prev_number else '',
                    'ground truth answer': item['ground truth answer'] if item['number'] != prev_number else '',
                    'ground truth answer pages': item['ground truth answer pages'].strip('[]') if item['number'] != prev_number else '',
                    'ground truth answer context': item['ground truth answer context'] if item['number'] != prev_number else '',
                    'method': method,
                    'answer': item['llm response'][method]['answer'],
                    'rag answer pages': str(item['llm response'][method]['pages']).strip('[]'),
                    'rag answer page contents': item['llm response'][method]['page contents'],
                }
            else:
                row = {
                    'number': item['number'] if item['number'] != prev_number else '',
                    'question': item['question'] if item['number'] != prev_number else '',
                    'ground truth answer': item['ground truth answer'] if item['number'] != prev_number else '',
                'ground truth answer pages': item['ground truth answer pages'].strip('[]') if item['number'] != prev_number else '',
                'method': method,
                'answer': item['llm response'][method]['answer'],
                'rag answer pages': str(item['llm response'][method]['pages']).strip('[]'),
                'rag answer page contents': item['llm response'][method]['page contents'],
                # '실제QA': item['실제QA'] if item['number'] != prev_number else '',
                # 'document' : item['document'] if item['number'] != prev_number else ''
            }
            excel_data.append(row)
            prev_number = item['number']

    # DataFrame 생성
    df = pd.DataFrame(excel_data)

    # 칼럼 순서 지정
    columns = ['number', 'question', 'ground truth answer', 'ground truth answer context', 'ground truth answer pages', 'method', 'answer', 'rag answer pages', 'rag answer page contents']
    df = df[columns]

    # Excel 파일로 저장 (encoding 파라미터 제거)
    df.to_excel(output_file.replace('.json', '.xlsx'), index=False)


if __name__ == "__main__":
    main()
