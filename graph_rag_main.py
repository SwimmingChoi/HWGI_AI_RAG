from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from modules.config_loader import load_config

import csv
import pandas as pd
# from eval import evaluate_results
from datetime import datetime
from zoneinfo import ZoneInfo
import json

from utils.logger import setup_logger
import torch
import logging

import argparse


import os
import csv


def main():
    
    parser = argparse.ArgumentParser(description='Enter user name for the config script')
    parser.add_argument('--config', type=str, required=False, default='config')
    args = parser.parse_args()
    
    # 설정 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(current_dir, "config", f"{args.config}.yaml"))
    logger = setup_logger() # DEFAULT_LOG_PATH=config['log']['log_path']


    # QnA dataset load
    df = pd.read_excel(os.path.join(current_dir, "documents", config['dataset']['qna']))
    reference_questions = df[['number', 'question']].dropna().set_index('number')['question'].to_dict()
    reference_answers = df.set_index('number')['answers'].dropna().to_dict()
    reference_pages = df.set_index('number')['answer_pages'].dropna().to_dict()
    logger.info(f"'answer_pages' 컬럼에서 {len(reference_pages)}개의 정답페이지를 가져왔습니다.")
    
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    graph = Neo4jGraph(
        url=config['graph_rag']['NEO4J_URI'],
        username=config['graph_rag']['NEO4J_USERNAME'],
        password=config['graph_rag']['NEO4J_PASSWORD'],
    )


    data_loader = f"LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/gw16/for_data/main/{config['dataset']['graph_document']}' AS row"
    # Cypher 쿼리: CSV 파일에서 데이터 로드
    q_load_data = data_loader + """

    MERGE (p:Product_Name {name: row.Product_Name})
    MERGE (c:Clause_Name {name: row.Clause_Name})
    MERGE (ch:Chapter {name: row.Chapter})
    MERGE (a:Article {name: row.Article})
    MERGE (s:Subsection {name: row.Subsection})
    MERGE (fc:Content {text: row.Full_Content})
    MERGE (pg:Page {number: row.Page})

    MERGE (p)-[:포함하다]->(c)
    MERGE (c)-[:포함하다]->(ch)
    MERGE (ch)-[:포함하다]->(a)
    MERGE (a)-[:포함하다]->(s)
    MERGE (s)-[:설명하다]->(fc)
    MERGE (fc)-[:구성하다]->(pg)
    """

    graph.query(q_load_data)
    graph.refresh_schema()
    logger.info("데이터가 성공적으로 Neo4j로 로드되었습니다!...")

    llm = ChatOpenAI(
        model_name="gpt-4o",
        api_key=config['openai']['api_key'],
        max_tokens=1024,
        temperature=0.5
    )

    chain = GraphCypherQAChain.from_llm(
        llm, graph=graph, verbose=True, allow_dangerous_requests=True
    )

    today_date = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d')
    qna_name = os.path.splitext(config['dataset']['qna'])[0]
    # document_name = os.path.splitext(config['dataset']['graph_document'])[0]
    save_results = []

    # 출력 파일 경로 설정
    output_file = os.path.join(
        config['output']['save_path'], 
        f"result_{today_date}_q_{qna_name}.json"
    )

    # 기존 결과 로드
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            save_results = {int(item['number']): item for item in json.load(f)}
        logger.info("기존 결과 파일을 불러왔습니다.")
    else:
        save_results = {}
        logger.info("새로운 결과 파일을 생성합니다.")

    # new_key 정의
    new_key = "graph_rag"

    # 결과 처리
    for num, question in reference_questions.items():
        if num in save_results:
            # 기존 데이터에서 new_key 확인
            if new_key in save_results[num]['llm response']:
                logger.info(f"중복된 키 '{new_key}'가 질문 번호 {num}에 이미 존재합니다. 추가하지 않습니다.")
                continue
            else:
                # 새로운 키 추가
                result = chain.invoke({"query": question})
                save_results[num]['llm response'][new_key] = {
                    "answer": result['result']
                    # "page contents": result['contents']
                }
                logger.info(f"'{new_key}'가 질문 번호 {num}에 추가되었습니다.")
        else:
            # 새로운 질문 추가
            result = chain.invoke({"query": question})
            save_results[num] = {
                "number": num,
                "question": question,
                "answer": reference_answers.get(num, ""),
                "answer pages": reference_pages.get(num, []),
                "llm response": {
                    new_key: {
                        "answer": result['result']
                        # "page contents": result['contents']
                    }
                }
            }
            logger.info(f"새 질문 번호 {num}의 결과를 저장했습니다.")

    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(list(save_results.values()), json_file, ensure_ascii=False, indent=4)
    logger.info(f"\n모든 질문에 대한 결과가 json 파일로 저장되었습니다: {output_file}")


if __name__ == "__main__":
    main()
