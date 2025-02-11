import os
import json
import argparse
import pandas as pd
from time import time

from modules.config_loader import load_config
from modules.gpt4o_judge import (
    CorrectnessGrader,
    RelevanceGrader,
    GroundedGrader,
    RetrievalRelevanceGrader,
                                 )


parser = argparse.ArgumentParser(description='Enter user name for the config script')
parser.add_argument('--config', type=str, required=False, default='config')
args = parser.parse_args()

# 설정 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(os.path.join(current_dir, "config", f"{args.config}.yaml"))['openai']

def main():
    # 결과 파일 경로 (수정 예정)
    # 지정된 폴더의 json 파일 중 가장 최근 파일 선택
    results_dir = os.path.join(current_dir, 'results')
    results_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    results_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))
    results_file = results_files[-1]
    print(results_file)
    
    with open(os.path.join(current_dir, 'results', results_file), 'r', encoding='utf-8') as file:
        results = json.load(file)
        
    # JSON을 DataFrame으로 변환
    data = []
    for method in results[0]['llm response'].keys():
        for row in results:
            data.append({
            "question": row['question'],
            "ground truth answer": row['ground truth answer'],
            "ground truth answer pages": row['ground truth answer pages'],
            "method": method,
            "rag answer": row['llm response'][method]['answer'],
            "rag answer page contents": row['llm response'][method]['page contents']
        })
    
    df = pd.DataFrame(data)

    print("GPT-4o as a Judge를 활용하여 평가를 시작합니다.")
    
    # 1. 정확성 채점
    start_time = time()
    correctness_grader = CorrectnessGrader(openai_config)
    df[['correctness_score', 'correctness_explanation']] = df.apply(
        lambda row: pd.Series(correctness_grader.grade(row['question'], row['rag answer'], row['ground truth answer'])),
        axis=1
    )
    end_time = time()
    true_ratio = (df['correctness_score'].value_counts(normalize=True).get(True, 0) * 100)
    print(f"✅ 정확성 채점 완료: {true_ratio}% ({end_time - start_time}초)")
    
    # 2. 관련성 채점
    relevance_grader = RelevanceGrader(openai_config)
    df[['relevance_score', 'relevance_explanation']] = df.apply(
        lambda row: pd.Series(relevance_grader.grade(row['question'], row['rag answer'])),
        axis=1
    )
    end_time = time()
    true_ratio = (df['relevance_score'].value_counts(normalize=True).get(True, 0) * 100)
    print(f"✅ 관련성 채점 완료: {true_ratio}% ({end_time - start_time}초)")
    
    # 3. 근거성 채점
    start_time = time()
    grounded_grader = GroundedGrader(openai_config)
    df[['grounded_score', 'grounded_explanation']] = df.apply(
        lambda row: pd.Series(grounded_grader.grade(row['question'], row['rag answer page contents'])),
        axis=1
    )
    end_time = time()
    true_ratio = (df['grounded_score'].value_counts(normalize=True).get(True, 0) * 100)
    print(f"✅ 근거성 채점 완료: {true_ratio}% ({end_time - start_time}초)")

    
    # 4. 추출 관련성 채점
    start_time = time()
    retrieval_relevance_grader = RetrievalRelevanceGrader(openai_config)
    df[['retrieval_relevance_score', 'retrieval_relevance_explanation']] = df.apply(
        lambda row: pd.Series(retrieval_relevance_grader.grade(row['question'], row['rag answer page contents'])),
        axis=1
    )
    end_time = time()
    true_ratio = (df['retrieval_relevance_score'].value_counts(normalize=True).get(True, 0) * 100)
    print(f"✅ 추출 관련성 채점 완료: {true_ratio}% ({end_time - start_time}초)")
    
    # 결과 저장
    os.makedirs('evaluation_results', exist_ok=True)
    results_file = results_file.replace('.json', '.xlsx')
    output_path = f'evaluation_results/gpt4o-judge_{results_file}'
    df.to_excel(output_path, index=False)
    
    print(f"✅ 평가 결과 저장 완료: {output_path}")
    
if __name__ == "__main__":
    main()