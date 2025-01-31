import os
import json
import argparse
import pandas as pd

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
config = load_config(os.path.join(current_dir, "config", f"{args.config}.yaml"))

os.environ["LANGCHAIN_TRACING_V2"] = config['langchain']['tracing_v2']
os.environ["LANGCHAIN_API_KEY"] = config['langchain']['api_key']
os.environ["OPENAI_API_KEY"] = config['openai']['api_key']


def main():
    
    # 결과 파일 경로 (수정 예정)
    results_file = 'result_20250131_q_QnA_car_sample_30_context_labeling_markdown.json'
    results_path = f'results/{results_file}'
    
    with open(results_path, 'r') as file:
        results = json.load(file)
        
    # JSON을 DataFrame으로 변환
    data = []
    for method in results[0]['llm response'].keys():
        for i in range(len(results)):
            data.append({
            "question": results[i]['question'],
            "ground truth answer": results[i]['ground truth answer'],
            "ground truth page contents": results[i]['ground truth answer context'],
            "method": method,
            "rag answer": results[i]['llm response'][method]['answer'],
            "rag answer page contents": results[i]['llm response'][method]['page contents']
        })
    
    df = pd.DataFrame(data)
    df = df[:1]

    print("GPT-4o as a Judge를 활용하여 평가를 시작합니다.")
    
    # 1. 정확성 채점
    correctness_grader = CorrectnessGrader(config)
    df[['correctness_score', 'correctness_explanation']] = df.apply(
        lambda row: pd.Series(correctness_grader.grade(row['question'], row['rag answer'], row['ground truth answer'])),
        axis=1
    )
    print("✅ 정확성 채점 완료")
    
    # 2. 관련성 채점
    relevance_grader = RelevanceGrader(config)
    df[['relevance_score', 'relevance_explanation']] = df.apply(
        lambda row: pd.Series(relevance_grader.grade(row['question'], row['rag answer'])),
        axis=1
    )
    print("✅ 관련성 채점 완료")
    
    # 3. 근거성 채점
    grounded_grader = GroundedGrader(config)
    df[['grounded_score', 'grounded_explanation']] = df.apply(
        lambda row: pd.Series(grounded_grader.grade(row['question'], row['rag answer page contents'])),
        axis=1
    )
    print("✅ 근거성 채점 완료")
    
    # 4. 추출 관련성 채점
    retrieval_relevance_grader = RetrievalRelevanceGrader(config)
    df[['retrieval_relevance_score', 'retrieval_relevance_explanation']] = df.apply(
        lambda row: pd.Series(retrieval_relevance_grader.grade(row['question'], row['rag answer page contents'])),
        axis=1
    )
    print("✅ 추출 관련성 채점 완료")
    
    # 결과 저장
    results_file = results_file.replace('.json', '.xlsx')
    output_path = f'evaluation_results/gpt4o-judge_{results_file}'
    df.to_excel(output_path, index=False)
    
    print(f"✅ 평가 결과 저장 완료: {output_path}")
    
if __name__ == "__main__":
    main()