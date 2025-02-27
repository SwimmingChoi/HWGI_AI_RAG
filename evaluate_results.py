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
from modules.eval import (
    F1Score,
    rouge_evaluator,
    semscore_evaluator,
    bleu_evaluator,
    fail_rate_evaluator
)

from utils.logger import setup_logger


parser = argparse.ArgumentParser(description='Enter user name for the config script')
parser.add_argument('--config', type=str, required=False, default='config')
args = parser.parse_args()

# 설정 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
openai_config = load_config(os.path.join(current_dir, "config", f"{args.config}.yaml"))['openai']
logger = setup_logger()

def gpt4o_eval(df):
    # 1. 정확성 채점
    start_time = time()
    correctness_grader = CorrectnessGrader(openai_config)
    def compute_correctness_score(row):
        score, explanation = correctness_grader.grade(row['question'], row['rag answer'], row['ground truth answer'])
        return {
            'correctness_score': score,
            'correctness_explanation': explanation
        }
    df.loc[:, ['correctness_score', 'correctness_explanation']] = df.apply(
        lambda row: pd.Series(compute_correctness_score(row)),
        axis=1
    )
    end_time = time()
    print(f"정확성 채점 완료: {end_time - start_time:.2f}초")
    
    # 2. 관련성 채점
    start_time = time()
    relevance_grader = RelevanceGrader(openai_config)
    def compute_relevance_score(row):
        score, explanation = relevance_grader.grade(row['question'], row['rag answer'])
        return {
            'relevance_score': score,
            'relevance_explanation': explanation
        }
    df.loc[:, ['relevance_score', 'relevance_explanation']] = df.apply(
        lambda row: pd.Series(compute_relevance_score(row)),
        axis=1
    )
    end_time = time()
    print(f"관련성 채점 완료: {end_time - start_time:.2f}초")
    
    # 3. 근거성 채점
    start_time = time()
    grounded_grader = GroundedGrader(openai_config)
    def compute_grounded_score(row):
        score, explanation = grounded_grader.grade(row['question'], row['rag answer page contents'])
        return {
            'grounded_score': score,
            'grounded_explanation': explanation
        }
    df.loc[:, ['grounded_score', 'grounded_explanation']] = df.apply(
        lambda row: pd.Series(compute_grounded_score(row)),
        axis=1
    )
    end_time = time()
    print(f"근거성 채점 완료: {end_time - start_time:.2f}초")
    
    # 4. 추출 관련성 채점
    start_time = time()
    retrieval_relevance_grader = RetrievalRelevanceGrader(openai_config)
    def compute_retrieval_relevance_score(row):
        score, explanation = retrieval_relevance_grader.grade(row['question'], row['rag answer page contents'])
        return {
            'retrieval_relevance_score': score,
            'retrieval_relevance_explanation': explanation
        }
    df.loc[:, ['retrieval_relevance_score', 'retrieval_relevance_explanation']] = df.apply(
        lambda row: pd.Series(compute_retrieval_relevance_score(row)),
        axis=1
    )
    end_time = time()
    print(f"추출 관련성 채점 완료: {end_time - start_time:.2f}초")
    
    return df

def quantitative_eval(df):
    # 1. Precision, Recall, F1 점수 계산
    def conpute_f1(row):
        f1_score = F1Score(row['ground truth answer'], row['rag answer'])
        precision, recall, f1 = f1_score.compute_f1_metrics()
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    df.loc[:, ['precision', 'recall', 'f1']] = df.apply(
        lambda row: pd.Series(conpute_f1(row)),
        axis=1
    )
    
    # 2. ROUGE 점수 계산
    df.loc[:, 'rougeL'] = df.apply(
        lambda row: pd.Series(
            rouge_evaluator(row['ground truth answer'], row['rag answer']).fmeasure
            ),
        axis=1
    )

    # 3. SemScore 점수 계산
    df.loc[:, 'semscore'] = df.apply(
        lambda row: pd.Series(
            semscore_evaluator(row['ground truth answer'], row['rag answer'])
            ),
        axis=1
    )
    
    # 4. BLEU 점수 계산
    df.loc[:, 'bleu'] = df.apply(
        lambda row: pd.Series(
            bleu_evaluator(row['ground truth answer'], row['rag answer'])
            ),
        axis=1
    )
    
    # 5. Fail Rate 계산
    df.loc[:, 'fail_rate'] = df.apply(
        lambda row: pd.Series(
            fail_rate_evaluator(row['rag answer'])
            ),
        axis=1
    )
    
    return df

def main():
    # 결과 파일 경로 (수정 예정)
    # 지정된 폴더의 json 파일 중 가장 최근 파일 선택
    results_dir = os.path.join(current_dir, 'results')
    results_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    results_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))
    results_file = results_files[-1]
    
    with open(os.path.join(current_dir, 'results', results_file), 'r', encoding='utf-8') as file:
        results = json.load(file)
        
    os.makedirs('evaluation_results', exist_ok=True)
    results_file = results_file.replace('.json', '.xlsx')
    output_file = f'evaluation_results/gpt4o-judge_{results_file}'
    
    # 컬럼 순서 정리
    column_order = [
        'method', 'question', 'ground truth answer', 'rag answer',
        'correctness_score', 'correctness_explanation',
        'relevance_score', 'relevance_explanation',
        'grounded_score', 'grounded_explanation',
        'retrieval_relevance_score', 'retrieval_relevance_explanation',
        'precision', 'recall', 'f1',
        'rougeL', 'bleu', 'semscore', 'fail_rate', "time"
    ]
    
    # 기존 결과 로드
    if os.path.exists(output_file):
        save_results = pd.read_excel(output_file)
        logger.info("기존 결과 파일을 불러왔습니다.")
    else:
        save_results = pd.DataFrame(
            columns=column_order
        )
        logger.info("새로운 결과 파일을 생성합니다.")
    
# DataFrame 생성 부분 수정
    data = []
    for row in results:
        for method, response in row['llm response'].items():
            data.append({
                "question": row['question'],
                "ground truth answer": row['ground truth answer'],
                "ground truth answer pages": row['ground truth answer pages'],
                "method": method,
                "rag answer": response['answer'],  # 메서드별 응답 접근
                "rag answer page contents": response['page contents'],  # 메서드별 페이지 내용
                "time": response['time']
            })
    
    df = pd.DataFrame(data)
    
    print("평가를 시작합니다.")
    evaluation_results = []
    new_methods_found = False  # 새로운 method가 있는지 확인하는 플래그

    for method in df['method'].unique():
        if method in save_results['method'].unique():
            logger.info(f"중복된 메서드 '{method}'가 이미 존재합니다. 추가하지 않습니다.")
            continue
        else:
            new_methods_found = True  # 새로운 method 발견
            df_method = df[df['method'] == method].copy()

            print('-'*100)
            print(f"[{method}]")
            print('-'*100)
            
            # 양적 평가
            print("양적 평가 시작...")
            df_method = quantitative_eval(df_method)
            print("✅ 양적 평가 완료")
            
            # GPT-4o 평가
            print(f"GPT-4o 평가 시작")
            df_method = gpt4o_eval(df_method)
            print("✅  GPT-4o 평가 완료")
            
            evaluation_results.append(df_method)
    
    # 결과 통합
    if new_methods_found:  # 새로운 method가 있을 경우에만 concat 수행
        final_df = pd.concat(evaluation_results, axis=0, ignore_index=True)
        if len(save_results) > 0:
            final_df = pd.concat([save_results, final_df], axis=0, ignore_index=True)
    else:
        final_df = save_results.copy()  # 새로운 method가 없으면 기존 결과를 그대로 사용

    
    # 결과 저장 전 데이터 확인
    print("\n=== 최종 데이터 확인 ===")
    print(f"전체 데이터 수: {len(final_df)}")
    print(f"메서드별 데이터 수:")
    print(final_df['method'].value_counts())
    print("\n정확성 점수 분포:")
    print(final_df['correctness_score'].value_counts(normalize=True) * 100)
    
    final_df = final_df[column_order]
    
    # 결과 저장
    final_df.to_excel(output_file, index=False)
    
    # 메서드별 평균 점수 출력
    print("\n=== 평가 결과 요약 ===")
    for method in final_df['method'].unique():
        method_df = final_df[final_df['method'] == method]
        print(f"\n[{method}]")
        print(f"정확성 점수: {(method_df['correctness_score'] == True).mean():.1%}")
        print(f"관련성 점수: {(method_df['relevance_score'] == True).mean():.1%}")
        print(f"근거성 점수: {(method_df['grounded_score'] == True).mean():.1%}")
        print(f"추출 관련성 점수: {(method_df['retrieval_relevance_score'] == True).mean():.1%}")
        print(f"F1 점수: {method_df['f1'].mean():.3f}")
        print(f"ROUGE-L 점수: {method_df['rougeL'].mean():.3f}")
        print(f"SemScore 점수: {method_df['semscore'].mean():.3f}")
        print(f"BLEU 점수: {method_df['bleu'].mean():.3f}")
        print(f"Fail Rate: {method_df['fail_rate'].mean():.3f}")
        print(f"소요 시간: {method_df['time'].mean():.3f}초")
        
    print(f"\n✅ 평가 결과 저장 완료: {output_file}")

if __name__ == "__main__":
    main()