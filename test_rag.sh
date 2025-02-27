# [2팀]
# 모든 조합에 대한 RAG 테스트 스크립트
# 최종 조합에 대해서는 config.yaml 파일에 저장
# 해당 스크립트는 test_config.yaml 파일을 사용

#!/bin/bash

# 시작 시간 측정
start_time=$(date +%s)

# 기본 설정
CONFIG_FILE="./config/test_config.yaml"
SCRIPT="./rag_main.py"
# SCRIPT="./graph_rag_main.py"

# 가능한 조건 값 정의
chunking_values=("False" "True")
retrieval_types=("hybrid" "sparse" "dense")
model_types=("sts" "api")
reranker_values=("True" "False")
query_translation_values=("QueryRewrite" "HyDE" "None")

# 모든 조합을 반복
for chunking in "${chunking_values[@]}"
do
  for retrieval_type in "${retrieval_types[@]}"
  do
    for reranker in "${reranker_values[@]}"
    do
      for model_type in "${model_types[@]}"
      do
        for query_translation in "${query_translation_values[@]}"
        do
          # top_k 값을 조건에 따라 설정
          if [ "$reranker" == "True" ]; then
            top_k=10
          else
            top_k=10
          fi

          sed -i "s/^  chunking:.*$/  chunking: $chunking/" $CONFIG_FILE
          sed -i "/^retrieval:/,/^[a-z]/ s/^  type:.*$/  type: $retrieval_type/" $CONFIG_FILE
          sed -i "/^retrieval:/,/^[a-z]/ s/^  top_k:.*$/  top_k: $top_k/" $CONFIG_FILE
          sed -i "/^retrieval:/,/^[a-z]/ s/^  reranker:.*$/  reranker: $reranker  # Reranker 사용 여부/" $CONFIG_FILE
          sed -i "/^dense_model:/,/^[a-z]/ s/^  model_type:.*$/  model_type: $model_type/" $CONFIG_FILE
          sed -i "/^query_translation:/,/^[a-z]/ s/^  type:.*$/  type: $query_translation/" $CONFIG_FILE
          # 변경된 config.yaml 출력
          echo "Updated config.yaml:"
          # grep -A 5 "retrieval:" $CONFIG_FILE
          echo "Running with combination: chunking=$chunking, query_translation=$query_translation, retrieval_type=$retrieval_type, top_k=$top_k, reranker=$reranker"
        
          python $SCRIPT

          echo "----------------------------------------"
        done
      done
    done
  done
done

# 종료 시간 측정
end_time=$(date +%s)
# 실행 시간 계산
elapsed_time=$((end_time - start_time))
echo "Execution time: $elapsed_time seconds"