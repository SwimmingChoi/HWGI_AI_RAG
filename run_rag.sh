# [2팀]
# 데이터셋 결과 출력 및 평가 실행 스크립트
# step 1. rag_main.py 실행 -> QnA 데이터셋에 대한 결과 파일 생성
# step 2. evaluate_results.py 실행 -> 결과 파일 평가 후 evaluation_results 폴더에 저장

python rag_main.py --config config

wait

python evaluate_results.py --config config

# [2팀] 챗봇 데모 실행 시 아래 코드 주석 해제
# python chatbot_gradio.py --config config