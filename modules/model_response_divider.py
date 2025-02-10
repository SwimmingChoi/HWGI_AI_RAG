import re

def parse_model_response(response: str) -> dict:
    """
    모델 응답을 explanation과 answer로 분리
    
    Args:
        response (str): 모델의 전체 응답 텍스트
    
    Returns:
        dict: {'Explanation': str, 'Answer': str} 형태의 딕셔너리
    """
    try:
        # 정규표현식 패턴
        explanation_pattern = r"explanation:\s*(.*?)(?=\n*answer:)"
        answer_pattern = r"answer:\s*(.*?)$"
        
        # explanation 추출
        explanation_match = re.search(explanation_pattern, response, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        
        # answer 추출
        answer_match = re.search(answer_pattern, response, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""
        
        return {
            "Explanation": explanation,
            "Answer": answer
        }
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        return {
            "Explanation": "",
            "Answer": response  # 파싱 실패시 전체 응답을 answer로 반환
        }