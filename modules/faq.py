from openai import OpenAI
import os

class FAQ:
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        
    def find_similar_sentence(self, input_sentence, question_dictionary):
        # 각 문장에 대해 유사도 검사
        prompt = f"""입력 문장이 question_dictionary에 있는 문장과 의미적으로 동일한지 판단해주세요:
        
입력문장: {input_sentence}
question_dictionary: {question_dictionary}

반드시 짧게 숫자로만 답변해주세요.
무조건 의미가 동일해야 합니다. 입력문장 일부분이 누락되거나 과장되지 않아야 합니다. 
의미적으로 동일한 문장이 없다면 -1을,
동일한 문장이 있다면 해당 문장의 인덱스를 반환해주세요.
동일한 문장이 여러개 있다면 가장 유사한 문장의 인덱스를 반환해주세요. """
            # API
        response = self.client.chat.completions.create(
            model="gpt-4o",  # 또는 "gpt-3.5-turbo"
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 일관된 결과를 위해 낮은 temperature 사용
        )
        # 응답 확인
        answer= int(response.choices[0].message.content.strip().lower())
        return answer
    
    def run_faq(self, question, question_dictionary, faq_data):
        return_answer=''
        checker = FAQ(self.api_key)
        qd = question_dictionary  # 딕셔너리인 경우 리스트로 변환
        try:
            similar_index = int(checker.find_similar_sentence(question, qd))
            if similar_index != -1:
                return_answer =faq_data[similar_index]['answer']
                if faq_data[similar_index]['pages'] is not None:
                    return_answer+=f"""*****
    자세한 내용은 {faq_data[similar_index]['pages']}페이지 {faq_data[similar_index]['title']}에서 확인해주세요."""
                return return_answer
            else:
                return False
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            return False
