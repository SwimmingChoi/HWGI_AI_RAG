from openai import OpenAI
import os

class FAQ:
    """
    FAQ를 처리하기 위한 클래스입니다.
    OpenAI API를 사용하여 입력된 질문과 FAQ 데이터베이스의 질문들 간의 의미적 유사성을 판단합니다.
    """
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        
    def find_similar_sentence(self, input_sentence, question_dictionary):
        """
        사용자 질문(input sentence)과 FAQ질문사전(question_dictionary)에 있는 질문들 간의 의미적 동일성을 판단합니다.
        
        Parameters:
            input_sentence (str): 사용자가 입력한 질문 문장
            question_dictionary (list): FAQ 데이터셋 내 질문들의 리스트
            
        Returns:
            int: 입력 문장과 의미적으로 가장 가까운 문장의 인덱스
                 동일한 문장이 없는 경우 -1 반환
        """
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
            model="gpt-4o",  
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, 
        )
        # 응답 확인
        answer= int(response.choices[0].message.content.strip().lower())
        return answer
    
    def run_faq(self, question, question_dictionary, faq_data):
        """
        사용자 질문이 FAQ데이터베이스 내의 질문들과 의미적으로 동일한지 판단하고, 의미적으로 동일한 질문이 존재한다면 FAQ 데이터베이스 내 해당당 답변을 찾아 반환합니다.
        
        Parameters:
            question (str): 사용자가 입력한 질문
            question_dictionary (list): 비교할 질문들이 저장된 리스트
            faq_data (list): FAQ 데이터베이스 ('question', 'answer', 'pages', 'title' 키 포함)
            
        Returns:
            str: 찾은 답변 문자열 (페이지, 조항항 정보가 있으면 함께 표시)
            bool: 매칭되는 질문을 찾지 못한 경우 False 반환
        """
        return_answer=''
        checker = FAQ(self.api_key)
        qd = question_dictionary  
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


