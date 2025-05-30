# [2팀]
# 질문 translation/expansion 모듈
# HyDE(Hypothetical Document Expansion) 및 Query Rewriting 모듈

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import Annotated, TypedDict


# [2팀] Query Expansion: HyDE
class HyDE:
    def __init__(self, openai_config: dict):
        self.config = openai_config
        self.model = ChatOpenAI(
            model=openai_config['gpt_model'], temperature=0.5, api_key=openai_config['api_key']
            )

    def translate(self, question: str) -> str:
        """
        [2팀] 질문에 대한 보험 약관 단락을 생성하는 함수입니다.
        
        Parameters:
        question (str): 사용자의 질문
        
        Returns:
        str: LLM이 생성한 가상의 보험 약관 단락
        """
        prompt = ChatPromptTemplate.from_template(
            "해당 질문에 답하기 위한 보험 약관 단락을 작성해주세요. \n 질문: {question} \n 단락:"
        )
        chain = prompt | self.model | StrOutputParser()
        return chain.invoke({"question": question})


# [2팀] Query Expansion: Query Rewriting
class Rewriting(TypedDict):
    explanation: Annotated[str, ..., "왜 이렇게 재구성했는지 설명해주세요."]
    answer: Annotated[str, ..., "재구성된 질문을 써주세요."]

class QueryRewrite:
    def __init__(self, openai_config: dict):
        self.config = openai_config
        self.model = ChatOpenAI(
            model=openai_config['gpt_model'], temperature=0.5, api_key=openai_config['api_key']
            ).with_structured_output(
                Rewriting, method="json_schema", strict=True
            )

    def rewrite(self, question: str) -> str:
        """
        [2팀] RAG에 용이한 형태로 질문을 재구성하는 함수입니다.
        
        Parameters:
        question (str): 사용자의 질문
        
        Returns:
        str: LLM이 생성한 재구성된 질문
        """
        rewrite_instructions = """당신은 보험 약관과 관련된 질문을 받게 될 것입니다.

다음과 같은 기준을 충족하며 질문을 재구성해주십시오:
1. 질문의 의도를 파악하고, 약관 문서에서 질문자가 원하는 내용을 찾을 수 있도록 재구성해주십시오.
2. 재구성된 질문은 원래 질문에 있는 내용을 바꾸거나 빠트려서는 안됩니다.
3. 특히 약관 이름의 경우 바꾸지 않도록 유의하십시오.
4. 재구성할 필요가 없는 단순한 질문은 그대로 쓰십시오.

답변을 단계별로 설명하여 충족되지 않은 기준이 없는지 확인하십시오.

처음부터 단순히 재구성된 질문을 쓰지 마십시오."""

        original_question = f"질문: {question}"
        input_prompt = [
            {"role": "system", "content": rewrite_instructions},
            {"role": "user", "content": original_question}
        ]
        # print(input_prompt)
        rewritten_question = self.model.invoke(input_prompt)
        return rewritten_question['answer']
