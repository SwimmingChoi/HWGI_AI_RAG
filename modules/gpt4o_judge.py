import os
import collections
import pandas as pd
from typing import Annotated, TypedDict, List, Optional, Dict
from collections import Counter

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
import logging

####################################################################
#                        Correctness Judge                         #
####################################################################

class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "왜 이 점수를 주었는지 설명해주세요."]
    correct: Annotated[bool, ..., "답변이 옳다면 True, 틀리다면 False"]

class CorrectnessGrader:
    def __init__(self, openai_config: dict):
        # 초기화 시 config 값을 저장하고 OpenAI 모델을 설정
        self.openai_config = openai_config
        self.grader_llm = ChatOpenAI(
            model=openai_config['gpt_model'],
            temperature=0,
            openai_api_key=openai_config['api_key']
        ).with_structured_output(
            CorrectnessGrade, method="json_schema", strict=True
        )

    def grade(self, inputs: dict, outputs: dict, reference_outputs: dict) -> tuple:
        # 학생 답변을 채점하는 메서드
        correctness_instructions = """당신은 문제를 채점하는 교사입니다. 

        질문, 정답, 그리고 학생 답변이 주어질 것입니다.

        채점 기준은 다음과 같습니다:
        1. 학생 답변을 정답과 비교하여 사실적 정확성만을 평가합니다.
        2. 학생 답변에 상충되는 진술이 포함되지 않았는지 확인하십시오.
        3. 학생 답변이 정답보다 더 많은 정보를 포함하더라도, 정답과 비교했을 때 사실적으로 정확하다면 정답으로 평가합니다.

        정답 여부:
        정답 여부가 True라면, 학생 답변이 모든 채점 기준을 충족했음을 의미합니다.
        정답 여부가 False라면, 학생 답변이 모든 채점 기준을 충족하지 않았음을 의미합니다.

        답변을 단계별로 설명하여 추론과 결론이 올바른지 확인하십시오.

        처음부터 정답을 단순히 말하지 마십시오.
        """

        answers = f"""질문: {inputs}
        정답: {reference_outputs}
        학생 답변: {outputs}
        """

        # OpenAI LLM을 호출하여 채점 수행
        grade = self.grader_llm.invoke([
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": answers}
        ])
        
        return grade["correct"], grade["explanation"]
    

####################################################################
#                          Relevance Judge                         #
####################################################################

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "왜 이 점수를 주었는지 설명해주세요."]
    relevant: Annotated[bool, ..., "답변이 질문을 적절히 다루는지에 대해 True, 아니면 False를 제공하세요."]

class RelevanceGrader:
    def __init__(self, openai_config: dict):
        # 초기화 시 config 값을 저장하고 OpenAI 모델을 설정
        self.openai_config = openai_config
        self.grader_llm = ChatOpenAI(
            model=openai_config['gpt_model'],
            temperature=0,
            openai_api_key=openai_config['api_key']
        ).with_structured_output(
            RelevanceGrade, method="json_schema", strict=True
        )
    
    def grade(self, inputs: dict, outputs: dict) -> tuple:
        # 학생 답변을 채점하는 메서드
        relevance_instructions="""당신은 문제를 채점하는 교사입니다.

        질문과 학생 답변이 주어질 것입니다.

        채점 기준은 다음과 같습니다:
        1. 학생 답변이 질문에 대해 간결하고 적절한지 확인하세요.
        2. 학생 답변이 질문에 대한 답을 제공하는 데 도움이 되는지 확인하세요.

        관련성:
        관련성이 True일 경우, 학생의 답변이 위의 기준을 모두 충족했음을 의미합니다.
        관련성이 False일 경우, 학생의 답변이 위의 기준을 모두 충족하지 못했음을 의미합니다.

        답변을 단계별로 설명하여 추론과 결론이 올바른지 확인하시오.

        처음부터 정답을 단순히 말하지 마시오.
        """
        
        answer = f"""질문: {inputs}
        학생 답변: {outputs}"""
        
        grade = self.grader_llm.invoke([
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": answer}
        ])

        return grade["relevant"], grade["explanation"]


####################################################################
#                       Groundedness Judge                         #
####################################################################

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "왜 이 점수를 주었는지 설명해주세요."]
    grounded: Annotated[bool, ..., "답변이 문서를 근거로 하고 있는지에 대해 True, 아니면 False를 제공하세요."]

class GroundedGrader:
    def __init__(self, openai_config: dict):
        # 초기화 시 config 값을 저장하고 OpenAI 모델을 설정
        self.openai_config = openai_config
        self.grader_llm = ChatOpenAI(
            model=openai_config['gpt_model'],
            temperature=0,
            openai_api_key=openai_config['api_key']
        ).with_structured_output(
            GroundedGrade, method="json_schema", strict=True
        )
    
    def grade(self, inputs: str, contents: list) -> tuple:
        # 학생 답변을 채점하는 메서드
        grounded_instructions = """당신은 문제를 채점하는 교사입니다. 

        사실과 학생 답변이 주어질 것입니다.

        채점 기준은 다음과 같습니다:
        1. 학생 답변이 주어진 사실에 근거하고 있는지 확인하세요.
        2. 학생 답변이 주어진 사실에 포함되지 않은 "허위 정보"를 포함하지 않았는지 확인하세요.

        근거 여부:
        근거 여부가 True라면, 학생 답변이 모든 채점 기준을 충족했음을 의미합니다.
        근거 여부가 False라면, 학생 답변이 모든 채점 기준을 충족하지 않았음을 의미합니다.

        답변을 단계별로 설명하여 추론과 결론이 올바른지 확인하시오.

        처음부터 정답을 단순히 말하지 마시오.
        """
        doc_string = "".join(contents)
        answer = f"""사실: {doc_string}
        학생 답변: {inputs}"""

        grade = self.grader_llm.invoke([
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": answer}
        ])
        
        return grade["grounded"], grade["explanation"]


####################################################################
#                    Retrieval Relevance Judge                     #
####################################################################

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "왜 이 점수를 주었는지 설명해주세요."]
    relevant: Annotated[bool, ..., "수집된 문서가 질문과 관련되어 있는지에 대해 True, 아니면 False를 제공하세요."]

class RetrievalRelevanceGrader:
    def __init__(self, openai_config: dict):
        # 초기화 시 config 값을 저장하고 OpenAI 모델을 설정
        self.openai_config = openai_config
        self.grader_llm = ChatOpenAI(
            model=openai_config['gpt_model'],
            temperature=0,
            openai_api_key=openai_config['api_key']
        ).with_structured_output(
            RetrievalRelevanceGrade, method="json_schema", strict=True
        )

    def grade(self, inputs: str, contents: list) -> tuple:
        # 학생 답변을 채점하는 메서드
        retrieval_relevance_instructions = """당신은 문제를 채점하는 교사입니다.

        질문과 학생이 제공한 사실이 주어질 것입니다.

        채점 기준은 다음과 같습니다.
        1. 당신의 목표는 질문과 완전히 무관한 사실을 식별하는 것입니다.
        2. 만약 사실이 질문과 관련된 키워드나 의미를 조금이라도 포함하고 있다면, 해당 사실을 관련성이 있다고 평가하세요.
        3. 사실과 무관한 정보가 일부 포함되어 있더라도, 위의 2. 기준을 충복한다면 관련성이 있다고 판단하세요.

        관련성:
        관련성이 True라면, 사실이 질문과 관련된 키워드나 의미를 조금이라도 포함하고 있음을 의미합니다.
        관련성이 False라면, 사실이 질문과 완전히 무관함을 의미합니다.

        답변을 단계별로 설명하여 추론과 결론이 올바른지 확인하시오.

        처음부터 정답을 단순히 말하지 마시오.

        답변은 한국어로 제시하시오."""

        doc_string = "".join(contents)
        answer = f"""사실: {doc_string}
        질문: {inputs}"""
        
        grade = self.grader_llm.invoke([
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": answer}
        ])
        
        return grade["relevant"], grade["explanation"]