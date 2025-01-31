from modules.config_loader import load_config

import re
import os
import string
import collections
import pandas as pd
from typing import Annotated, TypedDict
from collections import Counter

from sklearn.metrics import precision_score, recall_score, f1_score

from langchain_openai import ChatOpenAI

import argparse

parser = argparse.ArgumentParser(description='Enter user name for the config script')
parser.add_argument('--config', type=str, required=False, default='config')
args = parser.parse_args()

# 설정 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(os.path.join(current_dir, "config", f"{args.config}.yaml"))

# make lists of prediction and reference
def make_lists(results, method):
    predictions = []
    references = []
    for i in range(len(list(results['llm response'].keys()))):
        predictions.append(results['llm response'][i][method]['answer'])
        references.append(results['ground truth answer'][i])
    return predictions, references

####################################################################
#                        GPT-4o as a Judge                         #
####################################################################
# Grade output schema
class CorrectnessGrade(TypedDict):
    # Note that the order in the fields are 
    # defined is the order in which the model will generate them.
    # It is useful to put explanations before responses because it forces the model to think through
    # its final response before generating it:
    explanation: Annotated[str, ..., "왜 이 점수를 주었는지 설명해주세요."]
    correct: Annotated[bool, ..., "답변이 옳다면 True, 틀리다면 False"]
# Grade prompt
correctness_instructions = f"""당신은 문제를 채점하는 교사입니다. 

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

# Grader LLM
grader_llm = ChatOpenAI(model=config['openai']['gpt_model'], temperature=0).with_structured_output(CorrectnessGrade, method="json_schema", strict=True)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for RAG answer accuracy"""
    answers = f"""질문: {inputs}
정답: {reference_outputs}
학생 답변: {outputs}
"""

    # Run evaluator
    grade = grader_llm.invoke([{"role": "system", "content": correctness_instructions}, {"role": "user", "content": answers}])
    return grade["correct"], grade["explanation"]