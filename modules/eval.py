from modules.config_loader import load_config

import re
import os
import torch
import string
import collections
import pandas as pd
from typing import Annotated, TypedDict, List, Dict, Union
from collections import Counter

from langchain_teddynote.community.kiwi_tokenizer import KiwiTokenizer
from langsmith.schemas import Run, Example
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


kiwi_tokenizer = KiwiTokenizer()

class F1Score:
    """
    F1Score 클래스는 참조 답변과 예측 답변의 F1 점수를 계산합니다.
    
    """
    def __init__(self, a_gold, a_pred):
        self.a_gold = a_gold
        self.a_pred = a_pred

    def _normalize_answer(self, s: str) -> str:
        """텍스트 정규화: 소문자 변환, 구두점 제거, 관사 제거, 공백 정리"""
        def remove_articles(text: str) -> str:
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _get_tokens(self, s: str) -> List[str]:
        """정규화된 텍스트를 토큰화"""
        if not s:
            return []
        return self._normalize_answer(s).split()

    def compute_f1_metrics(self) -> Dict[str, Union[float, List[str]]]:
        """F1 score와 관련 메트릭을 계산"""
        gold_toks = self._get_tokens(self.a_gold)
        pred_toks = self._get_tokens(self.a_pred)
        
        # Counter 객체로 변환하여 각 토큰의 빈도수 계산
        gold_counter = collections.Counter(gold_toks)
        pred_counter = collections.Counter(pred_toks)
        
        # 공통 토큰 찾기
        common = gold_counter & pred_counter
        num_same = sum(common.values())
        
        # 빈 답변 처리
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return {
                'f1': float(gold_toks == pred_toks),
                'precision': float(gold_toks == pred_toks),
                'recall': float(gold_toks == pred_toks),
                'common_tokens': list(common.keys()),
                'gold_only': list(gold_counter - common),
                'pred_only': list(pred_counter - common)
            }
        
        # 메트릭 계산
        precision = num_same / len(pred_toks) if pred_toks else 0
        recall = num_same / len(gold_toks) if gold_toks else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1


def rouge_evaluator(reference: str, candidate: str) -> Dict:
    # wrapper function 정의
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"], use_stemmer=True, tokenizer=kiwi_tokenizer
    )
    rouge_score = scorer.score(reference, candidate)
    return rouge_score['rougeL']
    
    
def semscore_evaluator(reference: str, candidate: str) -> dict:
    try:
        # SentenceTransformer 모델 로드
        model = SentenceTransformer("all-mpnet-base-v2",)

        # 문장 임베딩 생성
        student_embedding = model.encode(candidate, convert_to_tensor=True)
        reference_embedding = model.encode(reference, convert_to_tensor=True)

        # 코사인 유사도 계산
        cosine_similarity = util.pytorch_cos_sim(
            student_embedding, reference_embedding
        ).item()

        return cosine_similarity
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # GPU 캐시 메모리 정리
        
        # 모델이 존재하는 경우에만 삭제
        if 'model' in locals():
            del model
        if 'student_embedding' in locals():
            del student_embedding
        if 'reference_embedding' in locals():
            del reference_embedding


def bleu_evaluator(reference: str, candidate: str) -> dict:
    # 토큰화
    reference_tokens = kiwi_tokenizer.tokenize(reference, type="sentence")
    candidate_tokens = kiwi_tokenizer.tokenize(candidate, type="sentence")

    # BLEU 점수 계산
    bleu_score = sentence_bleu(
        [reference_tokens], 
        candidate_tokens, 
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1)

    return bleu_score

def fail_rate_evaluator(candidate: str) -> dict:
    fail_sentence = "모르겠습니다."
    return 1 if fail_sentence in candidate else 0
