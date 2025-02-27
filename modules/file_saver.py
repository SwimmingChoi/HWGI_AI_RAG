# [2팀]
# QnA 데이터셋으로 RAG 실행 -> 결과 파일 저장을 위한 모듈
# AI센터에서 처음에 받았던 baseline code의 rag_main.py에서 따로 가져와 모듈화함

import pandas as pd
import json
from modules.config_loader import load_config

config = load_config('config/config.yaml')

class JsonSaver:
    def __init__(self, file_path: str):
        self.output_file = file_path
        
    def save_response(self, result, execution_time):
        return {
            "team2 answer": result['answer'],
            "pages": result['context_pages'],
            "page contents": result['context_pages_content'],
            "time": execution_time
        }
        
    def save_to_json(self, save_results):
        with open(self.output_file, 'w', encoding='utf-8') as json_file:
            json.dump(list(save_results.values()), json_file, ensure_ascii=False, indent=4)

class ExcelSaver:
    def __init__(self, file_path: str):
        self.output_file = file_path
        
    def save_to_excel(self):
        # JSON 파일 읽기
        with open(self.output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 필요한 데이터 추출
        excel_data = []
        prev_number = None

        # methods 자동 추출 (첫 번째 항목의 llm response에서 키들을 가져옴)
        methods = list(data[0]['llm response'].keys())

        for item in data:
            # 각 method별로 행 생성
            for i, method in enumerate(methods):
                row = {
                    'number': item['number'] if item['number'] != prev_number else '',
                    'question': item['question'] if item['number'] != prev_number else '',
                    'ground truth answer': item['ground truth answer'] if item['number'] != prev_number else '',
                    'ground truth answer pages': item['ground truth answer pages'].strip('[]') if item['number'] != prev_number else '',
                    'method': method,
                    'team2 answer': item['llm response'][method]['team2 answer'],
                    'rag answer pages': str(item['llm response'][method]['pages']).strip('[]'),
                    'rag answer page contents': item['llm response'][method]['page contents'],
                    'time': item['llm response'][method]['time']
            }
            excel_data.append(row)
            prev_number = item['number']

        # DataFrame 생성
        df = pd.DataFrame(excel_data)

        # 칼럼 순서 지정
        columns = ['number', 'question', 'ground truth answer', 'ground truth answer pages', 'method', 'team2 answer', 'rag answer pages', 'rag answer page contents', 'time']
        df = df[columns]

        # Excel 파일로 저장 (encoding 파라미터 제거)
        df.to_excel(self.output_file.replace('.json', '.xlsx'), index=False)