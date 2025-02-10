import pandas as pd
import json
from modules.config_loader import load_config

config = load_config('config/config.yaml')

class ExcelSaver:
    def __init__(self, file_path: str):
        self.output_file = file_path
        
    def save_to_excel(self, data: dict, sheet_name: str):
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
                if 'context_labeling' in config['dataset']['qna']:
                    row = {
                        'number': item['number'] if item['number'] != prev_number else '',
                        'question': item['question'] if item['number'] != prev_number else '',
                        'ground truth answer': item['ground truth answer'] if item['number'] != prev_number else '',
                        'ground truth answer pages': item['ground truth answer pages'].strip('[]') if item['number'] != prev_number else '',
                        'method': method,
                        'answer': item['llm response'][method]['answer'],
                        'explanation': item['llm response'][method]['explanation'],
                        'rag answer pages': str(item['llm response'][method]['pages']).strip('[]'),
                        'rag answer page contents': item['llm response'][method]['page contents'],
                        'time': item['llm response'][method]['time']
                    }
                else:
                    row = {
                        'number': item['number'] if item['number'] != prev_number else '',
                        'question': item['question'] if item['number'] != prev_number else '',
                        'ground truth answer': item['ground truth answer'] if item['number'] != prev_number else '',
                        'ground truth answer pages': item['ground truth answer pages'].strip('[]') if item['number'] != prev_number else '',
                        'method': method,
                        'answer': item['llm response'][method]['answer'],
                        'explanation': item['llm response'][method]['explanation'],
                        'rag answer pages': str(item['llm response'][method]['pages']).strip('[]'),
                        'rag answer page contents': item['llm response'][method]['page contents'],
                        'time': item['llm response'][method]['time']
                    # '실제QA': item['실제QA'] if item['number'] != prev_number else '',
                    # 'document' : item['document'] if item['number'] != prev_number else ''
                }
                excel_data.append(row)
                prev_number = item['number']

        # DataFrame 생성
        df = pd.DataFrame(excel_data)

        # 칼럼 순서 지정
        columns = ['number', 'question', 'ground truth answer', 'ground truth answer pages', 'method', 'answer', 'rag answer pages', 'rag answer page contents', 'time']
        df = df[columns]

        # Excel 파일로 저장 (encoding 파라미터 제거)
        df.to_excel(self.output_file.replace('.json', '.xlsx'), index=False)