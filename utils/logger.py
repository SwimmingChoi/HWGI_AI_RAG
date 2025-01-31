import logging
import os
from datetime import datetime
import pytz

def setup_logger(LOG_FORMAT='%(asctime)s - %(levelname)s - %(message)s', DEFAULT_LOG_PATH=None):
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    if DEFAULT_LOG_PATH is None:
        DEFAULT_LOG_PATH = os.path.join(current_dir, "log", "rag_system.log")
    else:
        DEFAULT_LOG_PATH = os.path.join(current_dir, DEFAULT_LOG_PATH)
        
    # 로그 파일 디렉토리 생성
    os.makedirs(os.path.dirname(DEFAULT_LOG_PATH), exist_ok=True)
    
    logger = logging.getLogger(__name__)
    # 기존 핸들러 제거
    logger.handlers.clear()
    
    # KST 포맷터 클래스 정의
    class KSTFormatter(logging.Formatter):
        def converter(self, timestamp):
            dt = datetime.fromtimestamp(timestamp)
            return dt.astimezone(pytz.timezone('Asia/Seoul'))

        def formatTime(self, record, datefmt=None):
            dt = self.converter(record.created)
            if datefmt:
                return dt.strftime(datefmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    
    # 새로운 핸들러 추가
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(DEFAULT_LOG_PATH, encoding='utf-8')
    
    # KST 포맷터 설정
    formatter = KSTFormatter(LOG_FORMAT)
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    return logger
