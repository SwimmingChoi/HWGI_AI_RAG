import yaml
import logging

def load_config(config_path: str) -> dict:
    """YAML 설정 파일 로드"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logging.info("설정 파일 로드 성공")
            return config
    except Exception as e:
        logging.error(f"설정 파일 로드 중 오류: {e}")
        raise