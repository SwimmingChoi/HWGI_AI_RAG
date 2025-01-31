# LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
# DEFAULT_LOG_PATH = '/volume/log/rag_system.log'

from .logger import setup_logger

# logger = setup_logger()


__all__ = [
    'setup_logger'
]