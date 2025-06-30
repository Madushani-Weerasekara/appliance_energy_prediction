# src/pipeline/logger.py
import logging
import os

def setup_logger(log_file='logs/pipeline.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=logging.INFO
    )
    return logging.getLogger(__name__)
