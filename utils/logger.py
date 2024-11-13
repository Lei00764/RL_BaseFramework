"""
@File    :   logger.py
@Time    :   2024/11/13 15:38:56
@Author  :   Xiang Lei
@Email   :   xiang.lei.se@foxmail.com
@Version :   1.0
@Desc    :   None
"""

import logging
import os
from typing import Optional

class Logger:
    @staticmethod
    def setup_logger(
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO
    ) -> logging.Logger:
        """设置logger"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger