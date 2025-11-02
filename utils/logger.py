"""
日志管理
"""

import logging
import os
from datetime import datetime


def setup_logger(name, log_dir, filename=None):
    """
    设置日志记录器
    
    Args:
        name: logger名称
        log_dir: 日志目录（相对或绝对路径）
        filename: 日志文件名（如果为None，自动生成）
    
    Returns:
        logger: 配置好的logger
    """
    # 如果是相对路径，转换为基于当前工作目录的绝对路径
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(os.getcwd(), log_dir)
    
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'training_{timestamp}.log'
    
    log_file = os.path.join(log_dir, filename)
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    logger.handlers = []
    
    # 文件handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger


if __name__ == "__main__":
    # 测试
    logger = setup_logger('test', 'test_logs')
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    print("Logger test completed. Check test_logs/ directory.")
