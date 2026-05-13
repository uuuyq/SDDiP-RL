import logging


def get_logger(file_name):
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger("bundle_fast")
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler
    if not logger.handlers:
        # 文件输出（追加模式）
        file_handler = logging.FileHandler(file_name, mode='a', encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)

    return logger