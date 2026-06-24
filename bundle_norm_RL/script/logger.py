import logging


def get_logger(file_name):
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger("BundleMethod")
    logger.setLevel(logging.DEBUG)

    # 防止重复添加 handler
    if not logger.handlers:
        # 文件输出 - 指定 UTF-8 编码，避免乱码
        file_handler = logging.FileHandler(file_name, mode='w', encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)

    return logger