import logging
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

# for writing logs into the same file
def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
