import logging
def setup_logger(name: str = "pipeline", log_file: str = None,
                 console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Setup logger for the pipeline.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # capture all, handlers filter further

    # prevent adding handlers multiple times if setup_logger called again
    if logger.hasHandlers():
        logger.handlers.clear()

    log_format = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    # File handler (if requested)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(file_level)
        fh.setFormatter(log_format)
        logger.addHandler(fh)

    return logger