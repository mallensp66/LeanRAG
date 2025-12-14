import os
import time
import logging
from logging.handlers import RotatingFileHandler
import sys

# If color support is required, it is recommended to use colorama to ensure that colors are displayed correctly also on Windows
try:
    import colorama
    from colorama import Fore, Style
    colorama.init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add color to log messages based on log level.
    INFO: green
    WARNING: yellow
    ERROR: red
    """
    # Define colormap
    COLOR_MAP = {
        logging.DEBUG: Fore.CYAN if COLORAMA_AVAILABLE else '',
        logging.INFO: Fore.GREEN if COLORAMA_AVAILABLE else '',
        logging.WARNING: Fore.YELLOW if COLORAMA_AVAILABLE else '',
        logging.ERROR: Fore.RED if COLORAMA_AVAILABLE else '',
        logging.CRITICAL: Fore.MAGENTA if COLORAMA_AVAILABLE else '',
    }

    RESET = Style.RESET_ALL if COLORAMA_AVAILABLE else ''

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelno, '')
        message = super().format(record)
        if color:
            message = f"{color}{message}{self.RESET}"
        return message

def setup_logger(logger_name, level=logging.INFO):
    '''
    Initialize and return a logger.

    This function ensures that all loggers share the same handler, including:
    - Timestamp based log files
    - latest.log file
    - Console output (with color)

    parameter:
    - logger_name: name of the logger
    - level: log level, default is logging.INFO

    return:
    - logger: configured logger
    '''
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = True  # Make sure logs are delivered to the root logger

    # Check if the root logger has a handler configured
    if not logging.getLogger().hasHandlers():
        # Get the path to the main script
        main_script = sys.argv[0]
        if not main_script:
            # In some environments (such as interactive interpreters), sys.argv[0] may be empty
            main_script_dir = os.getcwd()
            main_script_name = 'unknown'
        else:
            main_script_path = os.path.abspath(main_script)
            main_script_dir = os.path.dirname(main_script_path)
            main_script_name = os.path.splitext(os.path.basename(main_script_path))[0]

        # Create log directory: {main script directory}/logs/{main script name}/
        log_dir = os.path.join('logs', main_script_name)
        os.makedirs(log_dir, exist_ok=True)

        # Generate log file names based on timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        timestamp_log_file = os.path.join(log_dir, f'{timestamp}.log')

        # Define latest.log file path
        latest_log_file = os.path.join(log_dir, 'latest.log')

        # Define log format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Define a colored log formatter
        colored_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Handler 1: Timestamp based log file (with rotation)
        timestamp_handler = RotatingFileHandler(
            timestamp_log_file,
            maxBytes=1000*1024*1024,  # 1000 MB
            backupCount=5,
            encoding='utf-8'
        )
        timestamp_handler.setFormatter(formatter)
        timestamp_handler.setLevel(level)
        logging.getLogger().addHandler(timestamp_handler)

        # Handler 2: latest.log file (overwrite mode)
        latest_handler = logging.FileHandler(
            latest_log_file,
            mode='w',                # Overwrite mode, overwrite files every time you run
            encoding='utf-8'
        )
        latest_handler.setFormatter(formatter)
        latest_handler.setLevel(level)
        logging.getLogger().addHandler(latest_handler)

        # Handler 3: Console output (with color)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(colored_formatter)
        console_handler.setLevel(level)
        logging.getLogger().addHandler(console_handler)

    return logger


class TqdmToLogger:
    '''
    Custom TqdmToLogger class
    Usage examples
    tqdm_out = TqdmToLogger(logger)

    Progress bar output redirected to log
    for step in tqdm(range(total_steps), desc="xxx", file=tqdm_out):
    '''
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buf = ''

    def write(self, buf):
        # Remove extra whitespace characters and record non-empty progress information
        self.buf = buf.strip()
        if self.buf:
            self.logger.log(self.level, self.buf)

    def flush(self):
        pass


def test_setup_setup_logger():
    # Usage
    logger = setup_logger('tool_test')
    logger.info('This is a log info')


if __name__ == "__main__":
    test_setup_setup_logger()
