import datetime
import threading
import os


class Logger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, args=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Logger, cls).__new__(cls)
                    cls._instance._initialize(args)
        return cls._instance

    def _initialize(self, args=None):
        self.log_file = None
        # if args.log_file is not None:
        #     log_dir = os.path.dirname(args.log_file)
        #     os.makedirs(log_dir, exist_ok=True)
        #     try:
        #         self.log_file = open(args.log_file, 'w')
        #     except Exception as e:
        #         print(f"An error occurred: {e}")

    def __del__(self):
        if self.log_file is not None:
            self.log_file.close()

    def _log(self, level, message):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{timestamp} - {level} - {message}"

        self.log_file.write(log_message + '\n') if self.log_file else print(log_message)

    def info(self, message):
        self._log("INFO", message)

    def warning(self, message):
        self._log("WARNING", message)

    def error(self, message):
        self._log("ERROR", message)

    def debug(self, message):
        self._log("DEBUG", message)


def info(message):
    logger = Logger()
    logger.info(message)


def warning(message):
    logger = Logger()
    logger.warning(message)


def error(message):
    logger = Logger()
    logger.error(message)


def debug(message):
    logger = Logger()
    logger.debug(message)
