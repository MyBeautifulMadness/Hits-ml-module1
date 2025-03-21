import logging
import os

LOG_PATH = "data/log_file.log"
os.makedirs("data", exist_ok=True)

class LoggerSingleton:
    _instance = None

    @staticmethod
    def get_instance():
        if LoggerSingleton._instance is None:
            LoggerSingleton()
        return LoggerSingleton._instance

    def __init__(self):
        if LoggerSingleton._instance is not None:
            raise Exception("Используйте get_instance()")
        logging.basicConfig(
            filename=LOG_PATH,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger("My_Classifier_Model")
        LoggerSingleton._instance = self

logger = LoggerSingleton.get_instance().logger