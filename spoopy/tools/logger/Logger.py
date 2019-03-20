import os
from datetime import datetime

from tools.file_utils import file_helper

dirname = os.path.dirname(__file__)


class Logger():
    def __init__(self, logs_path, verbose=True):
        file_helper.guarantee_path_preconditions(logs_path)
        file_name = os.path.join(logs_path, 'log_' + str(datetime.now()) + '.txt')
        self.logs_path = file_name
        self.verbose = verbose

    def process_log(self, type, message):
        file = open(self.logs_path, "a")
        msg = self.create_message(type, message)
        file.write(msg + '\n')
        file.close()

        if self.verbose:
            print(msg)

    def log(self, message):
        self.process_log('I', message)

    def logE(self, message):
        self.process_log('E', message)

    def create_message(self, log_type, message):
        current_time = datetime.now()
        message = "[%s] [%s] [%s]" % (log_type, current_time, message)
        return message


def test_logger():
    file_name = os.path.join(dirname, 'log_results.txt')
    l = Logger(file_name)
    l.log('Log teste')
    l.logE('Erro teste')

    print('done')


if __name__ == '__main__':
    test_logger()
