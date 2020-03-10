import os
import joblib
import shutil


class Logger:
    def __init__(self, path: str):
        self.count_files = 0
        self.path = path

        if not os.path.exists(path):  # ディレクトリがなければ
            os.makedirs(path)

    def clear(self):
        if os.path.exists(self.path):  # ディレクトリがあれば
            shutil.rmtree(self.path)
        os.makedirs(self.path)

    def save(self, save_data: list):
        joblib.dump(save_data, os.path.join(self.path, "{}.npy".format(self.count_files + 1)), compress=True)
        self.count_files = self.count_files + 1
