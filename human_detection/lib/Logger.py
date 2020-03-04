import os
import joblib
import shutil


class Logger:
    def __init__(self, path: str):
        self.count_files = 0
        self.path = path

        if os.path.exists(path):  # ディレクトリがあれば
            shutil.rmtree(path)
        os.makedirs(path)

    def save(self, save_data: list):
        joblib.dump(save_data, os.path.join(self.path, "{}.npy".format(self.count_files + 1)), compress=True)
        self.count_files = self.count_files + 1
