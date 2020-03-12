from glob import glob
import os

import cv2
from scipy.stats import entropy
from tqdm import tqdm

OBJECT_DATASET_PATH = "./data/JPEGImages"
OUTPUT_PATH = "./data/object"
KERNEL_SIZE = 96


def main():
    filenames = glob("{}/*".format(OBJECT_DATASET_PATH))
    for filename in tqdm(filenames):
        object_image = cv2.imread(filename)
        for row in range(0, object_image.shape[0], KERNEL_SIZE * 2):
            for col in range(0, object_image.shape[1], KERNEL_SIZE * 2):
                target_image = object_image[row:row + KERNEL_SIZE, col:col + KERNEL_SIZE]
                if entropy(cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY).flatten(), base=2) > 13:
                    cv2.imwrite("{}/{}.png".format(OUTPUT_PATH, len(os.listdir(OUTPUT_PATH))), target_image)


if __name__ == '__main__':
    main()
