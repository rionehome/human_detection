import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def normalize_image(image: np.ndarray, image_size: int):
    image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
    return image


def compare(img, img2):
    img = (img - img.mean()) / img.std()
    img2 = (img2 - img2.mean()) / img2.std()
    return np.mean(np.abs(img - img2))


def to_quaternion_rad(w, z):
    return math.acos(w) * 2 * np.sign(z)


def calc_real_position(x, y, z, pos_x, pos_y, pos_radian):
    relative_x = z
    relative_y = -x
    relative_z = y
    result_x = (relative_x * math.cos(pos_radian) - relative_y * math.sin(pos_radian)) + pos_x
    result_y = (relative_x * math.sin(pos_radian) + relative_y * math.cos(pos_radian)) + pos_y
    result_z = relative_z
    return result_x, result_y, result_z


def convert_unit_image(image: np.ndarray):
    if np.min(image) < 0:
        image = (image + 1) / 2
    if np.max(image) > 255:
        image = image * 255
    if image.shape[-1] == 1:
        image = image.reshape(image.shape[0], image.shape[1])
    image = image.astype('uint8')
    return image


def show_image_tile(images_array: list, save_dir=None):
    num_exist_files = 0
    save_path = None
    if save_dir is not None:
        save_path = os.path.join(save_dir, "output_images")
        if not os.path.exists(save_path):  # ディレクトリ がなければ
            os.makedirs(save_path)
        num_exist_files = len(os.listdir(save_path))

    for images_index, images in enumerate(images_array):
        images = np.asarray(images)
        if len(images.shape) == 3:
            display_img = convert_unit_image(images)
            if display_img.shape[-1] == 3:
                # color
                if save_dir is None:
                    plt.imshow(display_img)
                else:
                    plt.imsave(os.path.join(save_path, "{}.png".format(num_exist_files + images_index)), display_img)
            else:
                # gray
                if save_dir is None:
                    plt.imshow(display_img, cmap="gray")
                else:
                    plt.imsave(os.path.join(save_path, "{}.png".format(num_exist_files + images_index)), display_img,
                               cmap="gray")
            plt.show()
        elif len(images.shape) == 4:
            tile_length = int(math.sqrt(images.shape[0])) + 1
            if images.shape[-1] == 1:
                # gray
                tile = np.full((tile_length, tile_length, images.shape[1], images.shape[2]), 255)
                for i in range(images.shape[0]):
                    tile[i // tile_length, i % tile_length] = convert_unit_image(images[i])
                display_img = cv2.vconcat([cv2.hconcat(h) for h in tile])
                if save_dir is None:
                    plt.imshow(display_img, cmap="gray")
                else:
                    plt.imsave(os.path.join(save_path, "{}.png".format(num_exist_files + images_index)), display_img,
                               cmap="gray")
            else:
                # color
                tile = np.full((tile_length, tile_length, images.shape[1], images.shape[2], images.shape[3]), 255)
                for i in range(images.shape[0]):
                    tile[i // tile_length, i % tile_length] = convert_unit_image(images[i])
                display_img = cv2.vconcat([cv2.hconcat(h) for h in tile])
                if save_dir is None:
                    plt.imshow(display_img)
                else:
                    plt.imsave(os.path.join(save_path, "{}.png".format(num_exist_files + images_index)), display_img)
            plt.show()
        else:
            print("次元多し")


def show_eight(imgs, title):
    select_imgs = [imgs[np.random.choice(len(imgs))] for _ in range(8)]
    _, ax = plt.subplots(2, 4, sharex='col', sharey='row', figsize=(20, 6))
    plt.suptitle(title, size=20)
    for i, img in enumerate(select_imgs):
        ax[i // 4, i % 4].imshow(img)


def show_four(imgs, title):
    # select_imgs = [np.random.choice(imgs) for _ in range(4)]
    select_imgs = [imgs[np.random.choice(len(imgs))] for _ in range(4)]
    _, ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(20, 3))
    plt.suptitle(title, size=20)
    for i, img in enumerate(select_imgs):
        ax[i].imshow(img)
