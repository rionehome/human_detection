import os
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def convert_normal_to_unit(image: np.ndarray):
    if np.min(image) < 0:
        image = (image + 1) / 2
    if not np.max(image) >= 255:
        image = image * 255
    if image.shape[-1] == 1:
        image = image.reshape(image.shape[0], image.shape[1])
    image = image.astype('uint8')
    return image


def generate_color_pallet(id: int):
    if id == 0:
        return [0, 0, 0]
    base_pallet = (np.asarray([256, 256, 256]) / ((abs((id - 1)) // 7) + 1)).astype(int) - 1
    return list(np.asarray(base_pallet) * np.asarray(list(format(((7 - id) % 7) + 1, 'b').zfill(3))).astype(int))


def convert_index_to_color(index_array: np.ndarray):
    color_image = np.zeros((index_array.shape[0], index_array.shape[1], 3), np.uint8)
    index_array = index_array[:, :, np.newaxis]
    for pallet_id in np.unique(index_array):
        color_image = np.where(index_array == pallet_id, generate_color_pallet(pallet_id), color_image)
    return color_image


def save_history(history, dir_path: str):
    print(history.history)
    acc = []
    loss = []
    val_acc = []
    val_loss = []

    for key in dict(history.history).keys():
        if 'loss' in key:
            if 'val' in key:
                val_loss = history.history[key]
            else:
                loss = history.history[key]
        else:
            if 'val' in key:
                val_acc = history.history[key]
            else:
                acc = history.history[key]

    plt.plot(acc, label='training acc')
    plt.plot(val_acc, label='training val_acc')
    plt.legend()
    plt.savefig(os.path.join(dir_path, "acc_history.png"))

    plt.clf()

    plt.plot(loss, label='training loss')
    plt.plot(val_loss, label='training val_loss')
    plt.legend()

    plt.savefig(os.path.join(dir_path, "loss_history.png"))


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
        if len(images.shape) == 2:
            display_img = convert_normal_to_unit(images)
            # gray
            if save_dir is None:
                plt.imshow(display_img, cmap="gray")
            else:
                plt.imsave(os.path.join(save_path, "{}.png".format(num_exist_files + images_index)), display_img,
                           cmap="gray")
            plt.show()
        elif len(images.shape) == 3:
            display_img = convert_normal_to_unit(images)
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
                    tile[i // tile_length, i % tile_length] = convert_normal_to_unit(images[i])
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
                    tile[i // tile_length, i % tile_length] = convert_normal_to_unit(images[i])
                display_img = cv2.vconcat([cv2.hconcat(h) for h in tile])
                if save_dir is None:
                    plt.imshow(display_img)
                else:
                    plt.imsave(os.path.join(save_path, "{}.png".format(num_exist_files + images_index)), display_img)
            plt.show()
        else:
            print("次元多し")
