import os
import pickle
from math import floor

import cv2
import numpy as np
import tensorflow as tf


def tf_datasets_from_dir(src_dir="datasets/", nrows=3, ncols=6, img_size=(60, 200), batch_size=64,
                         val_fraction=0.15, test_fraction=0.10):
    big_dataset = []
    all_imgs = []
    all_labels = []
    dataset_list = sorted(os.listdir(src_dir))
    for filename in dataset_list:
        dataset_path = os.path.join(src_dir, filename)
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        big_dataset += dataset
    print("{} Records in big dataset".format(len(big_dataset)))

    def coords_to_index(cx, cy):
        cx = min((cx + 1) / 2, 0.99)
        cy = min((cy + 1) / 2, 0.99)
        col = floor(cx * ncols)
        row = floor(cy * nrows)
        index = row * ncols + col
        return index

    def pairwise_shuffle(arr1, arr2):
        shuffled_indices = np.random.permutation(arr1.shape[0])
        return arr1[shuffled_indices], arr2[shuffled_indices]

    for item in big_dataset:
        all_imgs.append(cv2.resize(item["eyes"], img_size))
        all_labels.append(coords_to_index(item["x"], item["y"]))

    total_examples = len(all_imgs)
    all_imgs = np.asarray(all_imgs)
    all_labels = np.asarray(all_labels)
    all_imgs, all_labels = pairwise_shuffle(all_imgs, all_labels)
    split_test = slice(0, floor(total_examples * test_fraction))
    split_val = slice(
        floor(total_examples * test_fraction),
        floor(total_examples * test_fraction) + floor(total_examples * val_fraction),
    )
    split_train = slice(
        floor(total_examples * test_fraction) + floor(total_examples * val_fraction),
        total_examples,
    )
    tst_imgs = all_imgs[split_test]
    val_imgs = all_imgs[split_val]
    trn_imgs = all_imgs[split_train]
    tst_labels = all_labels[split_test]
    val_labels = all_labels[split_val]
    trn_labels = all_labels[split_train]
    tst_dataset = tf.data.Dataset.from_tensor_slices((tst_imgs, tst_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_imgs, val_labels))
    trn_dataset = tf.data.Dataset.from_tensor_slices((trn_imgs, trn_labels))
    tst_len = len(tst_imgs)
    val_len = len(val_imgs)
    trn_len = len(trn_imgs)
    tst_dataset = tst_dataset.shuffle(tst_len)
    val_dataset = val_dataset.shuffle(val_len)
    trn_dataset = trn_dataset.shuffle(trn_len)
    tst_dataset = tst_dataset.batch(tst_len)
    val_dataset = val_dataset.batch(val_len)
    trn_dataset = trn_dataset.batch(batch_size)
    return ncols * nrows, trn_dataset, val_dataset, tst_dataset
