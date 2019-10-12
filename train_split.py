import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags
from data import load_openimage, partialdic
from models import get_Eq, get_Ek, get_Ev, get_img_D
import random
import argparse
import math
import scipy.stats as stats
import tensorflow_probability as tfp

import ipdb
import cv2

# import sys
# f = open('a.log', 'a')
# sys.stdout = f
# sys.stderr = f # redirect std err, if necessary

E_q = get_Eq([None, flags.img_size_h, flags.img_size_w, 1])
E_v = get_Ev([None, flags.img_size_h, flags.img_size_w, flags.c_dim])

def aug2(images):
    images = images.numpy()
    results = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转化为灰度图像

        # Sobel边缘检测
        sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)  # x方向的梯度
        sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)  # y方向的梯度

        sobelX = np.uint8(np.absolute(sobelX))  # x方向梯度的绝对值
        sobelY = np.uint8(np.absolute(sobelY))  # y方向梯度的绝对值

        image = cv2.bitwise_or(sobelX, sobelY)
        results.append(image)
    results = np.array(results)
    results = tf.convert_to_tensor(results, tf.float32)
    return tf.reshape(results, [-1, flags.img_size_h, flags.img_size_h, 1])


def KStest(real_z, fake_z):
    p_list = []
    for i in range(flags.batch_size_train):
        _, tmp_p = stats.ks_2samp(fake_z[i], real_z[i])
        p_list.append(tmp_p)
    return np.min(p_list), np.mean(p_list)


def get_q(batch_imgs, E_q):
    tq = E_q(batch_imgs)
    tq = tf.reshape(tq, [batch_imgs.shape[0], -1, tq.shape[3]])
    return tq


def get_v(batch_imgs, E_v):
    tv = E_v(batch_imgs)
    tv = tf.reshape(tv, [batch_imgs.shape[0], -1, tv.shape[3]])
    return tv

dic = {}
def save_dataset(images, labels):

    for image, label in zip(images, labels):
        label = label.numpy()
        label = str(label, encoding="utf-8")
        image = tf.convert_to_tensor([image])
        if label in dic:
            dic[label] += 1
        else:
            dic[label] = 1
            tl.files.exists_or_mkdir('dataset1/' + str(label))  # samples path
        tl.visualize.save_images(image.numpy(), [1, 1],
                                 'dataset1/' + str(label) + '/' + str(label) + '_{}.jpg'.format(dic[label]))
        # tl.visualize.save_images(image_aug.numpy(), [1, 1],
        #                          'dataset1/' + str(label) + '/' + str(label) + '_{}_aug.jpg'.format(dic[label]))


def get_attention_eles(tk, images_notexture, images_aug, E_v, E_q):
    v_k = get_v(images_aug, E_v)
    tq = get_q(images_notexture, E_q)
    _t_v = tf.reshape(images_aug, [images_aug.shape[0], -1, images_aug.shape[3]])
    tv = tf.keras.layers.Attention()([tk, _t_v, v_k])
    return tq, tv


def train_GE(con=False):
    # dataset, len_dataset = get_dataset_train()
    dataset, len_dataset = load_openimage()
    # if con:
    #     E_q.load_weights('./checkpoint/E_q.npz')
    #     E_k.load_weights('./checkpoint/E_k.npz')
    #     E_v.load_weights('./checkpoint/E_v.npz')
    # E_q.load_weights('./checkpoint/E_q.npz')
    # E_k.load_weights('./checkpoint/E_k.npz')
    # E_v.load_weights('./checkpoint/E_v.npz')
    # D1.load_weights('./checkpoint/D1.npz')
    # D2.load_weights('./checkpoint/D2.npz')

    E_q.train()
    E_v.train()

    n_step_epoch = int(len_dataset // flags.batch_size_train)
    n_epoch = int(flags.step_num // n_step_epoch)

    lr_E = flags.lr_E

    optimizer = tf.optimizers.Adam(lr_E, beta_1=flags.beta1, beta_2=flags.beta2)
    k_var = None
    tk = None
    for step, batches in enumerate(dataset):
        images = batches[0]
        images_aug = batches[1]
        labels = batches[2]
        images_notexture = aug2(images)
        epoch_num = step // n_step_epoch
        if epoch_num > 0:
            break
        save_dataset(images, labels)
        print(step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='DWGAN', help='train or eval')
    parser.add_argument('--is_continue', type=bool, default=False, help='load weights from checkpoints?')
    args = parser.parse_args()
    train_GE(con=args.is_continue)
    # train_Gz()
