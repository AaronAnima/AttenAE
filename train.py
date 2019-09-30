import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags
from data import get_dataset_train, get_Y2X_train
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
E_k = get_Ek([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
E_v = get_Ev([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
D1 = get_img_D([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
D2 = get_img_D([None, flags.img_size_h, flags.img_size_w, flags.c_dim])

def edge_detect(images):
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
    return results

def KStest(real_z, fake_z):
    p_list = []
    for i in range(flags.batch_size_train):
        _, tmp_p = stats.ks_2samp(fake_z[i], real_z[i])
        p_list.append(tmp_p)
    return np.min(p_list), np.mean(p_list)

def get_atten_ele(batch_imgs, E_q, E_k, E_v):
    noedge_imgs = edge_detect(batch_imgs)
    noedge_imgs = tf.reshape(noedge_imgs, [-1, 64, 64, 1])
    tq = E_q(noedge_imgs)
    tq = tf.reshape(tq, [batch_imgs.shape[0], -1, tq.shape[3]])
    tv = E_v(batch_imgs)
    tv = tf.reshape(tv, [batch_imgs.shape[0], -1, tv.shape[3]])
    tk = E_k(batch_imgs)
    tk = tf.reshape(tk, [batch_imgs.shape[0], -1, tk.shape[3]])
    return tq, tv, tk


def train_GE(con=False):
    # dataset, len_dataset = get_dataset_train()
    dataset, len_dataset = get_Y2X_train()
    len_dataset = flags.len_dataset
    num_tiles = int(np.sqrt(flags.sample_size))
    print(con)
    # if con:
    #     E_q.load_weights('./checkpoint/E_q.npz')
    #     E_k.load_weights('./checkpoint/E_k.npz')
    #     E_v.load_weights('./checkpoint/E_v.npz')
    E_q.load_weights('./checkpoint/E_q.npz')
    E_k.load_weights('./checkpoint/E_k.npz')
    E_v.load_weights('./checkpoint/E_v.npz')
    D1.load_weights('./checkpoint/D1.npz')
    D2.load_weights('./checkpoint/D2.npz')

    E_q.train()
    E_k.train()
    E_v.train()
    D1.train()
    D2.train()
    # print(E_q.config)

    n_step_epoch = int(len_dataset // flags.batch_size_train)
    n_epoch = int(flags.step_num // n_step_epoch)

    lr_E = flags.lr_E
    lr_D = flags.lr_D

    e_optimizer = tf.optimizers.Adam(lr_E, beta_1=flags.beta1, beta_2=flags.beta2)
    d_optimizer = tf.optimizers.Adam(lr_D, beta_1=flags.beta1, beta_2=flags.beta2)
    for step, Y_and_X in enumerate(dataset):
        epoch_num = step // n_step_epoch
        batch_imgs1 = Y_and_X[0]  # (1, 256, 256, 3)
        batch_imgs2 = Y_and_X[1]  # (1, 256, 256, 3)
        if step == 0:
            sample_images1 = batch_imgs1
            sample_images2 = batch_imgs2
            tl.visualize.save_images(sample_images1.numpy(), [num_tiles, num_tiles],
                                     '{}/_sample1.png'.format(flags.sample_dir))
            tl.visualize.save_images(sample_images2.numpy(), [num_tiles, num_tiles],
                                     '{}/_sample2.png'.format(flags.sample_dir))
        '''
        log = " ** new learning rate: %f (for GAN)" % (lr_v.tolist()[0])
        print(log)
        '''

        # ipdb.set_trace()
        with tf.GradientTape(persistent=True) as tape:
            tq1, tv1, tk1 = get_atten_ele(batch_imgs1, E_q, E_k, E_v)
            tq2, tv2, tk2 = get_atten_ele(batch_imgs2, E_q, E_k, E_v)
            # print(tq1.shape)
            # print(tv1.shape)
            # print(tk1.shape)
            # input()
            atten_res2 = tf.keras.layers.Attention()([tq1, tv2, tk1])
            atten_res1 = tf.keras.layers.Attention()([tq2, tv1, tk2])
            recon1 = tf.keras.layers.Attention()([tq1, tv1, tk1])
            recon2 = tf.keras.layers.Attention()([tq2, tv2, tk2])

            recon1 = tf.reshape(recon1, [flags.batch_size_train, 64, 64, 3])
            recon2 = tf.reshape(recon2, [flags.batch_size_train, 64, 64, 3])
            fake2 =tf.reshape(atten_res2, [flags.batch_size_train, 64, 64, 3])
            fake1 =tf.reshape(atten_res1, [flags.batch_size_train, 64, 64, 3])

            fake1_logits = D1(fake1)
            fake2_logits = D2(fake2)
            real1_logits = D1(batch_imgs1)
            real2_logits = D2(batch_imgs2)

            recon_loss = 25 * (tl.cost.absolute_difference_error(recon1, batch_imgs1, is_mean=True) + \
                         tl.cost.absolute_difference_error(recon2, batch_imgs2, is_mean=True))

            E_loss_adv = tl.cost.sigmoid_cross_entropy(fake1_logits, tf.ones_like(fake1_logits)) + \
                           tl.cost.sigmoid_cross_entropy(fake2_logits, tf.ones_like(fake2_logits))

            D_loss_adv = tl.cost.sigmoid_cross_entropy(fake1_logits, tf.zeros_like(fake1_logits)) + \
                           tl.cost.sigmoid_cross_entropy(fake2_logits, tf.zeros_like(fake2_logits)) + \
                           tl.cost.sigmoid_cross_entropy(real1_logits, tf.ones_like(real1_logits)) + \
                           tl.cost.sigmoid_cross_entropy(real2_logits, tf.ones_like(real2_logits))

        # Updating Encoder
        E_trainable_weights = E_q.trainable_weights + E_v.trainable_weights + E_k.trainable_weights
        D_trainable_weights = D1.trainable_weights + D2.trainable_weights
        grad = tape.gradient(E_loss_adv , E_trainable_weights)
        e_optimizer.apply_gradients(zip(grad, E_trainable_weights))

        grad = tape.gradient(recon_loss, E_trainable_weights)
        e_optimizer.apply_gradients(zip(grad, E_trainable_weights))

        grad = tape.gradient(D_loss_adv, D_trainable_weights)
        d_optimizer.apply_gradients(zip(grad, D_trainable_weights))

        # basic
        if np.mod(step, flags.show_freq) == 0 and step != 0:
            print("Epoch: [{}/{}] [{}/{}] recon_loss: {:.5f}, E_loss_adv: {:.5f}, D_loss_adv: {:.5f}".format
                  (epoch_num, n_epoch, step, n_step_epoch, recon_loss, E_loss_adv, D_loss_adv))

        if np.mod(step, n_step_epoch) == 0 and step != 0:
            E_q.save_weights('./checkpoint/E_q.npz')
            E_k.save_weights('./checkpoint/E_k.npz')
            E_v.save_weights('./checkpoint/E_v.npz')
            D1.save_weights('./checkpoint/D1.npz')
            D2.save_weights('./checkpoint/D2.npz')
            # G.train()
        # if np.mod(step, 1) == 0:
        if np.mod(step, flags.eval_step) == 0 and step != 0:
            realAfakeA = tf.concat([batch_imgs1[0:32], fake1[0:32]], 0)
            # realAfakeA = tf.concat([tf.split(0, 2, batch_imgs1)[0], tf.split(0, 2, fake1)[0]], 0)
            # realBfakeB = tf.concat([tf.split(0, 2, batch_imgs2)[0], tf.split(0, 2, fake2)[0]], 0)
            realBfakeB = tf.concat([batch_imgs2[0:32], fake2[0:32]], 0)
            textureA = tf.reshape(tv1, [-1, 16, 16, 3])
            textureB = tf.reshape(tv2, [-1, 16, 16, 3])
            # textureAB = tf.concat([tf.split(0, 2, textureA)[0], tf.split(0, 2, textureB)[0]], 0)
            textureAB = tf.concat([textureA[0:32], textureB[0:32]], 0)
            # reconAB = tf.concat([tf.split(0, 2, recon1)[0], tf.split(0, 2, recon2)[0]], 0)
            reconAB = tf.concat([recon1[0:32], recon2[0:32]], 0)

            tl.visualize.save_images(realAfakeA.numpy(), [8, 8],
                                     '{}/realAfakeA{:02d}_{:04d}.png'.format(flags.sample_dir, step // n_step_epoch, step))
            tl.visualize.save_images(realBfakeB.numpy(), [8, 8],
                                     '{}/realBfakeB{:02d}_{:04d}.png'.format(flags.sample_dir, step // n_step_epoch, step))
            tl.visualize.save_images(textureAB.numpy(), [8, 8],
                                     '{}/textureAB{:02d}_{:04d}.png'.format(flags.sample_dir, step // n_step_epoch, step))
            tl.visualize.save_images(reconAB.numpy(), [8, 8],
                                     '{}/reconAB{:02d}_{:04d}.png'.format(flags.sample_dir, step // n_step_epoch, step))

        del tape


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='DWGAN', help='train or eval')
    parser.add_argument('--is_continue', type=bool, default=False, help='load weights from checkpoints?')
    args = parser.parse_args()
    train_GE(con=args.is_continue)
    train_Gz()
