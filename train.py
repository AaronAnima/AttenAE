import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags
from data import get_dataset_train
from models import get_Eq, get_Ek, get_Ev
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


def train_GE(con=False):
    dataset, len_dataset = get_dataset_train()
    len_dataset = flags.len_dataset
    num_tiles = int(np.sqrt(flags.sample_size))
    print(con)
    if con:
        E_q.load_weights('./checkpoint/E_q.npz')
        E_k.load_weights('./checkpoint/E_k.npz')
        E_v.load_weights('./checkpoint/E_v.npz')
    # E_q.load_weights('./checkpoint/E_q.npz')
    # E_k.load_weights('./checkpoint/E_k.npz')
    # E_v.load_weights('./checkpoint/E_v.npz')

    E_q.train()
    E_k.train()
    E_v.train()

    n_step_epoch = int(len_dataset // flags.batch_size_train)
    n_epoch = int(flags.step_num // n_step_epoch)

    lr_E = flags.lr_E

    e_optimizer = tf.optimizers.Adam(lr_E, beta_1=flags.beta1, beta_2=flags.beta2)
    sample_images = None
    for step, image_labels in enumerate(dataset):
        '''
        log = " ** new learning rate: %f (for GAN)" % (lr_v.tolist()[0])
        print(log)
        '''
        batch_imgs = image_labels[0]
        if step == 0:
            sample_images = batch_imgs
            tl.visualize.save_images(sample_images.numpy(), [num_tiles, num_tiles],
                                     '{}/_sample.png'.format(flags.sample_dir))

        # ipdb.set_trace()

        epoch_num = step // n_step_epoch
        with tf.GradientTape(persistent=True) as tape:
            noedge_imgs = edge_detect(batch_imgs)
            noedge_imgs = tf.reshape(noedge_imgs, [-1, 64, 64, 1])
            tq = E_q(noedge_imgs)
            tq = tf.reshape(tq, [batch_imgs.shape[0], -1, tq.shape[3]])
            tv = E_v(batch_imgs)
            tv = tf.reshape(tv, [batch_imgs.shape[0], -1, tv.shape[3]])
            tk = E_k(batch_imgs)
            tk = tf.reshape(tk, [batch_imgs.shape[0], -1, tk.shape[3]])
            atten_res = tf.keras.layers.Attention()([tq, tv, tk])
            batch_imgs =tf.reshape(batch_imgs, [64, -1, 3])
            recon_loss = tl.cost.absolute_difference_error(atten_res, batch_imgs, is_mean=True)

        # Updating Encoder
        E_trainable_weights = E_q.trainable_weights + E_v.trainable_weights + E_k.trainable_weights
        grad = tape.gradient(recon_loss, E_trainable_weights)
        e_optimizer.apply_gradients(zip(grad, E_trainable_weights))

        # basic
        if np.mod(step, flags.show_freq) == 0 and step != 0:
            print("Epoch: [{}/{}] [{}/{}] recon_loss: {:.5f}".format
                  (epoch_num, n_epoch, step, n_step_epoch, recon_loss))

        if np.mod(step, n_step_epoch) == 0 and step != 0:
            E_q.save_weights('./checkpoint/E_q.npz')
            E_k.save_weights('./checkpoint/E_k.npz')
            E_v.save_weights('./checkpoint/E_v.npz')
            # G.train()
        if np.mod(step, flags.eval_step) == 0 and step != 0:
            E_q.eval()
            E_k.eval()
            E_v.eval()
            noedge_imgs = edge_detect(sample_images)
            noedge_imgs = tf.reshape(noedge_imgs, [-1, 64, 64, 1])
            tq = E_q(noedge_imgs)
            tq = tf.reshape(tq, [sample_images.shape[0], -1, tq.shape[3]])
            tv = E_v(sample_images)
            tv = tf.reshape(tv, [sample_images.shape[0], -1, tv.shape[3]])
            tk = E_k(sample_images)
            tk = tf.reshape(tk, [sample_images.shape[0], -1, tk.shape[3]])
            recon_imgs = tf.keras.layers.Attention()([tq, tv, tk])
            recon_imgs = tf.reshape(recon_imgs, [-1, 64, 64, 3])
            tv = tf.reshape(tv, [-1, 32, 32, 3])
            E_q.train()
            E_k.train()
            E_v.train()
            tl.visualize.save_images(recon_imgs.numpy(), [8, 8],
                                     '{}/recon_{:02d}_{:04d}.png'.format(flags.sample_dir, step // n_step_epoch, step))
            tl.visualize.save_images(tv.numpy(), [8, 8],
                                     '{}/texture{:02d}_{:04d}.png'.format(flags.sample_dir, step // n_step_epoch, step))
            tl.visualize.save_images(noedge_imgs.numpy(), [8, 8],
                                     '{}/notexture{:02d}_{:04d}.png'.format(flags.sample_dir, step // n_step_epoch, step))
        del tape


def train_Gz():
    dataset, len_dataset = get_dataset_train()
    len_dataset = flags.len_dataset

    G.eval()
    E.eval()
    G_z.train()
    D_z.trian()

    n_step_epoch = int(len_dataset // flags.batch_size_train)
    n_epoch = int(flags.step_num // n_step_epoch)

    lr_Dz = flags.lr_Dz
    lr_Gz = flags.lr_Gz

    dz_optimizer = tf.optimizers.Adam(lr_Dz, beta_1=flags.beta1, beta_2=flags.beta2)
    gz_optimizer = tf.optimizers.Adam(lr_Gz, beta_1=flags.beta1, beta_2=flags.beta2)
    for step, image_labels in enumerate(dataset):
        '''
        log = " ** new learning rate: %f (for GAN)" % (lr_v.tolist()[0])
        print(log)
        '''
        batch_imgs = image_labels[0]

        epoch_num = step // n_step_epoch
        with tf.GradientTape(persistent=True) as tape:
            z = np.random.normal(loc=0.0, scale=1, size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            fake_tensor = G_z(z)
            real_tensor = E(batch_imgs)

            fake_tensor_logits = D_z(fake_tensor)
            real_tensor_logits = D_z(real_tensor)

            gz_loss = tl.cost.sigmoid_cross_entropy(fake_tensor_logits, tf.ones_like(fake_tensor_logits))
            dz_loss = tl.cost.sigmoid_cross_entropy(real_tensor_logits, tf.ones_like(real_tensor_logits)) + \
                      tl.cost.sigmoid_cross_entropy(fake_tensor_logits, tf.zeros_like(fake_tensor_logits))
        # Updating Generator
        grad = tape.gradient(gz_loss, G_z.trainable_weights)
        gz_optimizer.apply_gradients(zip(grad, G_z.trainable_weights))
        #
        # Updating D_z & D_h
        grad = tape.gradient(dz_loss, D_z.trainable_weights)
        dz_optimizer.apply_gradients(zip(grad, D_z.trainable_weights))

        # basic
        if np.mod(step, flags.show_freq) == 0 and step != 0:
            print("Epoch: [{}/{}] [{}/{}] d_loss: {:.5f}, g_loss: {:.5f}".format
                  (epoch_num, n_epoch, step, n_step_epoch, e_loss, g_loss))

        if np.mod(step, n_step_epoch) == 0 and step != 0:
            G_z.save_weights('{}/G_z.npz'.format(flags.checkpoint_dir), format='npz')
            D_z.save_weights('{}/D_z.npz'.format(flags.checkpoint_dir), format='npz')
            # G.train()
        if np.mod(step, flags.eval_step) == 0 and step != 0:
            z = np.random.normal(loc=0.0, scale=1, size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            G.eval()
            sample_tensor = G_z(z)
            sample_img = G(sample_tensor)
            G.train()
            tl.visualize.save_images(sample_img.numpy(), [8, 8],
                                     '{}/sample_{:02d}_{:04d}.png'.format(flags.sample_dir,
                                                                         step // n_step_epoch, step))
        del tape


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='DWGAN', help='train or eval')
    parser.add_argument('--is_continue', type=bool, default=False, help='load weights from checkpoints?')
    args = parser.parse_args()
    train_GE(con=args.is_continue)
    train_Gz()
