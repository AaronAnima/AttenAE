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
from random import shuffle
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


def save_dataset(images, labels):
    for image, label in zip(images, labels):
        label = label.numpy()
        label = str(label, encoding="utf-8")
        image = tf.convert_to_tensor([image])
        if label in dic:
            dic[label] += 1
        else:
            dic[label] = 1
            tl.files.exists_or_mkdir('dataset/' + str(label))  # samples path
        tl.visualize.save_images(image.numpy(), [1, 1],
                                 'dataset/' + str(label) + '/' + str(label) + '_{}.jpg'.format(dic[label]))


def get_attention_eles(tk, images_notexture, images_aug, E_v, E_q):
    v_k = get_v(images_aug, E_v)
    tq = get_q(images_notexture, E_q)
    _t_v = tf.reshape(images_aug, [images_aug.shape[0], -1, images_aug.shape[3]])
    tv = tf.keras.layers.Attention()([tk, _t_v, v_k])
    return tq, tv


def get_test_imgs(num):
    test_imgs_path = tl.files.load_file_list(path='./test', regx='.*.jpg', keep_prefix=True, printable=False)
    shuffle(test_imgs_path)
    images = []
    for i in range(num):
        image = test_imgs_path[i].encode('utf-8')
        image = tf.io.read_file(image)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)
        if i == 0:
            images = image
            images = tf.convert_to_tensor(images)
        else:
            images = tf.concat([images, image], axis=0)
    return images


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
        images_notexture = aug2(images)
        epoch_num = step // n_step_epoch
        '''
        # log = " ** new learning rate: %f (for GAN)" % (lr_v.tolist()[0])
        # print(log)
        # '''

        # ipdb.set_trace()
        with tf.GradientTape(persistent=True) as tape:
            if step == 0:
                # tk will only be initialized once
                init_val = np.random.normal(loc=0.0, scale=0.02, size=[1, flags.img_size_h,  flags.img_size_h,
                                                                       flags.semantic_ch]).astype(np.float32)
                k_var = tf.Variable(init_val)

            tk = tf.tile(k_var, [flags.batch_size_train, 1, 1, 1])
            tk = tf.reshape(tk, [flags.batch_size_train, -1, tk.shape[3]])
            # v_k = get_v(images_aug, E_v)
            # tq = get_q(images_notexture, E_q)
            # _t_v = tf.reshape(images_aug, [flags.batch_size_train, -1, images_aug.shape[3]])
            # tv = tf.keras.layers.Attention()([tk, _t_v, v_k])
            tq, tv = get_attention_eles(tk, images_notexture, images_aug, E_v, E_q)
            atten_res = tf.keras.layers.Attention()([tq, tv, tk])
            recon = tf.reshape(atten_res, [flags.batch_size_train, flags.img_size_h, flags.img_size_w, 3])
            recon_loss = (tl.cost.absolute_difference_error(recon, images, is_mean=True))

            E_loss_total = recon_loss

        # Updating Encoder
        total_weights = E_q.trainable_weights + E_v.trainable_weights + [k_var]

        grad = tape.gradient(E_loss_total, total_weights)
        optimizer.apply_gradients(zip(grad, total_weights))
        # grad = tape.gradient(E_loss_total, tk)
        # e_optimizer.apply_gradients(zip(grad, tk))

        # basic
        if np.mod(step, flags.show_freq) == 0 and step != 0:
            print("Epoch: [{}/{}] [{}/{}] recon_loss: {:.5f}".format
                  (epoch_num, n_epoch, step, n_step_epoch, recon_loss))

        if np.mod(step, n_step_epoch) == 0 and step != 0:
            E_q.save_weights('./checkpoint/E_q.npz')
            E_v.save_weights('./checkpoint/E_v.npz')

        if np.mod(step, flags.eval_step) == 0 :
            E_q.eval()
            E_v.eval()
            # get a random batch of test image
            images = get_test_imgs(8)
            images = tf.convert_to_tensor(images)
            images1 = images[0:4]
            images2 = images[4:8]

            images_aug1 = images_aug[0:4]
            images_aug2 = images_aug[4:8]
            images_notexture1 = images_notexture[0:4]
            images_notexture2 = images_notexture[4:8]
            tk = tf.tile(k_var, [4, 1, 1, 1])
            tk = tf.reshape(tk, [4, -1, tk.shape[-1]])

            tq1, tv1 = get_attention_eles(tk, images_notexture1, images_aug1, E_v, E_q)
            tq2, tv2 = get_attention_eles(tk, images_notexture2, images_aug2, E_v, E_q)

            atten_res1 = tf.keras.layers.Attention()([tq1, tv1, tk])
            recon1 = tf.reshape(atten_res1, images1.shape)
            atten_res2 = tf.keras.layers.Attention()([tq2, tv2, tk])
            recon2 = tf.reshape(atten_res2, images2.shape)

            atten_res1_fake = tf.keras.layers.Attention()([tq2, tv1, tk])
            fake1 = tf.reshape(atten_res1_fake, images1.shape)
            atten_res2_fake = tf.keras.layers.Attention()([tq1, tv2, tk])
            fake2 = tf.reshape(atten_res2_fake, images2.shape)

            texture1 = tf.reshape(tv1, [-1, flags.img_size_h, flags.img_size_h, 3])
            texture2 = tf.reshape(tv2, [-1, flags.img_size_h, flags.img_size_h, 3])

            fakeAB = tf.concat([images1, fake1, images2, fake2], axis=0)
            reconAB = tf.concat([images1, recon1, images2, recon2], axis=0)
            textureAB = tf.concat([images1, texture1, images2, texture2], axis=0)
            augAB = tf.concat([images1, images_aug1, images2, images_aug2], axis=0)


            tl.visualize.save_images(fakeAB.numpy(), [4, 4],
                                     '{}/fakeAB{:02d}_{:04d}.png'.format(flags.sample_dir, step // n_step_epoch, step))
            tl.visualize.save_images(reconAB.numpy(), [4, 4],
                                     '{}/reconAB{:02d}_{:04d}.png'.format(flags.sample_dir, step // n_step_epoch, step))
            tl.visualize.save_images(textureAB.numpy(), [4, 4],
                                     '{}/textureAB{:02d}_{:04d}.png'.format(flags.sample_dir, step // n_step_epoch, step))
            tl.visualize.save_images(augAB.numpy(), [4, 4],
                                     '{}/augAB{:02d}_{:04d}.png'.format(flags.sample_dir, step // n_step_epoch, step))
            E_q.train()
            E_v.train()

        del tape


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='DWGAN', help='train or eval')
    parser.add_argument('--is_continue', type=bool, default=False, help='load weights from checkpoints?')
    args = parser.parse_args()
    train_GE(con=args.is_continue)
    # train_Gz()
