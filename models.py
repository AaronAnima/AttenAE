import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten, BatchNorm, Concat, GaussianNoise
from config import flags
from utils import SpectralNormConv2d

#
# def get_G(shape_z):    # Dimension of gen filters in first conv layer. [64]
#     # w_init = tf.glorot_normal_initializer()
#     print("for G")
#     ngf = 64
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     gamma_init = tf.random_normal_initializer(1., 0.02)
#     n_extra_layers = flags.n_extra_layers
#     isize = 64
#     cngf, tisize = ngf // 2, 4
#     while tisize != isize:
#         cngf = cngf * 2
#         tisize = tisize * 2
#     ni = Input(shape_z)
#
#
#     # nn = Reshape(shape=[-1, 1, 1, flags.z_dim])(ni)
#     # nn = DeConv2d(cngf, (4, 4), (1, 1), W_init=w_init, b_init=None, padding='VALID')(nn)
#     # nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
#
#     nn = ni
#     csize, cndf = 4, cngf
#     while csize < isize // 2:
#         cngf = cngf // 2
#         nn = DeConv2d(cngf, (4, 4), (2, 2), W_init=w_init, b_init=None)(nn)
#         nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
#         print(nn.shape)
#         csize = csize * 2
#
#     for t in range(n_extra_layers):
#         nn = DeConv2d(cngf, (3, 3), (1, 1), W_init=w_init, b_init=None)(nn)
#         nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
#         print(nn.shape)
#
#     nn = DeConv2d(3, (4, 4), (2, 2), act=tf.nn.tanh, W_init=w_init, b_init=None)(nn)
#     print(nn.shape)
#
#     return tl.models.Model(inputs=ni, outputs=nn)
#
# # E is reverse of G without activation in the output layer
# def get_E(shape):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     gamma_init = tf.random_normal_initializer(1., 0.02)
#     ngf = 64
#     isize = 64
#     n_extra_layers = flags.n_extra_layers
#     print(" for E")
#     ni = Input(shape)
#     nn = Conv2d(ngf, (4, 4), (2, 2), act=None, W_init=w_init, b_init=None)(ni)
#     print(nn.shape)
#     isize = isize // 2
#
#     for t in range(n_extra_layers):
#         nn = Conv2d(ngf, (3, 3), (1, 1), W_init=w_init, b_init=None)(nn)
#         nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
#         print(nn.shape)
#
#     while isize > 4:
#         ngf = ngf * 2
#         nn = Conv2d(ngf, (4, 4), (2, 2), W_init=w_init, b_init=None)(nn)
#         if isize != 8:
#             nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
#         print(nn.shape)
#         isize = isize // 2
#
#     # nn = Conv2d(flags.z_dim, (4, 4), (1, 1), act=None, W_init=w_init, b_init=None, padding='VALID')(nn)
#     # # print(nn.shape)
#     # nn = Reshape(shape=[-1, flags.z_dim])(nn)
#     return tl.models.Model(inputs=ni, outputs=nn)


# def res_block():

# E is reverse of G without activation in the output layer
def get_Ev(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    ngf = 64
    isize = 64
    n_extra_layers = flags.n_extra_layers
    ni = Input(shape)
    nn = ni
    nn = unet(nn, flags.semantic_ch, False)

    return tl.models.Model(inputs=ni, outputs=nn)

lrelu=lambda x: tl.act.lrelu(x, 0.2)
def unet(ni, out_channel, is_tanh, out_size=flags.img_size_h):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    ngf = 64
    conv1 = Conv2d(ngf, (3, 3), (1, 1), W_init=w_init, act=lrelu)(ni)

    conv2 = Conv2d(ngf, (4, 4), (2, 2), W_init=w_init, act=None, b_init=None)(conv1)
    conv2 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(conv2)

    conv3 = Conv2d(ngf * 2, (4, 4), (1, 1), W_init=w_init, act=None, b_init=None)(conv2)
    conv3 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(conv3)
    conv4 = Conv2d(ngf * 2, (4, 4), (2, 2), W_init=w_init, act=None, b_init=None)(conv3)
    conv4 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(conv4)

    conv5 = Conv2d(ngf * 4, (4, 4), (1, 1), W_init=w_init, act=None, b_init=None)(conv4)
    conv5 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(conv5)
    conv6 = Conv2d(ngf * 4, (4, 4), (2, 2), W_init=w_init, act=None, b_init=None)(conv5)
    conv6 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(conv6)

    conv7 = Conv2d(ngf * 8, (4, 4), (1, 1), W_init=w_init, act=None, b_init=None)(conv6)
    conv7 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(conv7)
    conv8 = Conv2d(ngf * 8, (4, 4), (2, 2), act=lrelu, W_init=w_init, b_init=None)(conv7)
    # 8 8 512 now start upsample

    c_size = conv8.shape[-2]
    ##############################################################################################
    no = None
    for _ in range(1):
        up8 = DeConv2d(ngf * 8, (4, 4), (2, 2), W_init=w_init, b_init=None)(conv8)
        up8 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(up8)
        up7 = Concat(concat_dim=3)([up8, conv7])
        up7 = DeConv2d(ngf * 8, (4, 4), (1, 1), W_init=w_init, b_init=None)(up7)
        up7 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(up7)
        c_size = c_size * 2
        if c_size == out_size:
            no = up7
            break
        up6 = Concat(concat_dim=3)([up7, conv6])
        up6 = DeConv2d(ngf * 4, (4, 4), (2, 2), W_init=w_init, b_init=None)(up6)
        up6 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(up6)
        up5 = Concat(concat_dim=3)([up6, conv5])
        up5 = DeConv2d(ngf * 4, (4, 4), (1, 1), W_init=w_init, b_init=None)(up5)
        up5 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(up5)
        c_size = c_size * 2
        if c_size == out_size:
            no = up5
            break
        up4 = Concat(concat_dim=3)([up5, conv4])
        up4 = DeConv2d(ngf * 2, (4, 4), (2, 2), W_init=w_init, b_init=None)(up4)
        up4 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(up4)
        up3 = Concat(concat_dim=3)([up4, conv3])
        up3 = DeConv2d(ngf * 2, (4, 4), (1, 1), W_init=w_init, b_init=None)(up3)
        up3 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(up3)
        c_size = c_size * 2
        if c_size == out_size:
            no = up3
            break
        up2 = Concat(concat_dim=3)([up3, conv2])
        up2 = DeConv2d(ngf * 1, (4, 4), (2, 2), W_init=w_init, b_init=None)(up2)
        up2 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(up2)
        up1 = Concat(concat_dim=3)([up2, conv1])
        up1 = DeConv2d(ngf * 1, (4, 4), (1, 1), W_init=w_init, b_init=None)(up1)
        up1 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(up1)
        c_size = c_size * 2
        if c_size == out_size:
            no = up1
            break
    if is_tanh:
        up0 = DeConv2d(out_channel, (3, 3), (1, 1), W_init=w_init, act=tf.nn.tanh)(no)
    else:
        up0 = DeConv2d(out_channel, (3, 3), (1, 1), W_init=w_init, b_init=None, act=None)(no)

    return up0

# E is reverse of G without activation in the output layer
def get_Eq(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    ngf = 64
    isize = 64
    u_layers = 3

    ni = Input(shape)
    nn = ni
    nn = unet(nn, 8, False)
    return tl.models.Model(inputs=ni, outputs=nn)


# E is reverse of G without activation in the output layer
def get_Ek(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    ngf = 64
    isize = 64
    n_extra_layers = flags.n_extra_layers

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    ngf = 64
    isize = 64
    n_extra_layers = flags.n_extra_layers

    ni = Input(shape)
    nn = Conv2d(ngf, (4, 4), (2, 2), W_init=w_init, act=tf.nn.relu)(ni)

    nn = Conv2d(ngf * 2, (4, 4), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)

    nn = Conv2d(ngf * 2, (4, 4), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)

    nn = DeConv2d(ngf // 2, (4, 4), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)

    nn = DeConv2d(ngf // 8, (1, 1), (1, 1), W_init=w_init, b_init=None)(nn)

    # nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
    #
    # nn = DeConv2d(ngf // 8, (4, 4), (2, 2), W_init=w_init, act=tf.nn.relu)(nn)
    return tl.models.Model(inputs=ni, outputs=nn)

#
#
def get_img_D(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    ndf = 64
    isize = 64
    n_extra_layers = flags.n_extra_layers

    ni = Input(shape)
    n = Conv2d(ndf, (4, 4), (2, 2), act=None, W_init=w_init, b_init=None)(ni)
    csize, cndf = isize / 2, ndf

    for t in range(n_extra_layers):
        n = SpectralNormConv2d(cndf, (3, 3), (1, 1), act=lrelu, W_init=w_init, b_init=None)(n)

    while csize > 4:
        cndf = cndf * 2
        n = SpectralNormConv2d(cndf, (4, 4), (2, 2), act=lrelu, W_init=w_init, b_init=None)(n)
        csize = csize / 2

    n = Conv2d(1, (4, 4), (1, 1), act=None, W_init=w_init, b_init=None, padding='VALID')(n)

    return tl.models.Model(inputs=ni, outputs=n)
#
#
# def get_z_D(shape_z):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     gamma_init = tf.random_normal_initializer(1., 0.02)
#     lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
#     nz = Input(shape_z)
#     nn = Reshape(shape=[-1, 4 * 4 * 512])(nz)
#     n = Dense(n_units= 4 * 512, act=lrelu, W_init=w_init)(nn)
#     n = Dense(n_units=512, act=lrelu, W_init=w_init, b_init=None)(n)
#     # n = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(n)
#     n = Dense(n_units=128, act=lrelu, W_init=w_init, b_init=None)(n)
#     # n = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(n)
#     n = Dense(n_units=1, act=None, W_init=w_init, b_init=None)(n)
#     return tl.models.Model(inputs=nz, outputs=n)
#
# # 4 * 4 * 512 = 128 * 4 * 4 * 4 G is the transpose of D
# def get_z_G(shape_z):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     gamma_init = tf.random_normal_initializer(1., 0.02)
#     lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
#     nz = Input(shape_z)
#     n = Dense(n_units=4 * 128, act=lrelu, W_init=w_init)(nz)
#     n = Dense(n_units=4 * 4 * 128, act=None, W_init=w_init, b_init=None)(n)
#     n = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init, name=None)(n)
#     n = Dense(n_units=4 * 4 * 4 * 128, act=None, W_init=w_init, b_init=None)(n)
#     n = Reshape(shape=[-1, 4, 4, 512])(n)
#     return tl.models.Model(inputs=nz, outputs=n)


# def get_trans_func(shape=[None, flags.z_dim]):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     act = 'relu'  # lambda x : tf.nn.leaky_relu(x, 0.2)
#
#     ni = Input(shape)
#     nn = Dense(n_units=flags.z_dim, act=act, W_init=w_init)(ni)
#     nn = Dense(n_units=flags.z_dim, act=act, W_init=w_init)(nn)
#     nn = Dense(n_units=flags.z_dim, act=act, W_init=w_init)(nn)
#     nn = Dense(n_units=flags.z_dim, W_init=w_init)(nn)
#
#     return tl.models.Model(inputs=ni, outputs=nn)


# def get_img_D(shape):
#     df_dim = 8
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     gamma_init = tf.random_normal_initializer(1., 0.02)
#     lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
#     ni = Input(shape)
#     n = Conv2d(df_dim, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(ni)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     n = Conv2d(df_dim * 2, (5, 5), (1, 1), act=None, W_init=w_init, b_init=None)(n)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     n = Conv2d(df_dim * 4, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     n = Conv2d(df_dim * 8, (5, 5), (1, 1), act=None, W_init=w_init, b_init=None)(n)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     n = Conv2d(df_dim * 8, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     nf = Flatten(name='flatten')(n)
#     n = Dense(n_units=1, act=None, W_init=w_init)(nf)
#     return tl.models.Model(inputs=ni, outputs=n, name='img_Discriminator')



#
# def get_classifier(shape=[None, flags.z_dim], df_dim=64):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     act = 'relu'  # lambda x : tf.nn.leaky_relu(x, 0.2)
#
#     ni = Input(shape)
#     nn = Dense(n_units=flags.z_dim, act=act, W_init=w_init)(ni)
#     nn = Dense(n_units=flags.z_dim, act=act, W_init=w_init)(nn)
#     nn = Dense(n_units=flags.z_dim, act=act, W_init=w_init)(nn)
#     nn = Dense(n_units=1, W_init=w_init)(nn)
#
#     return tl.models.Model(inputs=ni, outputs=nn)


# def get_G(shape_z, gf_dim=64):    # Dimension of gen filters in first conv layer. [64]
#
#     image_size = 32
#     s16 = image_size // 16
#     # w_init = tf.glorot_normal_initializer()
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     gamma_init = tf.random_normal_initializer(1., 0.02)
#
#     ni = Input(shape_z)
#     nn = Dense(n_units=(gf_dim * 16 * s16 * s16), W_init=w_init, b_init=None)(ni)
#     nn = Reshape(shape=[-1, s16, s16, gf_dim * 16])(nn) # [-1, 2, 2, gf_dim * 8]
#     nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
#     nn = DeConv2d(gf_dim * 8, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn) # [-1, 4, 4, gf_dim * 4]
#     nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
#     nn = DeConv2d(gf_dim * 4, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn) # [-1, 8, 8, gf_dim * 2]
#     nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
#     nn = DeConv2d(gf_dim * 2, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn) # [-1, 16, 16, gf_dim *]
#     nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
#     nn = DeConv2d(gf_dim, (5, 5), (2, 2), b_init=None, W_init=w_init)(nn) # [-1, 32, 32, 3]
#     nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
#     nn = DeConv2d(3, (5, 5), (2, 2), act=tf.nn.tanh, W_init=w_init)(nn)  # [-1, 64, 64, 3]
#
#     return tl.models.Model(inputs=ni, outputs=nn, name='generator')

# def get_E(shape):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     gamma_init = tf.random_normal_initializer(1., 0.02)
#     lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
#     ni = Input(shape)   # (1, 64, 64, 3)
#     n = Conv2d(3, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(ni)  # (1, 16, 16, 3)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     n = Conv2d(32, (5, 5), (1, 1), padding="VALID", act=None, W_init=w_init, b_init=None)(n)  # (1, 12, 12, 32)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     n = Conv2d(64, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)  # (1, 6, 6, 64)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     n = Flatten(name='flatten')(n)
#     nz = Dense(n_units=flags.z_dim, act=None, W_init=w_init)(n)
#     return tl.models.Model(inputs=ni, outputs=nz, name='encoder')
