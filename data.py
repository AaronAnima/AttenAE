import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags

import ipdb


# 1 5_o_Clock_Shadow：刚长出的双颊胡须
# 2 Arched_Eyebrows：柳叶眉
# 3 Attractive：吸引人的
# 4 Bags_Under_Eyes：眼袋
# 5 Bald：秃头
# 6 Bangs：刘海
# 7 Big_Lips：大嘴唇
# 8 Big_Nose：大鼻子
# 9 Black_Hair：黑发
# 10 Blond_Hair：金发
# 11 Blurry：模糊的
# 12 Brown_Hair：棕发
# 13 Bushy_Eyebrows：浓眉
# 14 Chubby：圆胖的
# 15 Double_Chin：双下巴
# 16 Eyeglasses：眼镜
# 17 Goatee：山羊胡子
# 18 Gray_Hair：灰发或白发
# 19 Heavy_Makeup：浓妆
# 20 High_Cheekbones：高颧骨
# 21 Male：男性
# 22 Mouth_Slightly_Open：微微张开嘴巴
# 23 Mustache：胡子，髭
# 24 Narrow_Eyes：细长的眼睛
# 25 No_Beard：无胡子
# 26 Oval_Face：椭圆形的脸
# 27 Pale_Skin：苍白的皮肤
# 28 Pointy_Nose：尖鼻子
# 29 Receding_Hairline：发际线后移
# 30 Rosy_Cheeks：红润的双颊
# 31 Sideburns：连鬓胡子
# 32 Smiling：微笑
# 33 Straight_Hair：直发
# 34 Wavy_Hair：卷发
# 35 Wearing_Earrings：戴着耳环
# 36 Wearing_Hat：戴着帽子
# 37 Wearing_Lipstick：涂了唇膏
# 38 Wearing_Necklace：戴着项链
# 39 Wearing_Necktie：戴着领带
# 40 Young：年轻人
def totaldic(filename):
    f = open(filename, mode="r")
    dic = {}
    for i in f:
        lst = i.split(",")
        dic[lst[0]] = lst[1]
    return dic

def partialdic():
    animals = 'Mule Antelope Brown_bear Panda Polar_bear Teddy_bear Cat Fox Jaguar Lynx Red_panda Tiger Lion Dog Leopard Cheetah Otter Raccoon Camel Cattle Giraffe Rhinoceros Goat Horse Hamster Kangaroo Koala Mouse Pig Rabbit Squirrel Sheep Zebra Monkey Hippopotamus Deer Elephant Porcupine Hedgehog Bull'

    dic = {}
    for item in animals.split(' '):
        if '_' in item:
            a, b = item.split('_')
            item = a + ' ' + b
        dic[item + '\n'] = True
    return dic

def get_lists(filename, dic1, dic2):
    f = open(filename, mode="r")
    list_mask = []
    list_img = []
    list_bbox = []
    list_label = []
    cnt = 0
    for i in f:
        if cnt == 0:
            cnt += 1
            continue
        lst = i.split(",")
        msk = lst[0]
        img = lst[1]
        cls = lst[2]
        ani = dic1[cls]
        x_min = float(lst[4])
        x_max = float(lst[5])
        y_min = float(lst[6])
        y_max = float(lst[7])
        # if ani == 'Dog':
        #     input()
        if ani in dic2:
            img_path = '/home/asus/data/OpenImage/test/' + img + '.jpg'
            mask_path = '/home/asus/data/OpenImage/masks/' + msk
            list_img.append(img_path)
            list_mask.append(mask_path)
            list_bbox.append([x_min, x_max, y_min, y_max])
            if ' ' in ani:
                a, b = ani.split(' ')
                ani = a + '_' + b
            list_label.append(ani)

    # print(len(list_img))
    # print(len(list_mask))
    # print(len(list_bbox))
    # input()
    return list_img, list_mask, list_bbox, list_label


def load_openimage():
    dic_code_animal = totaldic('/home/asus/data/OpenImage/description.csv')
    dic_animals = partialdic()
    list_img, list_mask, list_bbox, list_label = get_lists('/home/asus/data/OpenImage/labels.csv', dic_code_animal,
                                                           dic_animals)

    # images = images / 127.5 - 1

    def generator_train():
        for image, mask, bbox, label in zip(list_img, list_mask, list_bbox, list_label):
            yield image.encode('utf-8'), mask.encode('utf-8'), tf.convert_to_tensor(bbox), label



    def _map_fn(image, mask, bbox, label):
        image = tf.io.read_file(image)
        mask = tf.io.read_file(mask)
        # ipdb.set_trace()
        #
        image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
        mask = tf.image.decode_png(mask, channels=1)  # get RGB with 0~1
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
        mask = tf.concat([mask, mask, mask], 2)
        image = tf.image.resize(image, (flags.img_size_h * 8, flags.img_size_w * 8))
        mask = tf.image.resize(mask, (flags.img_size_h * 8, flags.img_size_w * 8))
        # M_rotate = tl.prepro.affine_rotation_matrix(angle=(-16, 16))
        # M_flip = tl.prepro.affine_horizontal_flip_matrix(prob=0.5)
        # M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.8, 1.2))

        # h, w, _ = x.shape
        # M_combined = M_zoom.dot(M_flip).dot(M_rotate)
        # transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=w, y=h)
        # x = tl.prepro.affine_transform_cv2(x, transform_matrix, border_mode='replicate')
        #
        # x = tl.prepro.flip_axis(x, axis=1, is_random=True)
        # x = tl.prepro.rotation(x, rg=16, is_random=True, fill_mode='nearest')
        # x = tl.prepro.crop(x, wrg=256, hrg=256, is_random=True)
        # x = x / 127.5 - 1.
        image = image * 2 - 1
        # image = tf.image.random_flip_left_right(image)
        return image, mask, bbox, label

    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.string, tf.string, tf.float32, tf.string))
    train_ds = train_ds.shuffle(buffer_size=4096)
    # ds = train_ds.shuffle(buffer_size=4096)
    ds = train_ds.repeat(flags.n_epoch)
    ds = ds.map(_map_fn, num_parallel_calls=4)
    ds = ds.batch(flags.batch_size_train)
    ds = ds.prefetch(buffer_size=4)  # For concurrency
    print('There are {} imgs'.format(len(list_img)))
    return ds, len(list_img)

#
# def get_Y2X_train():
#     cat_images_path = tl.files.load_file_list(path='/home/asus/Workspace/new_code/cat2dog/trainA/',
#                                                             regx='.*.jpg', keep_prefix=True, printable=False)
#     dog_images_path = tl.files.load_file_list(path='/home/asus/Workspace/new_code/cat2dog/trainB/',
#                                                             regx='.*.jpg', keep_prefix=True, printable=False)
#     len_cat = len(cat_images_path)
#     len_dog = len(dog_images_path)
#     dataset_len = min(len_cat, len_dog)
#     flags.len_dataset = dataset_len
#     cat_images_path = cat_images_path[0:dataset_len]
#     dog_images_path = dog_images_path[0:dataset_len]
#
#     def generator_train():
#         for cat_path, dog_path in zip(cat_images_path, dog_images_path):
#             yield cat_path.encode('utf-8'), dog_path.encode('utf-8')
#
#     def _map_fn(image_path, image_path_2):
#         image = tf.io.read_file(image_path)
#         image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
#         image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#         image = image[20:198, :, :]  # central crop
#         image = tf.image.resize(image, [64, 64])  # how to do BICUBIC?
#         image = image * 2 - 1
#
#         image2 = tf.io.read_file(image_path_2)
#         image2 = tf.image.decode_jpeg(image2, channels=3)  # get RGB with 0~1
#         image2 = tf.image.convert_image_dtype(image2, dtype=tf.float32)
#         image2 = image2[20:198, :, :]  # central crop
#         image2 = tf.image.resize(image2, [64, 64])  # how to do BICUBIC?
#         image2 = image2 * 2 - 1
#
#         return image, image2
#
#     train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.string, tf.string))
#     ds = train_ds.shuffle(buffer_size=4096)
#     # ds = ds.shard(num_shards=hvd.size(), index=hvd.rank())
#     n_step_epoch = int(dataset_len // flags.batch_size_train)
#     n_epoch = int(flags.step_num // n_step_epoch)
#     ds = ds.repeat(n_epoch)
#     ds = ds.map(_map_fn, num_parallel_calls=4)
#     ds = ds.batch(flags.batch_size_train)
#     ds = ds.prefetch(buffer_size=2)
#     return ds, len(dog_images_path) + len(cat_images_path)
#
#
#
#
#
# def get_celebA_attr(Attr_type):
#     image_path = "/home/asus/Workspace/WMDGAN/data/celebA/celebA"
#     CelebA_Attr_file = "/home/asus/Workspace/WMDGAN/data/celebA/list_attr_celeba.txt"
#     labels = []
#     with open(CelebA_Attr_file, "r") as Attr_file:
#         Attr_info = Attr_file.readlines()
#         Attr_info = Attr_info[2:]
#         index = 0
#         for line in Attr_info:
#             index += 1
#             info = line.split()
#             filename = info[0]
#             filepath_old = os.path.join(image_path, filename)
#             if os.path.isfile(filepath_old):
#                 labels.append(info[Attr_type])
#             else:
#                 print("%d: not found %s\n" % (index, filepath_old))
#                 not_found_txt.write(line)
#     return labels
#
# def load_imagenet(path='~/Workspace/BigGAN/data/ImageNet/train', regx='.*.JPEG', keep_prefix=True, printable=False):
#     images_path = []
#     folder_list = tl.files.load_folder_list(path)
#     cnt = 0
#     for temp_path in folder_list:
#         partial_path = tl.files.load_file_list(path=temp_path, regx=regx,
#                                                keep_prefix=keep_prefix, printable=printable)
#         images_path = images_path + partial_path
#     return images_path
#
#
# def get_dataset_train():
#     # images = tl.files.load_celebA_dataset('/home/asus/data/celebA/')
#     # ipdb.set_trace()
#     # images = tl.files.load_celebA_dataset(path='../data/celebA')
#     images_path = load_imagenet(path='/home/asus/Workspace/BigGAN/data/ImageNet/train', regx='.*.JPEG', keep_prefix=True,
#                             printable=False)
#     # images = images / 127.5 - 1
#
#     def generator_train():
#         for image in images_path:
#             # ipdb.set_trace()
#
#             yield image.encode('utf-8')
#
#     def _map_fn(image_path):
#         image = tf.io.read_file(image_path)
#         # ipdb.set_trace()
#         #
#         image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
#         x = tf.image.convert_image_dtype(image, dtype=tf.float32)
#
#
#         # M_rotate = tl.prepro.affine_rotation_matrix(angle=(-16, 16))
#         # M_flip = tl.prepro.affine_horizontal_flip_matrix(prob=0.5)
#         # M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.8, 1.2))
#
#
#         # h, w, _ = x.shape
#         # M_combined = M_zoom.dot(M_flip).dot(M_rotate)
#         # transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=w, y=h)
#         # x = tl.prepro.affine_transform_cv2(x, transform_matrix, border_mode='replicate')
#         #
#         # x = tl.prepro.flip_axis(x, axis=1, is_random=True)
#         # x = tl.prepro.rotation(x, rg=16, is_random=True, fill_mode='nearest')
#         #
#         x = tf.image.resize(x, (flags.img_size_h, flags.img_size_w))
#         # x = tl.prepro.crop(x, wrg=256, hrg=256, is_random=True)
#         # x = x / 127.5 - 1.
#         x = x * 2 - 1
#         x = tf.image.random_flip_left_right(x)
#         return x
#
#     train_ds = tf.data.Dataset.from_generator(generator_train, output_types=tf.string)
#     train_ds = train_ds.shuffle(buffer_size=4096)
#     # ds = train_ds.shuffle(buffer_size=4096)
#     ds = train_ds.repeat(flags.n_epoch)
#     ds = ds.map(_map_fn, num_parallel_calls=4)
#     ds = ds.batch(flags.batch_size_train)
#     ds = ds.prefetch(buffer_size=4)  # For concurrency
#     return ds, len(images_path)
#
#
#


