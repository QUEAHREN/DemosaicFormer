import shutil
from PIL import Image
import os
import numpy as np


def quadBayerSampler(image):
    img = image.copy()

    # Quad R
    img[::4, ::4, 1:3] = 0
    img[1::4, 1::4, 1:3] = 0
    img[::4, 1::4, 1:3] = 0
    img[1::4, ::4, 1:3] = 0

    # Quad B
    img[3::4, 2::4, 0:2] = 0
    img[3::4, 3::4, 0:2] = 0
    img[2::4, 3::4, 0:2] = 0
    img[2::4, 2::4, 0:2] = 0

    # Quad G12
    img[1::4, 2::4, ::2] = 0
    img[1::4, 3::4, ::2] = 0
    img[::4, 2::4, ::2] = 0
    img[::4, 3::4, ::2] = 0

    # Quad G21
    img[2::4, 1::4, ::2] = 0
    img[3::4, 1::4, ::2] = 0
    img[2::4, ::4, ::2] = 0
    img[3::4, ::4, ::2] = 0

    return img


def read_bin(img_path):
    """
    Read bin data
    """
    img_data = np.fromfile(img_path, dtype=np.uint16)
    w = int(img_data[0])
    h = int(img_data[1])
    assert w * h == img_data.size - 2
    quad = np.clip(img_data[2:].reshape([h, w]).astype(np.float32), 0, 1023)
    return quad


def crop_and_save(image, image_name, output_dir, crop_size=512, stride=256):

    i = 0
    height = image.shape[0]
    width = image.shape[1]
    print(width, height)
    for y in range(0, height, crop_size):
        for x in range(0, width, crop_size):
            if y + crop_size > height:
                y0 = height - 512
                y1 = height
            else:
                y0 = y
                y1 = y + crop_size
            if x + crop_size > width:
                x0 = width - 512
                x1 = width
            else:
                x0 = x
                x1 = x + crop_size

            crop = image[y0:y1, x0:x1, :]
            # print(y0, y1, x0, x1, y1 - y0, x1 - x0)
            # print(crop[:,:,0])
            # crop = crop/1023 * 255
            # crop = np.uint8(crop)
            crop = Image.fromarray(crop)
            cropped_image_name = f"{image_name}_{i}.png"

            crop.save(os.path.join(output_dir, cropped_image_name))
            i = i + 1


def Sampler(img):

    img = np.asarray(img)
    img = img.copy()
    img = np.stack((img,) * 3, axis=-1)
    # print(img.shape)
    H, W, _ = img.shape
    H4, W4 = H // 4, W // 4
    img = quadBayerSampler(img)

    return img


def save_imglist(imglist, image_name, output_dir):
    for i in range(len(imglist)):

        crop = np.transpose(imglist[i], (1, 2, 0))
        crop = Image.fromarray(crop)
        cropped_image_name = f"{image_name}_{i}.png"
        crop.save(os.path.join(output_dir, cropped_image_name))


def splitimage(imgtensor, crop_size=128, overlap_size=64):
    C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts and hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts and wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, hs : hs + crop_size, ws : ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts


def evsQuadBayerSampler(image, use_evsavg=False):
    img = image.copy()

    # evs pix
    if use_evsavg:
        img[1::4, 1::4, 0] = (
            img[::4, ::4, 0] + img[::4, 1::4, 0] + img[1::4, ::4, 0]
        ) / 3
        img[3::4, 3::4, 2] = (
            img[2::4, 2::4, 2] + img[2::4, 3::4, 2] + img[3::4, 2::4, 2]
        ) / 3
        # print(img[1::4,1::4, 0], img[3::4,3::4, 2])
    else:
        img[1::4, 1::4, 0] = 0
        img[3::4, 3::4, 2] = 0

    # Quad R
    img[::4, ::4, 1:3] = 0
    img[1::4, 1::4, 1:3] = 0
    img[::4, 1::4, 1:3] = 0
    img[1::4, ::4, 1:3] = 0

    # Quad B
    img[3::4, 2::4, 0:2] = 0
    img[3::4, 3::4, 0:2] = 0
    img[2::4, 3::4, 0:2] = 0
    img[2::4, 2::4, 0:2] = 0

    # Quad G12
    img[1::4, 2::4, ::2] = 0
    img[1::4, 3::4, ::2] = 0
    img[::4, 2::4, ::2] = 0
    img[::4, 3::4, ::2] = 0

    # Quad G21
    img[2::4, 1::4, ::2] = 0
    img[3::4, 1::4, ::2] = 0
    img[2::4, ::4, ::2] = 0
    img[3::4, ::4, ::2] = 0

    return img


if __name__ == "__main__":

    # config
    origin_bin_data_dir = "/root/autodl-tmp/MIPI_dataset/train_copy/input"
    origin_gt_data_dir = "/root/autodl-tmp/MIPI_dataset/train_copy/gt"
    output_gt_dir = "/root/autodl-tmp/MIPI_dataset/train_copy/new_qS_1024"
    output_input_dir = "/root/autodl-tmp/MIPI_dataset/train_copy/new_gt_1024"

    crop_size = 1024
    overlap_size = 512

    input_dir = origin_bin_data_dir
    output_dir = output_gt_dir
    flag = False
    filelist = sorted(os.listdir(input_dir))
    for filename in filelist:
        if filename.endswith(".bin"):
            pre_name = filename.split(".")[0]
            print(pre_name)

            image_path = os.path.join(input_dir, filename)
            input_quad = read_bin(image_path).astype(np.uint16)

            img = Sampler(input_quad)
            img = np.transpose(img, (2, 0, 1))
            print(img.shape)
            img = img / 1023 * 255
            img = np.uint8(img)
            split_data, starts = splitimage(
                img, crop_size=crop_size, overlap_size=overlap_size
            )

            print(split_data[0].shape)
            save_imglist(split_data, pre_name, output_dir)

    input_dir = origin_gt_data_dir
    output_dir = output_input_dir

    filelist = sorted(os.listdir(input_dir))
    for filename in filelist:
        if filename.endswith(".png"):
            pre_name = filename.split(".")[0]

            print(pre_name)
            image_path = os.path.join(input_dir, filename)
            img = Image.open(image_path)
            width, height = img.size
            print(width, height)
            img = np.asarray(img)
            img = np.transpose(img, (2, 0, 1))

            split_data, starts = splitimage(
                img, crop_size=crop_size, overlap_size=overlap_size
            )

            save_imglist(split_data, pre_name, output_dir)
