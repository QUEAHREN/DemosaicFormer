import os
import sys

sys.path.append("/root/MIPI/base_code")

import random
import sys
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import h5py


class LQGTDataset(data.Dataset):
    """
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, etc) and GT image pairs.
    If only GT images are provided, generate LQ images on-the-fly.
    """

    def __init__(self, opt):
        super(LQGTDataset, self).__init__()
        self.opt = opt
        # self.data_type = self.opt['data_type']
        self.data_type = "img"
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environments for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(
            self.data_type, opt["dataroot_GT"]
        )
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(
            self.data_type, opt["dataroot_Frame"]
        )

        if self.opt["use_augdata"]:
            self.paths_Noise = opt["dataroot_Noise"]
            self.paths_Noise_delta = opt["dataroot_Noise_delta"]

        self.nums = len(self.paths_GT)

        assert self.paths_GT, "Error: GT path is empty."
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), "GT and LQ datasets have different number of images - {}, {}.".format(
                len(self.paths_LQ), len(self.paths_GT)
            )
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt["dataroot_GT"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.LQ_env = lmdb.open(
            self.opt["dataroot_LQ"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.data_type == "lmdb" and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()
        GT_path, LQ_path = None, None
        scale = self.opt["scale"]
        GT_size = self.opt["GT_size"]

        # get GT image
        GT_path = self.paths_GT[index]
        resolution = (
            [int(s) for s in self.sizes_GT[index].split("_")]
            if self.data_type == "lmdb"
            else None
        )

        if os.path.islink(GT_path):
            GT_path = os.path.realpath(GT_path)

        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        # print(img_GT.shape)

        if img_GT.shape[2] == 1:
            img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)
        if self.opt["phase"] != "train":  # modcrop in the validation / test phase
            img_GT = util.modcrop(img_GT, scale)
            # img_LQ = util.modcrop(img_LQ, scale)
        if self.opt["color"]:  # change color space if necessary
            img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[
                0
            ]

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            resolution = (
                [int(s) for s in self.sizes_LQ[index].split("_")]
                if self.data_type == "lmdb"
                else None
            )
            img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)

            if img_LQ.shape[2] == 1:
                # for test gray to color
                img_LQ = cv2.cvtColor(img_LQ, cv2.COLOR_GRAY2BGR)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt["phase"] == "train":
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(img_GT, (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        if self.opt["phase"] == "train":
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(
                    img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR
                )
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

            if self.opt["use_augdata"]:
                H, W, C = img_GT.shape
                LQ_size = GT_size // scale

                # randomly crop
                rnd_h = random.randint(0, max(0, H - GT_size))
                rnd_w = random.randint(0, max(0, W - GT_size))

                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[
                    rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
                ]

                if self.opt["use_colorjitter"]:
                    img_GT = util.colorjitter(img_GT)

                # aug
                [img_GT] = util.augment(
                    [img_GT], self.opt["use_flip"], self.opt["use_rot"]
                )

                self.noise_h5 = h5py.File(self.paths_Noise, "r")

                # BGR2RGB
                img_GT = img_GT[:, :, [2, 1, 0]]

                # randomly select noise
                self.noise_delta = None
                rnd_noise_idx = random.randint(0, len(self.noise_h5.keys()) - 1)
                if self.paths_Noise_delta:
                    self.noise_delta_h5 = h5py.File(self.paths_Noise_delta, "r")
                    self.noise_delta = self.noise_delta_h5[str(rnd_noise_idx)]

                img_LQ_sample = util.augData(
                    img_GT,
                    self.noise_h5[str(rnd_noise_idx)],
                    GT_size,
                    self.opt["use_evsavg"],
                    self.opt["use_gaussian"],
                    self.opt["use_evsmask"],
                    self.noise_delta,
                )

                # HWC2CHW
                img_GT = torch.from_numpy(
                    np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
                ).float()
                img_LQ_sample = torch.from_numpy(
                    np.ascontiguousarray(np.transpose(img_LQ_sample, (2, 0, 1)))
                ).float()
                if LQ_path is None:
                    LQ_path = GT_path
                return {
                    "LQ": img_LQ_sample,
                    "GT": img_GT,
                    "LQ_path": LQ_path,
                    "GT_path": GT_path,
                }

            else:
                H, W, C = img_LQ.shape
                LQ_size = GT_size // scale
                # randomly crop
                rnd_h = random.randint(0, max(0, H - GT_size) // 4) * 4
                rnd_w = random.randint(0, max(0, W - GT_size) // 4) * 4
                rnd_h, rnd_w = int(rnd_h * scale), int(rnd_w * scale)
                img_LQ = img_LQ[rnd_h : rnd_h + LQ_size, rnd_w : rnd_w + LQ_size, :]

                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[
                    rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
                ]

        if self.opt["color"]:  # change color space if necessary
            img_LQ = util.channel_convert(C, self.opt["color"], [img_LQ])[
                0
            ]  # TODO during val no definition

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LQ = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))
        ).float()

        if LQ_path is None:
            LQ_path = GT_path

        return {"LQ": img_LQ, "GT": img_GT, "LQ_path": LQ_path, "GT_path": GT_path}

    def __len__(self):
        return len(self.paths_GT)


if __name__ == "__main__":
    sys.path.append("/root/MIPI/base_code")
    import options.options as option

    opt = option.parse(
        "/root/MIPI/base_code/options/train_finalfinal/Restormer_Denoise_80_5e-4_mixaug.yml",
        is_train=True,
    )
    x = LQGTDataset(opt["datasets"]["train"])
    x.__getitem__(0)
