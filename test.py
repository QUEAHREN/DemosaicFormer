import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torch

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
parser.add_argument(
    "-opt",
    type=str,
    default="",
    help="Path to options YMAL file.",
)
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)
util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []

    for data in test_loader:
        # need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        # if data['LQ_path'][0].split('/')[-1] == '0802.png':
        #     continue

        need_GT = False
        scale = opt["scale"]

        def save_img(img, suffix):
            save_path = osp.join(dataset_dir, img_name + suffix + ".png")
            util.save_img(img, save_path)

        input_ = data["LQ"]
        B, C, H, W = input_.shape
        split_data, starts = util.splitimage(input_, crop_size=192, overlap_size=96)

        for i, datai in enumerate(split_data):
            model.feed_data({"LQ": datai}, need_GT=need_GT)
            model.test()
            split_data[i] = model.get_current_visuals(need_GT=need_GT)["rlt"]
        print(len(split_data))
        img_path = data["GT_path"][0] if need_GT else data["LQ_path"][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        sr_img = util.mergeimage(
            split_data,
            starts,
            crop_size=192,
            resolution=(B, C, H, W),
            is_x2y2=False,
            is_gauss=True,
        )
        sr_img = util.tensor2img(sr_img)

        suffix = opt["suffix"]
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = osp.join(dataset_dir, img_name + ".png")

        if opt["save_img"]:
            util.save_img(sr_img, save_img_path)

        logger.info(img_name)

    if need_GT:  # metrics
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
        ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
        logger.info(
            "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
                test_set_name, ave_psnr, ave_ssim
            )
        )
        if test_results["psnr_y"] and test_results["ssim_y"]:
            ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
            ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
            logger.info(
                "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n".format(
                    ave_psnr_y, ave_ssim_y
                )
            )
