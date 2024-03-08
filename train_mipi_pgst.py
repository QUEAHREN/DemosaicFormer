import os
import math
import argparse
import random
import logging
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler, EnlargedSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def init_dist(backend="nccl", **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")
    # os.environ['RANK'] = '0'
    # os.environ['WORLD_SIZE'] = '2'
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'
    # os.environ["RANK"] = str(0)
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt",
        type=str,
        default="/root/MIPI/base_code/options/train/train_WGWS.yml",
        help="Path to option YAML file.",
    )
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local-rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            opt["name"],
            level=logging.INFO,
            screen=True,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            # tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
            tb_logger = SummaryWriter(log_dir="/root/tf-logs/" + opt["name"])
            print("/logs/" + opt["name"])
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=True
        )
        logger = logging.getLogger("base")

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt["train"]["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info("Random seed: {}".format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                # train_sampler = DistIterSampler(
                #     train_set, world_size, rank, dataset_ratio
                # )
                train_sampler = EnlargedSampler(train_set, world_size, rank, 1)
                total_epochs = int(math.ceil(total_iters / (train_size * 1)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    iters = opt["datasets"]["train"].get("iters")
    batch_size = opt["datasets"]["train"].get("batch_size_per_gpu")
    mini_batch_sizes = opt["datasets"]["train"].get("mini_batch_sizes")
    gt_size = opt["datasets"]["train"].get("GT_size")
    mini_gt_sizes = opt["datasets"]["train"].get("GT_sizes")

    groups = np.array([sum(iters[0 : i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)
    scale = opt["scale"]

    #### training
    print("total_epoch=", total_epochs)
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )
    logger_midaug = True
    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            ### ------Progressive learning ---------------------
            j = ((current_step > groups) != True).nonzero()[0]
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]

            if logger_j[bs_j]:
                logger.info(
                    "\n Updating Patch_Size to {} and Batch_Size to {} \n".format(
                        mini_gt_size, mini_batch_size * torch.cuda.device_count()
                    )
                )
                logger_j[bs_j] = False

            lq = train_data["LQ"]
            gt = train_data["GT"]

            if mini_batch_size < batch_size:
                indices = random.sample(range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]

            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random() // 4) * 4
                y0 = int((gt_size - mini_gt_size) * random.random() // 4) * 4
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:, :, x0:x1, y0:y1]
                gt = gt[:, :, x0 * scale : x1 * scale, y0 * scale : y1 * scale]

            train_data = {"LQ": lq, "GT": gt}


            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)
            #### update learning rate
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            #### log
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "[epoch:{:3d}, iter:{:8,d}, lr:(".format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += "{:.3e},".format(v)
                message += ")] "
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)
            #### validation
            if (
                opt["datasets"].get("val", None)
                and current_step % opt["train"]["val_freq"] == 0
            ):
                if (
                    opt["model"] in ["sr", "srgan", "sd"] and rank <= 0
                ):  # image restoration validation
                    # does not support multi-GPU validation
                    print(rank)
                    pbar = util.ProgressBar(len(val_loader))
                    avg_psnr = 0.0
                    avg_ssim = 0
                    idx = 0
                    for val_data in val_loader:
                        idx += 1
                        img_name = os.path.splitext(
                            os.path.basename(val_data["LQ_path"][0])
                        )[0]
                        img_dir = os.path.join(opt["path"]["val_images"], img_name)
                        util.mkdir(img_dir)

                        model.feed_data(val_data)
                        model.test()

                        visuals = model.get_current_visuals()
                        sr_img = util.tensor2img(visuals["rlt"])  # uint8
                        gt_img = util.tensor2img(visuals["GT"])  # uint8

                        # Save SR images for reference
                        save_img_path = os.path.join(
                            img_dir, "{:s}_{:d}.png".format(img_name, current_step)
                        )

                        if opt["save_img"]:
                            util.save_img(sr_img, save_img_path)

                        # calculate PSNR
                        sr_img, gt_img = util.crop_border(
                            [sr_img, gt_img], opt["scale"]
                        )
                        avg_psnr += util.calculate_psnr(sr_img, gt_img)
                        avg_ssim += util.calculate_ssim(sr_img, gt_img)
                        pbar.update("Test {}".format(img_name))

                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx

                    # log
                    logger.info("# Validation # PSNR: {:.4e}".format(avg_psnr))
                    logger.info("# Validation # SSIM: {:.4e}".format(avg_ssim))
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        tb_logger.add_scalar("psnr", avg_psnr, current_step)
                        tb_logger.add_scalar("ssim", avg_ssim, current_step)
                        # tb_logger.add_figure('output', sr_img, current_step)

            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of training.")
        tb_logger.close()


if __name__ == "__main__":
    main()
