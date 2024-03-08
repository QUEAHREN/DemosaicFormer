import logging
from collections import OrderedDict

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, FSLoss, GradientLoss, PSNRLoss, fftLoss
from scipy.spatial import KDTree

from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from utils.EMA import EMA


logger = logging.getLogger("base")


class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        self.opt = opt
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        inchannel = 6
        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.use_ema = False
        if train_opt["ema"]:
            self.use_ema = True
            self.netG_ema = EMA(model=self.netG, decay=train_opt["ema_decay"])
            self.netG_ema.register()

        if opt["dist"]:
            self.netG = DistributedDataParallel(
                self.netG, device_ids=[torch.cuda.current_device()]
            )
        else:
            self.netG = DataParallel(self.netG)

        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt["pixel_criterion"]
            self.loss_type = loss_type
            if loss_type == "l1":
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == "l2":
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == "psnr":
                self.cri_pix == PSNRLoss().to(self.device)
            else:
                raise NotImplementedError(
                    "Loss type [{:s}] is not recognized.".format(loss_type)
                )
            self.l_pix_w = train_opt["pixel_weight"]

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.netG.named_parameters():  # can optimize for a part of the model
                if train_opt["frozen_stage1"]:
                    if v.requires_grad and k.startswith("module.stage2"):
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning("Params [{:s}] will not optimize.".format(k))
                else:
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt["optim_type"] == "Adam":
                self.optimizer_G = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "CosineAnnealingLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt["T_period"],
                            eta_min=train_opt["eta_min"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                        )
                    )
            elif train_opt["lr_scheme"] == "CosineAnnealingRestartCyclicLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingRestartCyclicLR(
                            optimizer,
                            train_opt["periods"],
                            restart_weights=train_opt["restart_weights"],
                            eta_mins=train_opt["eta_mins"],
                        )
                    )
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):

        self.var_L = data["LQ"].to(self.device)
        if need_GT:
            self.real_H = data["GT"].to(self.device)  # GT

    def optimize_parameters(self, step):

        self.optimizer_G.zero_grad()

        LR_input = self.var_L

        self.fake_H = self.netG(LR_input)

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)

        l_pix.backward()
        self.optimizer_G.step()
        if self.use_ema:
            self.netG_ema.update()
        # set log
        self.log_dict["l_pix"] = l_pix.item()

    def test(self):
        self.netG.eval()
        if self.use_ema:
            self.netG_ema.apply_shadow()

        LR_input = self.var_L

        with torch.no_grad():
            if self.opt["save_mid"]:
                self.fake_H, self.fake_H_sample = self.netG(LR_input)
            else:
                self.fake_H = self.netG(LR_input)
            # self.fake_H = self.netG(LR_img_event,None)
        if self.use_ema:
            self.netG_ema.restore()
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):

        out_dict = OrderedDict()
        out_dict["LQ"] = self.var_L.detach()[0].float().cpu()

        out_dict["rlt"] = self.fake_H.detach()[0].float().cpu()
        if self.opt["save_mid"]:
            out_dict["mid"] = self.fake_H_sample.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.real_H.detach()[0].float().cpu()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(
            self.netG, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.netG.__class__.__name__, self.netG.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.netG, "G", iter_label)
        if self.use_ema:
            self.netG_ema.apply_shadow()
            self.save_network(self.netG, "G_EMA", iter_label)
            self.netG_ema.restore()
