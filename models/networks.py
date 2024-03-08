import torch

import argparse
import options.options as option
import models.archs.DemosaicFormer_archs.DemosaicFormer_arch as DemosaicFormer


# Generator
def define_G(opt):
    opt_net = opt["network_G"]
    which_model = opt_net["which_model_G"]

    if which_model == "DemosaicFormer":
        netG = DemosaicFormer.DemosaicFormer(
            inp_channels=opt_net["inp_channels"],
            out_channels=opt_net["out_channels"],
            dim=opt_net["dim"],
            num_blocks=opt_net["num_blocks"],
            num_refinement_blocks=opt_net["num_refinement_blocks"],
            heads=opt_net["heads"],
            ffn_expansion_factor=opt_net["ffn_expansion_factor"],
            bias=opt_net["bias"],
            LayerNorm_type=opt_net["LayerNorm_type"],  ## Other option 'BiasFree'
            dual_pixel_task=opt_net[
                "dual_pixel_task"
            ],  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        )
    else:
        raise NotImplementedError(
            "Generator model [{:s}] not recognized".format(which_model)
        )

    return netG
