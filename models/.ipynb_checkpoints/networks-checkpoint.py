import torch

import argparse
import models.archs.DenoiseNet as DenoiseNet
import models.archs.mwisp.mwisp_arch as MWRCAN
import options.options as option
import models.archs.WGWSNet as WGWSNet
import models.archs.WGWSNet_2 as WGWSNet_2
import models.archs.ECFNet as ECFNet
import models.archs.NAF_archs.NAFNet_arch as NAFNet
import models.archs.mirnet_v2_arch as MIRNet
import models.archs.Restormer_archs.restormer_arch as Restormer
import models.archs.ShuffleFormer as ShuffleFormer
# import models.archs.PAN_event5_arch_group as PAN_event5_arch_group
# import models.archs.DDBPN as DDBPN




# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

  

    if which_model == 'WGWSNet':
        netG = WGWSNet.WGWSNet(base_channel=24, num_res=6)
    elif which_model == 'W2Net':
        netG = WGWSNet.W2Net(base_channel=24, num_res=6)
    elif which_model == 'WGWSNet_2':
        netG = WGWSNet_2.WGWSNet_2(base_channel=24, num_res=6)

    elif which_model == 'ECFNet':
        netG = ECFNet.MIMOUNet_complete(base_channel=24, num_res=6)
    elif which_model == 'E2Net':
        netG = ECFNet.E2Net(base_channel=24, num_res=6)
    elif which_model == 'E2Net_3depth':
        netG = ECFNet.E2Net_3depth(base_channel=24, num_res=6)
    elif which_model == 'E2Net_Omni':
        netG = ECFNet.E2Net_Omni(base_channel=24, num_res=6)

    elif which_model == 'NAFNet':
        netG = NAFNet.NAFNet(img_channel=3, width=opt_net['width'], middle_blk_num=opt_net['middle_blk_num'], enc_blk_nums=opt_net['enc_blk_nums'], dec_blk_nums=opt_net['dec_blk_nums'])
    elif which_model == 'N2Net':
        netG = NAFNet.N2Net(img_channel=3, width=opt_net['width'], middle_blk_num=opt_net['middle_blk_num'], enc_blk_nums=opt_net['enc_blk_nums'], dec_blk_nums=opt_net['dec_blk_nums'])
    elif which_model == 'N2Net_par':
        netG = NAFNet.N2Net_par(img_channel=3, width=opt_net['width'], middle_blk_num=opt_net['middle_blk_num'], enc_blk_nums=opt_net['enc_blk_nums'], dec_blk_nums=opt_net['dec_blk_nums'])
    elif which_model == 'N2Net_pack':
        netG = NAFNet.N2Net_pack(img_channel=3, width=opt_net['width'], middle_blk_num=opt_net['middle_blk_num'], enc_blk_nums=opt_net['enc_blk_nums'], dec_blk_nums=opt_net['dec_blk_nums'])
    
    elif which_model == 'DenoiseNet':
        netG = DenoiseNet.DenoiseNet()
    elif which_model == 'MWRCAN':
        netG = MWRCAN.MWRCAN()
    
    
    elif which_model == 'W2NetLocal':
        netG = WGWSNet.W2NetLocal(base_channel=24, num_res=6)
        
    elif which_model == 'MIRNet':
        netG = MIRNet.MIRNet_v2()
    elif which_model == 'MIR2Net':
        netG = MIRNet.MIR2Net()
    elif which_model == 'MIR2Net_next':
        netG = MIRNet.MIR2Net_next()

    elif which_model == 'Restormer':
        netG = Restormer.Restormer(inp_channels=opt_net['inp_channels'], 
            out_channels= opt_net['out_channels'], 
            dim = opt_net['dim'],
            num_blocks = opt_net['num_blocks'], 
            num_refinement_blocks = opt_net['num_refinement_blocks'],
            heads = opt_net['heads'],
            ffn_expansion_factor = opt_net['ffn_expansion_factor'],
            bias = opt_net['bias'],
            LayerNorm_type = opt_net['LayerNorm_type'],   ## Other option 'BiasFree'
            dual_pixel_task = opt_net['dual_pixel_task']        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        )
    elif which_model == 'Restormer_2stage':
        netG = Restormer.Restormer_2stage(inp_channels=opt_net['inp_channels'], 
            out_channels= opt_net['out_channels'], 
            dim = opt_net['dim'],
            num_blocks = opt_net['num_blocks'], 
            num_refinement_blocks = opt_net['num_refinement_blocks'],
            heads = opt_net['heads'],
            ffn_expansion_factor = opt_net['ffn_expansion_factor'],
            bias = opt_net['bias'],
            LayerNorm_type = opt_net['LayerNorm_type'],   ## Other option 'BiasFree'
            dual_pixel_task = opt_net['dual_pixel_task']        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        )
    elif which_model == 'ShuffleFormer_2stage':
        netG = ShuffleFormer.ShuffleFormer_2stage(
            img_size=opt_net['input_size'], 
            embed_dim=32, win_size=8, 
            token_projection='linear', 
            token_mlp='leff', 
            depths=opt_net['depths'], 
            dd_in=3, 
            repeat=opt_net['repeat_num']
        )
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


