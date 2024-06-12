## DemosaicFormer: Coarse-to-Fine Demosaicing Network for HybridEVS Camera

## Overview

This repository contains the official implementation of our paper "DemosaicFormer: Coarse-to-Fine Demosaicing Network for HybridEVS Camera", accepted at the CVPR Workshop 2024. 

See more details in [[report]](https://arxiv.org/abs/2405.04867), [[paper]](https://openaccess.thecvf.com/content/CVPR2024W/MIPI/papers/Xu_DemosaicFormer_Coarse-to-Fine_Demosaicing_Network_for_HybridEVS_Camera_CVPRW_2024_paper.pdf), [[certificate]](https://mipi-challenge.org/MIPI2024/award_certificates_2024.pdf)

Our solution competes in MIPI 2024 Demosaic for HybridEVS Camera, achieving the BEST performance in terms of PNSR and SSIM.


## Installation

Use the following command line for the installation of dependencies required to run DemosaicFormer.

```
git clone https://github.com/QUEAHREN/DemosaicFormer.git
cd DemosaicFormer
conda create -n pytorch181 python=3.7
conda activate pytorch181
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

## Dataset & Data Preprocess

You can download the training set from [MIPI2024 training set](https://drive.google.com/drive/folders/1Yi4ZqNm-0AfdWm8gzLAhxX9ooIWkhqZt?usp=drive_link)

Please change the path in ``` data_preprocess.py``` and use command line as following to convert the bin files to RGB images:

```
python data_preprocess.py
```

## Evaluation

#### Pretrained Model Download 

- Download the [model](https://drive.google.com/file/d/1Fc9LA5KRoprYMlQ8gZmsddKHdDQW0Z8z/view?usp=drive_link) and place it in ./pretrained_models/   

#### Test

- We provide the preprocessd test data so that you can directly test using them. You can download the [final-test sampled input](https://drive.google.com/file/d/1M7xjlIWpHePxzErVc6zwJf8H_nmtZDoC/view?usp=drive_link).
- Please change the path in ``` config/test_demo.yml``` at first
- To test the pre-trained DemosaicFormer models of demosaicing for HybridEVS data on your own images,  you can use command line as following

```shell
python test.py -opt config/test_demo.yml
```

## Training

- Please change the path in ``` config/train_stage1.yml```  and ```config/train_stage2.yml```at first, then execute the following commands to start training:

``````
python -m torch.distributed.launch --nproc_per_node=6 --master_port=4321 train_mipi_pgst.py -opt config/train_stage1.yml --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=6 --master_port=4321 train_mipi.py -opt config/train_stage2.yml --launcher pytorch
``````

## Citation

If you find our work useful in your research, please consider citing:

```
@InProceedings{Xu_2024_CVPR,
    author    = {Xu, Senyan and Sun, Zhijing and Zhu, Jiaying and Zhu, Yurui and Fu, Xueyang and Zha, Zheng-Jun},
    title     = {DemosaicFormer: Coarse-to-Fine Demosaicing Network for HybridEVS Camera},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {1126-1135}
}
```

## Contact

Should you have any question, please contact [syxu@mail.ustc.edu.cn](syxu@mail.ustc.edu.cn)

**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox.
