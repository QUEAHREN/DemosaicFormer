## DemosaicFormer: Coarse-to-Fine Demosaicing Network for HybridEVS Camera

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

## Data Preprocess

Please change the path in ``` data_preprocess.py``` and use command line as following to convert the bin files to RGB images:

```
python data_preprocess.py
```

## Evaluation

#### Pretrained Model Download 

- Download the [model](https://drive.google.com/file/d/1Fc9LA5KRoprYMlQ8gZmsddKHdDQW0Z8z/view?usp=drive_link) and place it in ./pretrained_models/   

#### Test

- Please change the path in ``` config/test_demo.yml``` at first
- To test the pre-trained Restormer models of demosaicing for HybridEVS data on your own images,  you can use command line as following

```shell
python test.py -opt config/test_demo.yml
```



## Training

- Please change the path in ``` config/train_stage1.yml```  and ```config/train_stage2.yml```at first, then execute the following commands to start training:

``````
python -m torch.distributed.launch --nproc_per_node=6 --master_port=4321 train_mipi_pgst.py -opt config/train_stage1.yml --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=6 --master_port=4321 train_mipi.py -opt config/train_stage2.yml --launcher pytorch
``````

## Contact

Should you have any question, please contact [syxu@mail.ustc.edu.cn](syxu@mail.ustc.edu.cn)

**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox.

