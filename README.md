This is the code repository implementing the paper "RigNet: Neural Rigging for Articulated Characters" published on SIGGRAPH 2020 [[Project page]](https://zhan-xu.github.io/rig-net/).

**[2023.02.17]** About dataset: If you are from a research lab and interested in the dataset for non-commercial, research-only purposes, please send a request email to me at zhanxu@cs.umass.edu.

**[2021.07.20]** Another add-on for Blender, 
implemented by @[L-Medici](https://github.com/L-Medici). Please check the Github [link](https://github.com/L-Medici/Rignet_blender_addon).


**[2020.11.23]** There is now a great add-on for Blender based on our work, 
implemented by @[pKrime](https://github.com/pKrime). Please check the Github [link](https://github.com/pKrime/brignet), and the video [demo](https://www.youtube.com/watch?v=ueLlS3IoeGY&feature=youtu.be). 

## Dependecy and Setup

The project is developed on Ubuntu 16.04 with cuda10.0 and cudnn7.6.3. 
It has also been successfully tested on Windows 10.
On both platforms, we suggest to use conda virtual environment. 

#### For Linux user

**[2023.05.21]** I have tested the code on Ubuntu 22.04, with cuda 11.3. The following commands have been updated. 

```
conda create --name rignet python=3.7
conda activate rignet
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch

# load cuda_toolkit 11.3
export PATH=/usr/local/cuda_11.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda_11.3/lib64

# require g++ < 10 to install the following pytorch geometric version.
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html # this take a while
pip install torch-geometric==1.7.2

pip install numpy scipy matplotlib tensorboard open3d==0.9.0 opencv-python "rtree>=0.8,<0.9" trimesh[easy]  # Make sure to install open3d 0.9.0.
```

#### For Windows user

The code has been tested on Windows 11 with Python 3.11, Pytorch 2.2.1, and Cuda 12.4 
``` powershell
# install pytorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# install torch geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.1+cu121.html

pip install torch_geometric==2.5.1

# install other dependencies
pip install tensorboard==2.16.2 opencv-python==4.9.0.80 rtree==1.2.0 trimesh==4.2.0 open3d==0.18.0
```




## Quick start
We provide a script for quick start. First download our trained models from [here](https://umass-my.sharepoint.com/:u:/g/personal/zhanxu_umass_edu/EYKLCvYTWFJArehlo3-H2SgBABnY08B4k5Q14K7H1Hh0VA). 
Put the checkpoints folder into the project folder. 

Check and run quick_start.py. We provide some examples in this script. 
Due to randomness, the results might be slightly different among each run. 
Generally you will get the results similar to the ones shown below:

![results figure](quick_start/quick_start.png)

If you want to try your own models, remember to simplify the meshes so that 
the remeshed ones have vertices between 1K to 5K. I use quadratic edge collapse in MeshLab for this. 
Please name the simplified meshed as *_remesh.obj.

The predicted rigs are saved as *_rig.txt. You can combine the OBJ file and *_rig.txt into FBX format by 
running maya_save_fbx.py provided by us in Maya using mayapy. (To use numpy in mayapy, download windows compiled numpy from [here](https://drive.google.com/drive/u/0/folders/0BwsYd1k8t0lEMjBCa2N1Z25KZXc) and put it in mayapy library folder. For example, mine is C:\Program Files\Autodesk\Maya2019\Python\Lib\site-packages)

## Data

Our dataset ModelsResource-RigNetv1 has 2,703 models. 
We split it into 80% for training (2,163‬ models), 10%
for validation (270 models), and 10% for testing. 
All models in fbx format can be downloaded [here](https://umass-my.sharepoint.com/:u:/g/personal/zhanxu_umass_edu/EVgpX4uZEVNLu8OjX9JRaFYBzOjfm4znndui29evdEfs-g).

To use this dataset in this project, pre-processing is performed. 
We put the pre-processed data [here](https://umass-my.sharepoint.com/:u:/g/personal/zhanxu_umass_edu/EaUH-2lI6-xOrJ0N9fDbZOABREt4ryEtQ64wmELF5SReTg), which consists of several sub-folders.

* obj: all meshes in OBJ format.
* rig_info: we store the rigging information into a txt file. Each txt file has four blocks. (1) Lines starting with "joint" define a joint with its 3D position. Each of joint line has four elements, which are joint_name, X, Y, and Z. (2) Line starting with "root" defines the name of root joint. (3) Lines starting with "hier" define the hierarchy of skeleton. Each hierarchy line has two elements, which are parent joint name and its child joint name. One parent joint can have multiple children joints. (4) Lines starting with "skin" define the skinning weights. Each skinning line follows the format as vertex_id, bind_joint_name_1, bind_weight_1, bind_joint_name_2, bind_weight_2 ... The vertex_id follows the vertice order in obj files in the above obj folder.
* obj_remesh: This folder contains the obj files of the remeshed models. Meshes with fewer than 1K vertices were subdivided, and those with more than 5K vertices were simplified; as a result all training and test meshes contained between 1K and 5K vertices.
* rig_info_remesh: Rigging information files corresponding to the remeshed obj. Joints, hierarchy and root are the same. The skinning is recalculated based on nearest neighbor from each remeshed vertex to original vertices.
* pretrain_attention: Pre-calculated supervision to pretrin the attention module, which are calculated by the script geometric_proc/compute_pretrain_attn.py. Each file is a N-by-1 text where N is the number of vertices corresponding to remeshed OBJ file, the i-th row stores the surpervision for vertex i.  
* volumetric_geodesic: Pre-calculated volumetric geodesic distance between each vertex-bones pair. The algorithm is an approaximation, which is implemented in geometric_proc/compute_volumetric_geodesic.py. Each file is an N-by-B numpy array where N is the number of vertices corresponding to remeshed OBJ file, B is the number of bones, and (i, j) stores the volumetric geodesic distance between vertex i and bone j. 
* vox: voxelized models used for inside/outside check. Obtained with [binvox](https://www.patrickmin.com/binvox/). The resolution of the grid is 88x88x88.

After downloading the pre-processed data, one needs to create the data directly used for training/testing, please check and run our script: 

`python gen_dataset.py`

Remember to change the root_folder to the directory you uncompress the pre-processed data.

## Training

Notes: As new features, we have three improvements from the paper: (1) To train the joint prediction module, now we pretrain both the regression module and the attention module, and then fine-tune them together with differentiable clustering. (2) We optimized the hyper-parameters in the fine-tuning step. (3) the input feature for skinning now includes another dimension per bone (--Lf), indicating whether this bone is a virtual leaf bone or not. (To enable control from the end-joints, we presume a virtual bone for them. Please check the code for more details.)

1. Joint prediction: 

    1.1 Pretrain regression module:
    `python -u run_joint_pretrain.py --train_folder='DATASET_DIR/train/' --val_folder='DATASET_DIR/val/' --test_folder='DATASET_DIR/test/' --checkpoint='checkpoints/pretrain_jointnet' --logdir='logs/pretrain_jointnet' --train_batch=6 --test_batch=6 --lr 5e-4 --schedule 50 --arch='jointnet'`

    1.2 Pretrain attention module:
    `python -u run_joint_pretrain.py --train_folder='DATASET_DIR/train/' --val_folder='DATASET_DIR/val/' --test_folder='DATASET_DIR/test/' --checkpoint='checkpoints/pretrain_masknet' --logdir='logs/pretrain_masknet' --train_batch=6 --test_batch=6 --lr 1e-4 --schedule 50 --arch='masknet'`

    1.3 Finetune two modules with a clustering module:
    `python -u run_joint_finetune.py --train_folder='DATASET_DIR/train/' --val_folder='DATASET_DIR/val/' --test_folder='DATASET_DIR/test/' --checkpoint='checkpoints/gcn_meanshift' --logdir='logs/gcn_meanshift' --train_batch=1 --test_batch=1 --jointnet_lr=1e-6 --masknet_lr=1e-6 --bandwidth_lr=1e-6 --epoch=50`

2. Connectivity prediction

    2.1 BoneNet:
    `python -u run_pair_cls.py --train_folder='DATASET_DIR/train/' --val_folder='DATASET_DIR/val/' --test_folder='DATASET_DIR/test/' --checkpoint='checkpoints/bonenet' --logdir='logs/bonenet' --train_batch=6 --test_batch=6 --lr=1e-3`

    2.2 RootNet:
    `python -u run_root_cls.py --train_folder='DATASET_DIR/train/' --val_folder='DATASET_DIR/val/' --test_folder='DATASET_DIR/test/' --checkpoint='checkpoints/rootnet' --logdir='logs/rootnet' --train_batch=6 --test_batch=6 --lr=1e-3`

3. Skinning prediction:
    `python -u run_skinning.py --train_folder='DATASET_DIR/train/' --val_folder='DATASET_DIR/val/' --test_folder='DATASET_DIR/test/' --checkpoint='checkpoints/skinnet' --logdir='logs/skinnet' --train_batch=4 --test_batch=4 --lr=1e-4 --Dg --Lf`

##  License
This project is under LICENSE-GPLv3.
