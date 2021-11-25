# Weather4Cast2021-SwinUNet3D (AI4EX Team)

## Table of Content
* [General Info](#general-info)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
* [Inference](#inference)
* [Accessing the trained Checkpoint](#checkpoint)

## General Info
This resipository contains our code submitted to IEEE Big Data Weather4Cast competition (https://www.iarai.ac.at/weather4cast/2021-competition/challenge/#2021-ieee-big-data-cup)
This work is made available under the attached license

## Requirements
This resipository depends on the following packages availability
- Pytorch Lightning
- timm
- torch_optimizer
- pytorch_model_summary
- einops

## Installation:
```
unzip folder.zip
cd folder
conda create --name swinunet3d_env python=3.6
conda activate swinunet3d_env
conda install pytorch=1.9.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```
## Usage
- a.1)train from scratch (together with inference predictions) SwinUNet3d-1
    ```
    python ieee_bd/main.py --nodes 1 --gpus 4 --blk_type swin2unet3d --stages 4 --patch_size 2 --sf 160 --nb_layers 6  --use_neck --use_all_region --lr 1e-4 --optimizer adam --scheduler plateau --merge_type both  --mlp_ratio 2 --decode_depth 2 --precision 32 --epoch 100 --batch-size 4 --augment_data  --constant_dim --workers 12 --get_prediction --use_static --use_all_products
    ```
 
 - a.2)train from scratch. SwinUNet3d-2
    ```
    python ieee_bd/main.py --nodes 1 --gpus 4 --blk_type swin2unet3d --stages 4 --patch_size 2 --sf 128 --nb_layers 4  --use_neck --use_all_region --lr 1e-4 --optimizer adam --scheduler plateau --merge_type both  --mlp_ratio 2 --decode_depth 2 --precision 32 --epoch 100 --batch-size 4 --augment_data  --constant_dim --workers 12 --use_static --use_all_products
    ```
    
- b) fine tune a model from a checkpoint
    ```
    python ieee_bd/main.py --nodes 1 --gpus 4 --blk_type swin2unet3d --stages 4 --patch_size 2 --sf 128 --nb_layers 4  --use_neck --use_all_region --lr 1e-4 --optimizer adam --scheduler plateau --merge_type both  --mlp_ratio 2 --decode_depth 2 --precision 32 --epoch 100 --batch-size 4 --augment_data  --constant_dim --workers 12 --use_static --use_all_products --mode train --name ALL_real_swin2unet3d_4125520 --time-code 20211027T171154 --initial-epoch 33
    ```
 
## Inference
- a) To generate predictions using our trained model
```
python ieee_bd/main.py --gpus 1 --mode test --name ALL_real_swin2unet3d_5207312 --time-code 20211027T104444 --initial-epoch 34
```


## Accessing the trained checkpoint
Our trained model can be downloaded from https://drive.google.com/drive/folders/1O0KVMl_LCEwqRw9or-dp3U3_7r9YJF9C?usp=sharing
- SwinUNet3D-1 :ALL_real_swin2unet3d_5207312\20211027T104444\checkpoints\epoch=34-val_loss=0.683052.ckpt
- SwinUNet3D-2 :ALL_real_swin2unet3d_4125520\20211027T171154\checkpoints\epoch=33-val_loss=0.686488.ckpt
