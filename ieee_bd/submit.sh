#!/bin/bash -l
# NODES=1
# SLURM SUBMIT SCRIPT
#SBATCH --account=kuex0005
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --time=2-00:00:00
#SBATCH --job-name=ibd4
#SBATCH --partition=gpu
#SBATCH --output=ibd4.%j.out
#SBATCH --error=ibd4.%j.err
# auto resubmit job 90 seconds before training ends due to wall time
# SBATCH --signal=SIGUSR1@90


# debugging flags (optional)
# export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=1
# export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
module purge
module load cuda/11.3 miniconda/3 gcc/9.3

# activate conda env
conda activate ai4ex

# run script from above
for i in $(lsof /dev/nvidia0 | grep python | awk '{print $2}' | sort -u); do kill -9 $i; done

srun python ieee_bd/main.py --nodes 1 --gpus 4 --blk_type swin2unet3d --stages 4 --patch_size 2 --sf 128 --nb_layers 4  --use_neck --use_all_region --lr 1e-4 --optimizer adam --scheduler plateau --merge_type both  --mlp_ratio 2 --decode_depth 2 --precision 32 --epoch 100 --batch-size 4 --augment_data  --constant_dim --workers 12 --use_static --use_all_products
