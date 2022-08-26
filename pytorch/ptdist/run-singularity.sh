#!/bin/bash
#SBATCH --job-name=cnn-pytorch
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=0:10:0
#SBATCH --partition gpu
#SBATCH --account=<account>

module load LUMI/22.08
module load partition/G
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET_GDR_LEVEL=3

srun singularity exec deepspeed_rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0.sif python cnn_distr.py
