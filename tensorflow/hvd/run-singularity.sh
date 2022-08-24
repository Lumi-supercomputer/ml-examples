#!/bin/bash
#SBATCH --job-name=cnn-hvd
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=0:10:0
#SBATCH --partition gpu
#SBATCH --account=<account>

module load LUMI/22.08
module load partition/G
module load OpenMPI    # OpenMPI needs to be installed locally by the user

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

mpirun -np 32 singularity exec tensorflow_rocm5.0-tf2.7-dev.sif \
              bash -c '~/tf2.7_rocm5.0_env/bin/activate; python tf2_hvd_synthetic_benchmark.py --batch-size=512'
