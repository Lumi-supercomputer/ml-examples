#!/bin/bash -l

#SBATCH --job-name=test-pt
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=eap
#SBATCH --account=project_462000002
#SBATCH --gres=gpu:mi100:4

module swap PrgEnv-cray PrgEnv-gnu
export PATH=$HOME/software/openmpi-4.1.2-install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/software/openmpi-4.1.2-install/lib:$LD_LIBRARY_PATH

export NCCL_DEBUG=INFO

mpirun singularity exec $SCRATCH/deepspeed_rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0.sif \
                   bash -c '
                   cd $HOME/git_/ml-examples/pytorch/deepspeed/cnn;
	               python cnn_deepspeed.py --deepspeed_config ds_config.json'
