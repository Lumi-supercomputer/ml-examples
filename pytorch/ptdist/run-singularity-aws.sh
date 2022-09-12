#!/bin/bash
#SBATCH --job-name=cnn-pytorch
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=0:10:0
#SBATCH --partition gpu
#SBATCH --account=<account>

module load LUMI/22.08
module load partition/G
module load singularity-bindings
module load rccl
module load aws-ofi-rccl
module load OpenMPI

export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET_GDR_LEVEL=3
export SINGULARITYENV_LD_LIBRARY_PATH=/opt/ompi/lib:/opt/rocm-5.0.1/lib:$EBROOTAWSMINOFIMINRCCL/lib:/opt/cray/xpmem/2.4.4-2.3_9.1__gff0e1d9.shasta/lib64:$SINGULARITYENV_LD_LIBRARY_PATH

mpirun singularity exec \
                 -B"/appl:/appl" 
                 -B"$EBROOTRCCL/lib/librccl.so.1.0:/opt/rocm-5.0.1/rccl/lib/librccl.so.1.0.50001" \
                 deepspeed_rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0.sif \
                 bash -c '
                 . deepspeed_rocm5.0.1_env/bin/activate;
                 python cnn_distr.py'
