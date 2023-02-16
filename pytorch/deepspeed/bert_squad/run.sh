#!/bin/bash
#SBATCH --job-name=pt-cnn
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --time=0:10:0
#SBATCH --exclusive
#SBATCH --partition standard-g
#SBATCH --account=<project>
#SBATCH --gpus-per-node=8

module load LUMI/22.08
module load partition/G
module load singularity-bindings
module load aws-ofi-rccl
module load OpenMPI

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
export SINGULARITYENV_LD_LIBRARY_PATH=/opt/ompi/lib:${EBROOTAWSMINOFIMINRCCL}/lib:/opt/cray/xpmem/2.4.4-2.3_9.1__gff0e1d9.shasta/lib64:${SINGULARITYENV_LD_LIBRARY_PATH}

mpirun singularity exec -B"/appl:/appl" \
                        -B"$SCRATCH:$SCRATCH" \
                        --pwd $HOME/git_/ml-examples/pytorch/deepspeed/bert_squad \
                        $SCRATCH/pytorch_rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1.sif bash -c "
                        . ds-env/bin/activate; python 3_squad_bert_deepspeed.py --deepspeed_config ds_config.json"
