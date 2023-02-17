#!/bin/bash
#SBATCH --job-name=tf-distr-cnn
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:0
#SBATCH --partition standard-g
#SBATCH --account=<project>
#SBATCH --gpus-per-node=8

module load LUMI/22.08
module load partition/G
module load singularity-bindings
module load aws-ofi-rccl

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
export SINGULARITYENV_LD_LIBRARY_PATH=/openmpi/lib:/opt/rocm-5.4.1/lib:${EBROOTAWSMINOFIMINRCCL}/lib:/opt/cray/xpmem/2.4.4-2.3_9.1__gff0e1d9.shasta/lib64:$SINGULARITYENV_LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

srun singularity exec -B"/appl:/appl" \
                      -B"$SCRATCH:$SCRATCH" \
					  -B slurm_cluster_resolver.py:/usr/local/lib/python3.9/dist-packages/tensorflow/python/distribute/cluster_resolver/slurm_cluster_resolver.py \
                      --pwd $HOME/git_/ml-examples/tensorflow/tfdist \
                      $SCRATCH/tensorflow_rocm5.4.1-tf2.10-dev.sif bash -c \
                      ". tf-rocm5.4.1-env/bin/activate; python tf2_distr_synthetic_benchmark.py --batch-size=256"
