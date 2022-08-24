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
module load singularity-bindings
module load rccl
module load aws-ofi-rccl
module load OpenMPI

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export SINGULARITYENV_LD_LIBRARY_PATH=/openmpi/lib:/opt/rocm-5.0.0/lib:$EBROOTAWSMINOFIMINRCCL/lib:/opt/cray/xpmem/2.4.4-2.3_9.1__gff0e1d9.shasta/lib64:$SINGULARITYENV_LD_LIBRARY_PATH

mpirun singularity exec -B"/appl:/appl" \
                        -B"$EBROOTRCCL/lib/librccl.so.1.0:/opt/rocm-5.0.0/rccl/lib/librccl.so.1.0.50000" \
                        tensorflow_rocm5.0-tf2.7-dev.sif \
                        bash -c '. ~/tf2.7_rocm5.0_env/bin/activate; python tf2_hvd_synthetic_benchmark.py --batch-size=512'
