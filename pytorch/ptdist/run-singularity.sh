#!/bin/bash -l

#SBATCH --job-name=test-pt
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=eap
#SBATCH --account=<project>
#SBATCH --gres=gpu:mi100:4

export NCCL_DEBUG=INFO
export SINGULARITY_BIND='/opt/cray/libfabric/1.11.0.4.106/lib64/libfabric.so.1:/ext_cray/libfabric.so.1,/opt/cray/pe/lib64/libpmi2.so.0:/ext_cray/libpmi2.so.0,/opt/cray/pe/mpich/8.1.8/ofi/gnu/9.1/lib/libmpi_gnu_91.so.12:/ext_cray/libmpi_gnu_91.so.12,/usr/lib64/liblustreapi.so:/ext_cray/liblustreapi.so,/usr/lib64/libatomic.so.1:/usr/lib64/libatomic.so.1,/usr/lib64/libpals.so.0:/usr/lib64/libpals.so.0,/etc/libibverbs.d:/etc/libibverbs.d,/usr/lib64/libibverbs.so.1:/usr/lib/libibverbs.so.1,/var/opt/cray:/var/opt/cray,/appl:/appl,/opt/cray:/opt/cray,/usr/lib64/librdmacm.so.1:/ext_cray/librdmacm.so.1,/lib64/libtinfo.so.6:/ext_cray/libtinfo.so.6,/usr/lib64/libibverbs.so.1:/usr/lib/x86_64-linux-gnu/libibverbs.so.1'
export SINGULARITYENV_LD_LIBRARY_PATH="/etc/libibverbs.d:/var/opt/cray/pe/pe_images/aocc-compiler/usr/lib64/libibverbs:/usr/lib64:/opt/cray/pe/lib64:/opt/gcc/10.2.0/snos/lib64:/ext_cray:/usr/lib64:/opt/cray/pe/lib64:/opt/cray/xpmem/2.2.40-2.1_3.9__g3cf3325.shasta/lib64:${LD_LIBRARY_PATH}"

srun singularity exec $SCRATCH/pytorch_rocm4.2_ubuntu18.04_py3.6_pytorch_1.9.0.sif \
                 bash -c '
                 cd $HOME/git_/ml-examples/pytorch/ptdist
		 python cnn_distr.py'
