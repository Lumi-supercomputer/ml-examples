#!/bin/bash -l

#SBATCH --job-name=test-tf
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=eap
#SBATCH --account=<project>
#SBATCH --gres=gpu:mi100:4

module swap PrgEnv-cray PrgEnv-gnu
export PATH=$HOME/software/openmpi-4.1.2-install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/software/openmpi-4.1.2-install/lib:$LD_LIBRARY_PATH

export NCCL_DEBUG=INFO

export SINGULARITY_BIND="/opt/cray/libfabric/1.11.0.4.106/lib64/libfabric.so.1:/ext_cray/libfabric.so.1,/opt/cray/pe/lib64/libpmi2.so.0:/ext_cray/libpmi2.so.0,/opt/cray/pe/mpich/8.1.8/ofi/gnu/9.1/lib/libmpi_gnu_91.so.12:/ext_cray/libmpi_gnu_91.so.12,/usr/lib64/liblustreapi.so:/ext_cray/liblustreapi.so,/usr/lib64/libatomic.so.1:/usr/lib64/libatomic.so.1,/usr/lib64/libpals.so.0:/usr/lib64/libpals.so.0,/etc/libibverbs.d:/etc/libibverbs.d,/usr/lib64/libibverbs.so.1:/usr/lib/libibverbs.so.1,/var/opt/cray:/var/opt/cray,/appl:/appl,/opt/cray:/opt/cray,/usr/lib64/librdmacm.so.1:/ext_cray/librdmacm.so.1,/lib64/libtinfo.so.6:/ext_cray/libtinfo.so.6,${HOME}/software/openmpi-4.1.2-install:/ext_openmpi"

export SINGULARITYENV_LD_LIBRARY_PATH="/etc/libibverbs.d:/var/opt/cray/pe/pe_images/aocc-compiler/usr/lib64/libibverbs:/usr/lib64:/opt/cray/pe/lib64:/opt/gcc/10.2.0/snos/lib64:/ext_cray:/usr/lib64:/opt/cray/pe/lib64:/opt/cray/xpmem/2.2.40-2.1_3.9__g3cf3325.shasta/lib64:/ext_openmpi/lib:${LD_LIBRARY_PATH}"

mpirun singularity exec $SCRATCH/tensorflow_rocm5.0-tf2.7-dev.sif \
                      bash -c '
                      export LANG=en_US.utf-8;
					  export LC_ALL=en_US.utf-8;
                      . $HOME/tf_rocm5_env/bin/activate;
					  python $HOME/git_/ml-examples/tensorflow/hvd/tensorflow2_synthetic_benchmark.py --batch-size=256'
					  # python $HOME/git_/ml-examples/tensorflow/hvd/tensorflow2_keras_synthetic_benchmark.py --batch-size=256'

echo '' | cat - $0 | head -n -1  # copy the file content to the output
