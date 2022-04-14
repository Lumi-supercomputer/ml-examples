#!/bin/bash -l

#SBATCH --job-name=test-tf
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=eap
#SBATCH --account=<project>
#SBATCH --gres=gpu:mi100:4

export NCCL_DEBUG=INFO

# modify Slurm Cluster Resolver to properly parse the node names
# * this needs the `python-hostlist` package
export CUSTOM_SINGULARITY_BIND="${HOME}/git_/ml-examples/tensorflow/tfdist/slurm_cluster_resolver.py:/usr/local/lib/python3.9/dist-packages/tensorflow/python/distribute/cluster_resolver/slurm_cluster_resolver.py"

# [tf-2.5.0] avoid segmentation fault with MultiWorkerMirroredStrategy+NCCL https://github.com/tensorflow/tensorflow/issues/50926
# export CUSTOM_SINGULARITY_BIND="${HOME}/git_/tensorflow-training/cross_device_ops.py:/usr/local/lib/python3.9/dist-packages/tensorflow/python/distribute/cross_device_ops.py,${CUSTOM_SINGULARITY_BIND}"

export SINGULARITY_BIND="/opt/cray/libfabric/1.11.0.4.106/lib64/libfabric.so.1:/ext_cray/libfabric.so.1,/opt/cray/pe/lib64/libpmi2.so.0:/ext_cray/libpmi2.so.0,/opt/cray/pe/mpich/8.1.8/ofi/gnu/9.1/lib/libmpi_gnu_91.so.12:/ext_cray/libmpi_gnu_91.so.12,/usr/lib64/liblustreapi.so:/ext_cray/liblustreapi.so,/usr/lib64/libatomic.so.1:/usr/lib64/libatomic.so.1,/usr/lib64/libpals.so.0:/usr/lib64/libpals.so.0,/etc/libibverbs.d:/etc/libibverbs.d,/usr/lib64/libibverbs.so.1:/usr/lib/libibverbs.so.1,/var/opt/cray:/var/opt/cray,/appl:/appl,/opt/cray:/opt/cray,/usr/lib64/librdmacm.so.1:/usr/lib/librdmacm.so.1,/lib64/libtinfo.so.6:/ext_cray/libtinfo.so.6,/usr/lib64/libnl-3.so.200:/lib/x86_64-linux-gnu/libnl-3.so.200,/usr/lib64/libnl-route-3.so.200:/usr/lib/x86_64-linux-gnu/libnl-route-3.so.200,${CUSTOM_SINGULARITY_BIND}"

export SINGULARITYENV_LD_LIBRARY_PATH="/etc/libibverbs.d:/var/opt/cray/pe/pe_images/aocc-compiler/usr/lib64/libibverbs:/usr/lib64:/opt/cray/pe/lib64:/opt/gcc/10.2.0/snos/lib64:/ext_cray:/usr/lib64:/opt/cray/pe/lib64:/opt/cray/xpmem/2.2.40-2.1_3.9__g3cf3325.shasta/lib64:${LD_LIBRARY_PATH}"

# needed by the Slurm Cluster Resolver
export LUMI_VISIBLE_DEVICES=$(seq --separator="," 0 $(($SLURM_GPUS_ON_NODE - 1)))

# exports for the SlurmClusterResolver that were not defined in the system
export SLURM_STEP_NUM_TASKS=$SLURM_NTASKS
export SLURM_STEP_NODELIST=$SLURM_JOB_NODELIST
export SLURM_STEP_TASKS_PER_NODE=$SLURM_TASKS_PER_NODE

srun singularity exec $SCRATCH/tensorflow_rocm5.0-tf2.7-dev.sif \
                      bash -c '
                      export LANG=en_US.utf-8;
		      export LC_ALL=en_US.utf-8;
                      . $HOME/tf_rocm5_env/bin/activate;
                      python $HOME/git_/ml-examples/tensorflow/tfdist/tfdist_synthetic_benchmark.py --batch-size=256'
                      # python $HOME/git_/ml-examples/tensorflow/tfdist/tfdist_keras_synthetic_benchmark.py --batch-size=256'
		      # python $HOME/git_/ml-examples/tensorflow/tfdist/linear-model_keras.py'

echo '' | cat - $0 | head -n -1  # copy the file content to the output
