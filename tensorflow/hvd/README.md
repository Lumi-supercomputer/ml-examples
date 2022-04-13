# Notes

## [lumi-eap] Running a horovod example within a container 
Using `amdih/tensorflow:rocm4.2-tf2.5-dev` from the [AMD Infinity Hub](https://www.amd.com/en/technologies/infinity-hub/tensorflow) with singularity.
It can be pulled with
```bash
singularity pull docker://amdih/tensorflow:rocm4.2-tf2.5-dev
```

There are some options for `NCCL_IB_HCA`. For internode communication there is `hsn0`, which doesn't give a good scaling.
Another posibility is `NCCL_IB_HCA=mlx5_0`, which what NCCL finds by default if nothing is set. with `NCCL_DEBUG=INFO`, the log has
```
nid000014:57586:57614 [0] NCCL INFO Bootstrap : Using nmn0:10.252.1.70<0>
nid000014:57586:57614 [0] NCCL INFO NET/Plugin : No plugin found (librccl-net.so), using internal implementation
nid000014:57586:57614 [0] NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE ; OOB nmn0:10.252.1.70<0>
nid000014:57586:57614 [0] NCCL INFO Using network IB
```

The cotainer needs some setup before running
```
export SINGULARITY_BIND='/opt/cray/libfabric/1.11.0.4.106/lib64/libfabric.so.1:/ext_cray/libfabric.so.1,/opt/cray/pe/lib64/libpmi2.so.0:/ext_cray/libpmi2.so.0,/opt/cray/pe/mpich/8.1.8/ofi/gnu/9.1/lib/libmpi_gnu_91.so.12:/ext_cray/libmpi_gnu_91.so.12,/usr/lib64/liblustreapi.so:/ext_cray/liblustreapi.so,/usr/lib64/libatomic.so.1:/usr/lib64/libatomic.so.1,/usr/lib64/libpals.so.0:/usr/lib64/libpals.so.0,/etc/libibverbs.d:/etc/libibverbs.d,/usr/lib64/libibverbs.so.1:/usr/lib/libibverbs.so.1,/var/opt/cray:/var/opt/cray,/appl:/appl,/opt/cray:/opt/cray,/usr/lib64/librdmacm.so.1:/xext_cray/librdmacm.so.1,/lib64/libtinfo.so.6:/ext_cray/libtinfo.so.6,$HOME/software/openmpi-4.1.2-install:/ext_openmpi'
```
The last mount is the OpenMPI install dir. That needs to be installed locally.

Here `/usr/lib64/librdmacm.so.1` has been mounted deliberately at `xext_cray` instead of `ext_cray`.
That's only to have that library around for further investigation without using it in the container.
With or without it, the [rccl-test](https://github.com/ROCmSoftwarePlatform/rccl-tests) benchmark gives the same performance,
however tensorflow hangs.

```
export SINGULARITYENV_LD_LIBRARY_PATH='/etc/libibverbs.d:/var/opt/cray/pe/pe_images/aocc-compiler/usr/lib64/libibverbs:/usr/lib64:/opt/cray/pe/lib64:/opt/gcc/10.2.0/snos/lib64:/ext_cray:/usr/lib64:/opt/cray/pe/lib64:/opt/cray/xpmem/2.2.40-2.1_3.9__g3cf3325.shasta/lib64:/ext_openmpi/lib:$LD_LIBRARY_PATH'
```

The container needs to be run with `mpirun`, which needs to be installed locally
```bash
mpirun singularity exec tensorflow_rocm4.2-tf2.5-dev.sif python tensorflow2_synthetic_benchmark.py --batch-size=128
```

When running with Horovod, one has to set one rank per GPU. Doesn't matter whether it's multiple GPUs on a single node
or multiple GPUs over multiple nodes.

Because of the need for `mpirun`, the allocation setup needs to be done with special care. For instance
```bash
salloc -peap -N2 -Aproject_462000002 --gres=gpu:mi100:2 --time 1:00:00 --n=4
```
which makes 4 GPUs available, may create 1 rank on a one node and 3 ranks on the other node.

It looks like it's better to use `--ntasks-per-node`, instead of `--ntasks`/`-n`.
```bash
salloc -peap -N2 -Aproject_462000002 --gres=gpu:mi100:2 --time 1:00:00 --ntasks-per-node=2
```
This gives the right number of ranks per node.

### Example: horovod - [`tensorflow2_synthetic_benchmark.py`](https://github.com/horovod/horovod/blob/19f2f2119db34b1be0d9f9aedb66106c9131da89/examples/tensorflow2/tensorflow2_synthetic_benchmark.py) - ResNet50 - batch-size=128

| Nodes / GPU-node |       1      |       2       |        4       |
|:------------:|:----------------:|:-------------:|:--------------:|
|      1       |  520.1 +-2.7     |  923.1 +-45.0 | 1783.4 +-113.9 |
|      2       |  870.9 +-35.7    | 1681.3 +-66.3 | 3271.5 +-226.4 |


### Notes
 - Had to pip install `psutil` and `cloudpickle` since the container didn't have it.
 - In some cases with ResNet50, there's a rocblas error when using a batch size of 256. It could eaily be because of a too large batch:
 ```bash
 failed to run ROCBLAS routine rocblas_sgemm: rocblas_status_internal_error
 ```


## [grenoble] Running a horovod example within a container 
We are using the image [rocm/tensorflow:rocm4.1-tf2.3-dev](https://hub.docker.com/layers/rocm/tensorflow/rocm4.1-tf2.3-dev/images/sha256-0f369142a95872bef829fc61256a628828e0427284ff8f2f8d1f821023aa5b4c?context=explore) and running with singularity.

>> Before running it's necessary to install `psutil` and `cloudpickle` inside the container.
```bash
pip install --prefix $HOME/container-pip/ psutil cloudpickle
```

On the grenoble nodes,
[the horovod example](https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py) can be run 
interactively - login in the container interactively and run the script as follows: 
```bash
mpirun -n 2 python tensorflow2_keras_synthetic_benchmark.py --model=ResNet50 --batch-size=256 --num-batches-per-iter=10
```
Doing this, each rank will get a GPU.

Non-interactively it can be run with
```bash
singularity exec \
            tensorflow_rocm4.1-tf2.3-dev.sif \
            bash -c 'cd $HOME/tf-examples/;
                     export PYTHONPATH=$HOME/container-pip/lib/python3.6/site-packages:$PYTHONPATH;
                     mpirun -n 2 python hvd/tensorflow2_keras_synthetic_benchmark.py \
                            --model=ResNet50 \
                            --batch-size=256 \
                            --num-batches-per-iter=10'
```
Note that the command doens't have the flag `--rocm` for `exec`. Or with
```bash
singularity exec --rocm \
            tensorflow_rocm4.1-tf2.3-dev.sif \
            bash -c '. /etc/profile.d/horovod.sh; \
                     export HOME=/home/<username>; \
                     cd $HOME/tf-examples/;
                     export PYTHONPATH=$HOME/container-pip/lib/python3.6/site-packages:$PYTHONPATH;
                     mpirun -n 2 python hvd/tensorflow2_keras_synthetic_benchmark.py \
                            --model=ResNet50 \
                            --batch-size=256 \
                            --num-batches-per-iter=10'
```
Here we export `HOME` to `/home/<username>` since the sourcing of `/etc/profile.d/horovod.sh` sets `HOME` to `/root` which messes with
the configuration of MIOPen's cache database directory.


## [grenoble] Running with OpenMPI's `mpirun`
Running singularity with `srun singularity exec ...` is not working. It could be that the openmpi installation in the container doesn't
support slurm (not sure).

It can be run, however, with openmpi (installed locally) with the following batch script:
```bash
#!/bin/bash -l

#SBATCH --job-name=test
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=2

# module switch PrgEnv-cray/8.0.0 PrgEnv-gnu

export PATH=$HOME/software/openmpi-4.1.1/install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/software/openmpi-4.1.1/install/bin:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO

mpirun singularity exec \
            ~/tensorflow_rocm4.1-tf2.3-dev.sif \
            bash -c 'cd $HOME/tf-examples/;
                     export PYTHONPATH=$HOME/container-pip/lib/python3.6/site-packages:$PYTHONPATH;
                     python hvd/tensorflow2_keras_synthetic_benchmark.py \
                            --model=ResNet50 \
                            --batch-size=256 \
                            --num-batches-per-iter=10'
```
