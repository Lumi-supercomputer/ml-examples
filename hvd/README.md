# Notes

## [lumi-eap] Running a horovod example within a container 
Using `amdih/tensorflow:rocm4.2-tf2.5-dev` from the [AMD Infinity Hub](https://www.amd.com/en/technologies/infinity-hub/tensorflow) with singularity.

For internode communication, NCCL seems to need the following
```bash
export NCCL_IB_HCA=hsn0
```
Another posibility is `NCCL_IB_HCA=mlx5_0`, which what NCCL finds by default if nothing is set. That works for multiple GPUs on a single node,
but doesn't seem to work for multiple node. It may hang or crash with an 'unhandled error'.

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

### Notes
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
