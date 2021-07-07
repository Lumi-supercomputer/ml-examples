# Notes

## [grenoble] Running a horovod example within the container 
We are using the image [rocm/tensorflow:rocm4.1-tf2.3-dev](https://hub.docker.com/layers/rocm/tensorflow/rocm4.1-tf2.3-dev/images/sha256-0f369142a95872bef829fc61256a628828e0427284ff8f2f8d1f821023aa5b4c?context=explore) and running with singularity.

>> Before running it's necessary to install `psutil` and `cloudpickle` inside the container.
```bash
pip install --prefix $HOME/container-pip/ psutil cloudpickle
```

On the grenoble nodes, [the horovod example](https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py) can be run interactively - login in the container interactively and run the script as follows: 
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
Here we export `HOME` to `/home/<username>` since the sourcing of `/etc/profile.d/horovod.sh` sets `HOME` to `/root` which messes with the configuration of MIOPen's cache database directory.


## Running with OpenMPI's `mpirun`
Running singularity with `srun singularity exec ...` is not working. It could be that the openmpi installation in the container doesn't support slurm (not sure).

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
