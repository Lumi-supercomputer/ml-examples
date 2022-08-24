These are notes on running TensorFlow+Horovod with containers on LUMI using images from [`rocm/tensorflow`](https://hub.docker.com/r/rocm/tensorflow) in DockerHub.

In order to run the containers, some setup needs to be done. Please find more details [here](setup-tensorflow_rocm5.0-tf2.7-dev.md). Instructions for [rocm/tensorflow:rocm5.0-tf2.7-dev](https://hub.docker.com/layers/tensorflow/rocm/tensorflow/rocm5.0-tf2.7-dev/images/sha256-664fbd3e38234f5b4419aa54b2b81664495ed0a9715465678f2bc14ea4b7ae16) are given as an example, for the other tag the process shold be similar.

The containers need to be run with OpenMPI's `mpirun`, which can be installed locally by the user using an easyconfig from
[here](https://github.com/Lumi-supercomputer/LUMI-EasyBuild-contrib/blob/main/easybuild/easyconfigs/o/OpenMPI/).
Information on building software with EasyBuild on LUMI can be found [here](https://docs.lumi-supercomputer.eu/software/installing/easybuild/).

Once everything has been setup, the container can be run with [`run-singularity.sh`](run-singularity.sh). The horovd scripts used is
[tensorflow2_synthetic_benchmark.py](https://raw.githubusercontent.com/horovod/horovod/v0.24.2/examples/tensorflow2/tensorflow2_synthetic_benchmark.py).
When running with Horovod, one has to set one rank per GPU. Doesn't matter whether it's multiple GPUs on a single node
or multiple GPUs over multiple nodes.

With the batch script [`run-singularity-aws.sh`](run-singularity-aws.sh), the container can be run using the
[aws-ofi-rccl](https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl) plugin. It requires the to install the `aws-ofi-rccl` plugin and `rccl` with
EasyBuild. Please, find the recipes on the
[repository](https://github.com/Lumi-supercomputer/LUMI-EasyBuild-contrib/blob/main/easybuild/easyconfigs) (same as above for OpenMPI).

## Notes
In some cases, since the data is always the same, the computations are cached and the performance results do not make
sense. That can be solved by creating the random data in every iteration:
```patch
66,67c66,67
< data = tf.random.uniform([args.batch_size, 224, 224, 3])
< target = tf.random.uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)
---
> # data = tf.random.uniform([args.batch_size, 224, 224, 3])
> # target = tf.random.uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)
76a77,78
>         data = tf.random.uniform([args.batch_size, 224, 224, 3])
>         target = tf.random.uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)```
```
