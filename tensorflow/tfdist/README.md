# TensorFlow on LUMI

[TensorFlow](https://www.tensorflow.org) is an end-to-end open source platform for machine learning.

TensorFlow can be installed by the users with
```
module load cray-python
pip install tensorflow-rocm
```

TensorFlow can be run within containers as well. In particular, containers from the images provided by [AMD on DockerHub](https://hub.docker.com/r/rocm/tensorflow), which include [Horovod](https://horovod.readthedocs.io/en/stable/).
Those images are updated frequently and make it possible to try TensorFlow with recent ROCm versions. 
Another point in favor of using containers, is that TensorFlow's installation directory can be quite large both in terms of storage size and number of files.

## Running TensorFlow within containers

The images can be fetched with singularity:
```bash
SINGULARITY_TMPDIR=$SCRATCH/tmp-singularity singularity pull docker://rocm/tensorflow:rocm5.4.1-tf2.10-dev
```
This will create an image file named `tensorflow_rocm5.4.1-tf2.10-dev.sif` on the directory where the command was run. After the image has been pulled, the directory `$SCRATCH/tmp-singularity singularity` can be removed.

### Installing other packages along the container's TensorFlow installation

Often we may need to install other packages to be used along TensorFlow.
This can be done by running the container interactively and creating a virtual environment in a host directory, for instance, your `$HOME`.
Unfortunately, the images from [rocm/tensorflow](https://hub.docker.com/r/rocm/tensorflow) miss the `ensurepip` module, which is part of the python standard library. Without that module it's not possible to create a virtual environment.
However `ensurepip` can be fetched from the [cpython repository](https://github.com/python/cpython/tree/main/Lib/ensurepip) using [this script](download-ensurepip-py3.9.sh) and then made available in the container by mounting it on `usr/lib/python3.9/ensurepip`.

As an example, let's do that to install the package `psutil`, which is needed by Horovod:
```bash
$> singularity exec -B ensurepip:/usr/lib/python3.9/ensurepip tensorflow_rocm5.4.1-tf2.10-dev.sif bash
Singularity> python -m venv tf_rocm5.4.1_env --system-site-packages
Singularity> . tf_rocm5.4.1_env/bin/activate
(tf_rocm5.4.1_env) Singularity> pip install psutil
```
Now when running the container, the virtual environment must be activated before calling python. Once the virtual environment has been created, the `ensurepip` module is no longer needed.

## Multi-GPU training with Horovod

> We have seen Horovod crashing with a `segmentation fault` error when using the image [rocm/tensorflow:rocm5.4.1-tf2.10-dev](https://hub.docker.com/layers/rocm/tensorflow/rocm5.4.1-tf2.10-dev/images/sha256-cfe54bcee70f7b0fa50dcfd9bf6b46e177366b2747a795acfa71647ee83c6e95?context=explore). In that case, Horovod must be reinstalled (on a virtual environment). That can be done with [this script](install-horovod.sh).

### RCCL

The communication between LUMI's GPUs during training with TensorFlow is done via [RCCL](https://github.com/ROCmSoftwarePlatform/rccl), which is a library of  collective communication routines for GPUs. RCCL works out of the box on LUMI's, however, a special plugin is required so it can take advantage of the [Slingshot interconnect](https://www.hpe.com/emea_europe/en/compute/hpc/slingshot-interconnect.html). That's the [`aws-ofi-rccl`](https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl) plugin, which is a library that can be used as a back-end for RCCL to interact with the interconnect via libfabric.

The `aws-ofi-rccl` plugin can be installed by the user with EasyBuild:
```bash
module load LUMI/22.08 partition/G
module load EasyBuild-user
eb aws-ofi-rccl-66b3b31-cpeGNU-22.08.eb -r
```
Once installed, loading the module `aws-ofi-rccl` will add the path to the library to the `LD_LIBRARY_PATH` so RCCL can detect it.

### Accessing the Slingshot interconnect

To make available the libraries needed for the inter-node communication on LUMI, a number of libraries must be mounted in the container. That has been taken care with the module `singularity-bindings`. It can be installed with EasyBuild:
```bash
module load LUMI/22.08 partition/G
module load EasyBuild-user
eb singularity-bindings-system-cpeGNU-22.08.eb -r
```

### OpenMPI

The images from [rocm/tensorflow](https://hub.docker.com/r/rocm/tensorflow) are bassed on [OpenMPI](https://www.open-mpi.org) and launching a TensorFlow+Horovod script with `srun` on LUMI may not work. They must be submitted instead with OpenMPI's `mpirun`. OpenMPI can be installed with EasyBuild:
```bash
module load LUMI/22.08 partition/G
module load EasyBuild-user
eb OpenMPI-4.1.3-cpeGNU-22.08.eb -r
```
Once installed, loading the module `OpenMPI/4.1.3-cpeGNU-22.08`, will make the `mpirun` command available.

## Example

Let's run the [synthetic benchmark example](https://github.com/horovod/horovod/blob/v0.26.1/examples/tensorflow2/tensorflow2_synthetic_benchmark.py) from Horovod's repository. We can use the script [run.sh](run.sh).

We have used a few environment variables in the [run.sh](run.sh) script. The ones starting with `NCCL_` and `CXI_`, as well as `FI_CXI_DISABLE_CQ_HUGETLB` are used by RCCL for the communication over Slingshopt. The `MIOPEN_` ones are needed to make [MIOpen](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/index.html) create its caches on `/tmp`. Finally, with `SINGULARITYENV_LD_LIBRARY_PATH` some directories are included in the container's `LD_LIBRARY_PATH`. This is important for RCCL to find the `aws-ofi-rccl` plugin. In addition, `NCCL_DEBUG=INFO`, can be used to increase RCCL's logging level to make sure that the `aws-ofi-rccl` plugin is being used: The lines
```bash
NCCL INFO NET/OFI Using aws-ofi-rccl 1.4.0
```
and
```bash
NCCL INFO NET/OFI Selected Provider is cxi
```
should appear in the output.

After running the script above, the output should include something like this
```bash
Iter #0: 538.6 img/sec per GPU
Iter #1: 544.9 img/sec per GPU
Iter #2: 544.2 img/sec per GPU
Iter #3: 544.8 img/sec per GPU
Iter #4: 543.0 img/sec per GPU
Iter #5: 545.9 img/sec per GPU
Iter #6: 544.5 img/sec per GPU
Iter #7: 545.4 img/sec per GPU
Iter #8: 544.6 img/sec per GPU
Iter #9: 545.1 img/sec per GPU
Img/sec per GPU: 544.1 +-3.9
Total img/sec on 32 GPU(s): 17411.4 +-123.6
```

### Notes on the example
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
