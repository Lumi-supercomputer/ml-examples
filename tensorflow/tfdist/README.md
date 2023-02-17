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
However `ensurepip` can be fetched from the [cpython repository](https://github.com/python/cpython/tree/main/Lib/ensurepip) using [this script](download-ensurepip-py3.9.sh) and then made available in the container by mounting it on `usr/lib/python3.9/ensurepip`:

```bash
$> singularity exec -B ensurepip:/usr/lib/python3.9/ensurepip tensorflow_rocm5.4.1-tf2.10-dev.sif bash
Singularity> python -m venv tf_rocm5.4.1_env --system-site-packages
Singularity> . tf_rocm5.4.1_env/bin/activate
(tf_rocm5.4.1_env) Singularity> pip install ...
```
Now when running the container, the virtual environment must be activated before calling python. Once the virtual environment has been created, the `ensurepip` module is no longer needed.

## Multi-GPU training with `tf.distribute`

As LUMI is composed by a set of multi-GPU nodes, you probably will use `tf.distribute.MultiWorkerMirroredStrategy` combined with the `SlurmClusterResolver` to run a distributed training job. In that case, the job should have one rank per node.

It may happen that the file `tensorflow/python/distribute/cluster_resolver/slurm_cluster_resolver.py` in the container's is outdated with respect to
[slurm_cluster_resolver.py](https://raw.githubusercontent.com/tensorflow/tensorflow/66e587c780c59f6bad2ddae5c45460440002dc68/tensorflow/python/distribute/cluster_resolver/slurm_cluster_resolver.py). In that case, you need to download the new version and replace it by the container's installation. That can be done by mounting the file [like this](https://github.com/Lumi-supercomputer/ml-examples/blob/82a5d1151fa888e16c603972ccc6e01e58d6fb9b/tensorflow/tfdist/run.sh#L28).

Together with that, `CUDA_VISIBLE_DEVICES` must be set. For instance:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

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

## Example

Let's run the [tf2_distr_synthetic_benchmark.py](tf2_distr_synthetic_benchmark.py) example. We can use the script [run.sh](run.sh).

We have used a few environment variables in the [run.sh](run.sh) script. The ones starting with `NCCL_` and `CXI_`, as well as `FI_CXI_DISABLE_CQ_HUGETLB` are used by RCCL for the communication over Slingshopt. The `MIOPEN_` ones are needed to make [MIOpen](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/index.html) create its caches on `/tmp`. Finally, with `SINGULARITYENV_LD_LIBRARY_PATH` some directories are included in the container's `LD_LIBRARY_PATH`. This is important for RCCL to find the `aws-ofi-rccl` plugin. In addition, `NCCL_DEBUG=INFO`, can be used to increase RCCL's logging level to make sure that the `aws-ofi-rccl` plugin is being used: The lines
```bash
NCCL INFO NET/OFI Using aws-ofi-rccl 1.4.0
```
and
```bash
NCCL INFO NET/OFI Selected Provider is cxi
```
should appear in the output.
