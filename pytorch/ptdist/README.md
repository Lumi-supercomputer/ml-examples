# PyTorch on LUMI

[PyTorch](https://pytorch.org) is an open source Python package that provides tensor computation, like NumPy, with GPU acceleration and deep neural networks built on a tape-based autograd system.

PyTorch can be installed by the users following the code's [instructions](https://pytorch.org/get-started/locally/). The options to choose for LUMI in the interactive instructions are `Linux`, `Pip` and `ROCm5.X`. For installing with `pip`, the `cray-python` module should be loaded. PyTorch comes with ROCm binaries needed for the GPU support. Even if a particular version of ROCm is not available on LUMI, PyTorch may still be able to use the GPUs.

PyTorch can be run within containers as well. In particular, containers from the images provided by [AMD on DockerHub](https://hub.docker.com/u/rocm).
Those images are updated frequently and make it possible to try PyTorch with recent ROCm versions. 
Another point in favor of using containers, is that PyTorch's installation directory can be quite large both in terms of storage size and number of files.

## Running PyTorch within containers

We recommend using container images from [`rocm/pytorch`](https://hub.docker.com/r/rocm/pytorch) or [`rocm/deepspeed`](https://hub.docker.com/r/rocm/deepspeed).

The images can be fetched with singularity:
```bash
SINGULARITY_TMPDIR=$SCRATCH/tmp-singularity singularity pull docker://rocm/pytorch:rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1
```
This will create an image file named `pytorch_rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1.sif` on the directory where the command was run. After the image has been pulled, the directory `$SCRATCH/tmp-singularity singularity` can be removed.

### Installing other packages along the container's PyTorch installation

Often we may need to install other packages to be used along PyTorch.
That can be done by creating a virtual environment within the container in a host directory.
This can be done by running the container interactively and creating a virtual environment in your `$HOME`.
As an example, let's do that to install the package `python-hostlist`:
```bash
$> singularity exec -B $SCRATCH:$SCRATCH pytorch_rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1.sif bash
Singularity> python -m venv pt_rocm5.4.1_env --system-site-packages
Singularity> . pt_rocm5.4.1_env/bin/activate
(pt_rocm5.4.1_env) Singularity> pip install python-hostlist
```
Now when running the container, the virtual environment must be activated before calling python.

## Multi-GPU training

The communication between LUMI's GPUs during training with Pytorch is done via [RCCL](https://github.com/ROCmSoftwarePlatform/rccl), which is a library of  collective communication routines for GPUs. RCCL works out of the box on LUMI's, however, a special plugin is required so it can take advantage of the [Slingshot interconnect](https://www.hpe.com/emea_europe/en/compute/hpc/slingshot-interconnect.html). That's the [`aws-ofi-rccl`](https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl) plugin, which is a library that can be used as a back-end for RCCL to interact with the interconnect via libfabric.

The `aws-ofi-rccl` plugin can be installed by the user with EasyBuild:
```bash
module load LUMI/22.08 partition/G
module load EasyBuild-user
eb aws-ofi-rccl-66b3b31-cpeGNU-22.08.eb -r
```
Once installed, loading the module `aws-ofi-rccl` will add the path to the library to the `LD_LIBRARY_PATH` so RCCL can detect it.

## Example

Let's now consider an example to test the steps above. We will use the script [cnn_distr.py](https://github.com/Lumi-supercomputer/lumi-reframe-tests/blob/main/checks/apps/deeplearning/pytorch/src/cnn_distr.py) which uses the [pt_distr_env.py](https://github.com/Lumi-supercomputer/lumi-reframe-tests/blob/main/checks/apps/deeplearning/pytorch/src/pt_distr_env.py) module to setup PyTorch's distributed environment. That module is based on `python-hostlist`, which we installed earlier.

The Slurm submission script can be something like this:
```bash
#!/bin/bash
#SBATCH --job-name=pt-cnn
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --time=0:10:0
#SBATCH --exclusive
#SBATCH --partition standard-g
#SBATCH --account=<project>
#SBATCH --gpus-per-node=8

module load LUMI/22.08 partition/G
module load singularity-bindings
module load aws-ofi-rccl

. ~/pt_rocm5.4.1_env/bin/activate

export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
export SINGULARITYENV_LD_LIBRARY_PATH=/opt/ompi/lib:${EBROOTAWSMINOFIMINRCCL}/lib:/opt/cray/xpmem/2.4.4-2.3_9.1__gff0e1d9.shasta/lib64:${SINGULARITYENV_LD_LIBRARY_PATH}

srun singularity exec -B"/appl:/appl" \
                      -B"$SCRATCH:$SCRATCH" \
                      $SCRATCH/pytorch_rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1.sif python cnn_distr.py
```
Here we have used a few environment variables. The ones starting with `NCCL_` and `CXI_`, as well as `FI_CXI_DISABLE_CQ_HUGETLB` are used by RCCL for the communication over Slingshopt. The `MIOPEN_` ones are needed to make [MIOpen](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/index.html) create its caches on `/tmp`. Finally, with `SINGULARITYENV_LD_LIBRARY_PATH` some directories are included in the container's `LD_LIBRARY_PATH`. This is important for RCCL to find the `aws-ofi-rccl` plugin. In addition, `NCCL_DEBUG=INFO`, can be used to increase RCCL's logging level to make sure that the `aws-ofi-rccl` plugin is being used: The lines
```bash
NCCL INFO NET/OFI Using aws-ofi-rccl 1.4.0
```
and
```bash
NCCL INFO NET/OFI Selected Provider is cxi
```
should appear in the output.

To make available the libraries needed for the inter-node communication on LUMI, a number of libraries must be mounted in the container. That has been taken care with the module `singularity-bindings`. It can be installed with EasyBuild:
```bash
module load LUMI/22.08 partition/G
module load EasyBuild-user
eb singularity-bindings-system-cpeGNU-22.08.eb -r
```

After running the script above, the output should include something like this
```bash
 * Rank 0 - Epoch  1: 212.39 images/sec per GPU
 * Rank 0 - Epoch  2: 212.34 images/sec per GPU
 * Rank 0 - Epoch  3: 212.55 images/sec per GPU
 * Rank 0 - Epoch  4: 212.44 images/sec per GPU
 * Total average: 6792.58 images/sec
```

# Notes

`torch.distributed` needs some setup before starting. All that's needed has been put together in [pt_distr_env.py](pt_distr_env.py).
That needs the [`python-hostlist`](https://pypi.org/project/python-hostlist) package.

If you prefer to do not install `python-hostlist`, `MASTER_ADDR` can be obtained from
```
scontrol show hostname $SLURM_NODELIST | head -n1
```
When using a container, `scontrol` is not available. `MASTER_ADDR` needs to be defined on the batch script with
```bash
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
```
and adapt [pt_distr_env.py](pt_distr_env.py) accordingly.
