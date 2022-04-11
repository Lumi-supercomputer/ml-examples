# Notes

## [lumi-eap] Running a torch.distributed example within a container

Using `amdih/pytorch:rocm4.2_ubuntu18.04_py3.6_pytorch_1.9.0` from the [AMD Infinity Hub](https://www.amd.com/en/technologies/infinity-hub/pytorch)
with singularity. It can be pulled with

```bash
singularity pull docker://amdih/pytorch:rocm4.2_ubuntu18.04_py3.6_pytorch_1.9.0
```

See the [singularity setup for tensorflow+hvd](hvd/README.md).

It doesn't need to be run with `mpirun`.
With `torch.distributed`, one needs 1 rank per node.
The multiple GPUs within a node, are handled by that one rank.
```
#!/bin/bash -l

#SBATCH --job-name=test-pt
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=eap
#SBATCH --account=project_462000002
#SBATCH --gres=gpu:mi100:4

# export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)

export NCCL_DEBUG=INFO
export SINGULARITY_BIND='/opt/cray/libfabric/1.11.0.4.106/lib64/libfabric.so.1:/ext_cray/libfabric.so.1,/opt/cray/pe/lib64/libpmi2.so.0:/ext_cray/libpmi2.so.0,/opt/cray/pe/mpich/8.1.8/ofi/gnu/9.1/lib/libmpi_gnu_91.so.12:/ext_cray/libmpi_gnu_91.so.12,/usr/lib64/liblustreapi.so:/ext_cray/liblustreapi.so,/usr/lib64/libatomic.so.1:/usr/lib64/libatomic.so.1,/usr/lib64/libpals.so.0:/usr/lib64/libpals.so.0,/etc/libibverbs.d:/etc/libibverbs.d,/usr/lib64/libibverbs.so.1:/usr/lib/libibverbs.so.1,/var/opt/cray:/var/opt/cray,/appl:/appl,/opt/cray:/opt/cray,/usr/lib64/librdmacm.so.1:/ext_cray/librdmacm.so.1,/lib64/libtinfo.so.6:/ext_cray/libtinfo.so.6,/usr/lib64/libibverbs.so.1:/usr/lib/x86_64-linux-gnu/libibverbs.so.1'
export SINGULARITYENV_LD_LIBRARY_PATH='/etc/libibverbs.d:/var/opt/cray/pe/pe_images/aocc-compiler/usr/lib64/libibverbs:/usr/lib64:/opt/cray/pe/lib64:/opt/gcc/10.2.0/snos/lib64:/ext_cray:/usr/lib64:/opt/cray/pe/lib64:/opt/cray/xpmem/2.2.40-2.1_3.9__g3cf3325.shasta/lib64:$LD_LIBRARY_PATH'

srun singularity exec pytorch_rocm4.2_ubuntu18.04_py3.6_pytorch_1.9.0.sif python cnn_distr.py
```
The test above is a CNN from [cnn_distr.py](https://github.com/eth-cscs/pytorch-training/blob/master/cnn_synthetic_benchmark/cnn_distr.py).

He it's necessary to add `/usr/lib64/libibverbs.so.1:/usr/lib/x86_64-linux-gnu/libibverbs.so.1` in the `SINGULARITY_BIND`. That wasn't
necessary with the tensorflow+hvd container. Also, there's no need to mount any openmpi libraries.

For the distribution setup, something like this can be used:
```python
import os
import subprocess
import hostlist


def setup_distr_env():
    os.environ['MASTER_PORT'] = '39591'
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NNODES']
    os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    # node_list = os.environ['SLURM_NODELIST']
    # master_node = subprocess.getoutput(
    #     f'scontrol show hostname {node_list} | head -n1'
    # )
    # os.environ['MASTER_ADDR'] = master_node
    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
    os.environ['MASTER_ADDR'] = hostnames[0]
```
That needs the [`python-hostlist`](https://pypi.org/project/python-hostlist) package.
If it's not available, once can get the `MASTER_ADDR` from
```
scontrol show hostname $SLURM_NODELIST | head -n1
```
When using a container, `scontrol` is not available. `MASTER_ADDR` needs to be defined on the batch script as in
the commented line above.
