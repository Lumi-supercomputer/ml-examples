# Notes

## [lumi-eap] Running a torch.distributed example within a container

Using `amdih/pytorch:rocm4.2_ubuntu18.04_py3.6_pytorch_1.9.0` from the [AMD Infinity Hub](https://www.amd.com/en/technologies/infinity-hub/pytorch)
with singularity. It can be pulled with

```bash
singularity pull docker://amdih/pytorch:rocm4.2_ubuntu18.04_py3.6_pytorch_1.9.0
```

For internode communication, NCCL seems to need the following
```
export NCCL_IB_HCA=hsn0
```
Another posibility is NCCL_IB_HCA=mlx5_0, which what NCCL finds by default if nothing is set.
That works for multiple GPUs on a single node, but doesn't seem to work for multiple node. It may hang or crash with an 'unhandled error'.


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
export NCCL_IB_HCA=hsn0
srun singularity exec pytorch_rocm4.2_ubuntu18.04_py3.6_pytorch_1.9.0.sif python cnn_distr.py
```
The test above is a CNN from [cnn_distr.py](https://github.com/eth-cscs/pytorch-training/blob/master/cnn_synthetic_benchmark/cnn_distr.py).

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
