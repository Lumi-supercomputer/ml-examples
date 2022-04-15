# Notes

## [lumi-eap] Running a torch.distributed example within a container

Using `amdih/pytorch:rocm4.2_ubuntu18.04_py3.6_pytorch_1.9.0` from the [AMD Infinity Hub](https://www.amd.com/en/technologies/infinity-hub/pytorch)
with singularity. It can be pulled with

```bash
singularity pull docker://amdih/pytorch:rocm4.2_ubuntu18.04_py3.6_pytorch_1.9.0
```

It doesn't need to be run with `mpirun`.

With `torch.distributed`, one needs 1 rank per node.
The multiple GPUs within a node, are handled by that one rank.

In the [batch script](run-singularity.sh) there is some setup for singularity although it doesn't seem to
always be necessary. With the exports, NCCL finds
```
nid000012:86554:86554 [0] NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE ; OOB nmn0:10.252.1.69<0>
```
which is what's expected. With no exports, NCCL finds
```
nid000012:86066:86066 [0] NCCL INFO NET/Socket : Using [0]nmn0:10.252.1.69<0> [1]hsn0:10.253.6.188<0>
```
However, the throughput in both cases are very close.
With no exports, there is the message in the output
```
libibverbs: Warning: couldn't open config directory '/etc/libibverbs.d'.
```

It's necessary to add `/usr/lib64/libibverbs.so.1:/usr/lib/x86_64-linux-gnu/libibverbs.so.1` in the `SINGULARITY_BIND`. That wasn't
necessary with the tensorflow+hvd container. Also, there's no need to mount any openmpi libraries.

`torch.distributed` needs some setup before starting. All that's needed has been put together in [pt_distr_env.py](pt_distr_env.py).
That needs the [`python-hostlist`](https://pypi.org/project/python-hostlist) package.

If `python-hostlist` is not available, `MASTER_ADDR` can be obtained from
```
scontrol show hostname $SLURM_NODELIST | head -n1
```
When using a container, `scontrol` is not available. `MASTER_ADDR` needs to be defined on the batch script with
```bash
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
```
and adapt [pt_distr_env.py](pt_distr_env.py) accordingly.
