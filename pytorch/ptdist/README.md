# Notes

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
