## [lumi-eap] Text summarization with T5 on XSum

Here we used the image [rocm/deepspeed:rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0](https://hub.docker.com/layers/deepspeed/rocm/deepspeed/rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0/images/sha256-69798ed9488ae84a47ce256196953c74de1fdf75e75854d72b9afa27143a3129?context=explore).

Deepspeed is already installed there. For the t5 models, few packages more are needed
```
# For BERT
MPICC=mpicc pip install --user mpi4py  # probably `MPICC=mpicc` is not needed here
pip install --user datasets
pip install --user transformers
```

Deepspeed scripts are run with a rank per GPU.

We run the script [`t5-text-summarization-deepspeed.py`](https://github.com/eth-cscs/pytorch-training/blob/master/t5_xsum/t5-text-summarization-deepspeed.py).

### The data
The data is downloaded automatically from the internet. It is stored by default on `$HOME/.cache/huggingface/datasets/xsum`.

### Running
When the script is run for the first time, it will fetch stuff from the internet, so it can't be run on a compute node.
For that, the script should be run on a login node with the option `--download-only`:
```bash
singularity exec -B $SCRATCH:/scratch $SCRATCH/deepspeed_rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0.sif \
            batch -c'
            cd /scratch/bert_squad;
            python bert_squad_deepspeed.py --model t5-small --download-only'
```

When everything has been fetched, it can be run on the compute nodes.
For that, it's necessary to set `datasets` and `huggingface` to work offline.
That's done by setting `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`.
With 8 GPUs, which needs 8 ranks, It can be run like this
(the batch size should be changed first to 2048 [here](https://github.com/eth-cscs/pytorch-training/blob/master/t5_xsum/ds_config.json):
```bash
salloc -peap -N2 -Aproject_462000002 --gres=gpu:mi100:4 --time 1:00:00 --ntasks-per-node=4

module swap PrgEnv-cray PrgEnv-gnu
export PATH=$HOME/software/openmpi-4.1.2-install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/software/openmpi-4.1.2-install/lib:$LD_LIBRARY_PATH

mpirun singularity exec -B $SCRATCH:/scratch $SCRATCH/deepspeed_rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0.sif \
                   bash -c '
                   export TRANSFORMERS_OFFLINE=1;
                   export HF_DATASETS_OFFLINE=1;
                   cd /scratch/bert_squad;
                   python bert_squad_deepspeed.py --deepspeed_config ds_config.json --model t5-small'
```
No need to mount anything. NCCL finds everything it needs:
```bash
nid000013:32994:32994 [0] NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE ; OOB nmn0:10.252.1.67<0>
```
The training runs at about 3480 samples/sec.

With the config file linked above, deepspeed will use the highest level of ZeRo optimization including cpu offloading.
That's not necessary for the `t5_small` model, but it's necessary for for the large `t5-3b` model which doesn't fit on a MI100 GPU. 
The following table shows the sizes of the variants of the t5 model implemented in huggingface:
| Variant                                     |   Parameters    |
|:-------------------------------------------:|----------------:|
| [t5-small](https://huggingface.co/t5-small) |    60,506,624   | 
| [t5-large](https://huggingface.co/t5-large) |   737,668,096   | 
| [t5-3b](https://huggingface.co/t5-3b)       | 2,851,598,336   | 

 With the ZeRo-3 optimization plus the cpu offloading, it's possible to train the `t5-3b` model on a single GPU (batch size 16).
 I tried it with 2 GPUs too, using 1 GPU per node. With more than 1 GPU per node, that training with the `t5-3b` model crashes at some
 point with an MPI error.
