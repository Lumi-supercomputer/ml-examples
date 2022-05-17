## [lumi-eap] SquadQA BERT finetuning

### The image
Here we used the image [rocm/deepspeed:rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0](https://hub.docker.com/layers/deepspeed/rocm/deepspeed/rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0/images/sha256-69798ed9488ae84a47ce256196953c74de1fdf75e75854d72b9afa27143a3129?context=explore).

### Installing packages for BERT
Deepspeed is already installed there. To run BERT it needs only a few packages more
```bash
# For BERT
MPICC=mpicc pip install --user mpi4py  # probably `MPICC=mpicc` is not needed here
pip install --user datasets
pip install --user transformers
pip install rich   # only for the script we use

# For the torch.distributed custom init script (not used here)
pip install --user python-hostlist
```

### Updating DeepSpeed
DeepSpeed can be updated (as long as it's compatible with the PyTorch version installed on the image) by simple installing it with pip within the container
```bash
pip install --user --upgrade deepspeed
```

### Running a BERT SquadQA fine-tuning with DeepSpeed
Deepspeed scripts are run with a rank per GPU and they need to be launched with OpenMPI's `mpirun` (an OpenMPI installation is required in the system).

We run the script [`bert_squad_deepspeed.py`](https://github.com/eth-cscs/pytorch-training/blob/master/bert_squad/bert_squad_deepspeed.py).
Other files there are needed (`dataset_utils.py`, `ds_config.json` and `eval_utils.py`), so it's easier to clone the repo to have all the
content in the directory [`pytorch-training/bert_squad`](https://github.com/eth-cscs/pytorch-training/tree/master/bert_squad).

The data can be downloaded from the internet and then it needs to be put inside a directory called `cache/data` at the same level
where the script will run:
```bash
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
mv train-v1.1.json dev-v1.1.json cache/data
```

When the script is run for the first time, it will fetch stuff from the internet, so it needs to be run on a login node.
The script has a command line option for only downloading and exit:
```bash
singularity exec -B $SCRATCH:/scratch $SCRATCH/deepspeed_rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0.sif \
            batch -c'
            cd /scratch/bert_squad;
            python bert_squad_deepspeed.py --download-only'
```

When everything has been fetched, it can be run 'for real' on the compute nodes.
For that, it's necessary to set `datasets` and `huggingface` to work offline.
That's done by setting `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`.
With 8 GPUs, which needs 8 ranks, it can be run like this
(the batch size should be changed first to 256 [here](https://github.com/eth-cscs/pytorch-training/blob/2e623d1b3b56f37f94c4a28d8671b491ebf39f77/bert_squad/ds_config.json#L2)
to take advantage of the 8 GPUs).

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
                   python bert_squad_deepspeed.py --deepspeed_config ds_config.json'
```
No need to mount anything. NCCL finds everything it needs:
```bash
nid000013:32994:32994 [0] NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE ; OOB nmn0:10.252.1.67<0>
```
With 8 GPUs the training and evaluation together take around 9 minutes. The training runs at about 548 samples/sec.
At the end it shows QA examples with the answers highlighted withing the paragrahps.

### Mixed Precision
With the config
```json
{
    "train_batch_size": 512,
    "gradient_accumulation_steps": 1,
    "steps_per_print": 10,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-05
        }
    },
    "fp16": {
        "enabled": true
    }
}
```
which enables the mixed precision training (which in turns allows for a larger batch size), the throughput can be increased to around 1200 samples/sec.
This reduced the run time to around 5 minutes. Keeping the same batch size of 256 as before, the throughput is around 1000 images/sec.

### ZeRO
The ZeRO optimization stage 2, can be enabled with
```json
{
    "train_batch_size": 512,
    "gradient_accumulation_steps": 1,
    "steps_per_print": 10,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-05
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}
```
This is not really needed but we can check that as expected it keeps the same non-ZeRO throughput (~1240 images/sec).
