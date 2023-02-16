## [lumi-eap] BERT finetuning on the SquadQA dataset

Here we use the same image of the [PyTorch distributed example](https://github.com/Lumi-supercomputer/ml-examples/tree/main/pytorch/ptdist).
Please, have a look to the example before continuing with this, as it has important information on how to run PyTorch on LUMI.

### Installing packages 
```bash
# For BERT
singularity exec $SCRATCH/pytorch_rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1.sif bash
python -m venv ds-env
. ds-env/bin/activate
MPICC=mpicc pip install mpi4py  # probably `MPICC=mpicc` is not needed here
pip install datasets transformers
pip install rich   # only for formatting the testing at the end of the run
pip install deepspeed
```

We need to do a small change on one DeepSpeed module, so it properly get's the namesof the compute nodes on LUMI:
```bash
sed 's/hostname -I/hostname -i/g' ds-env/lib/python3.7/site-packages/deepspeed/comm/comm.py
```


### Running a BERT SquadQA fine-tuning with DeepSpeed
Deepspeed scripts are run with a rank per GPU and they need to be launched with OpenMPI's `mpirun` (an OpenMPI installation is required in the system).

We will run the script [`3_squad_bert_deepspeed.py`](https://github.com/eth-cscs/pytorch-training/blob/oct2022/bert_squad/3_squad_bert_deepspeed.py).
Other files there are needed so it's easier to clone the repo to have all the
content in the directory [`pytorch-training/bert_squad`](https://github.com/eth-cscs/pytorch-training/blob/oct2022/bert_squad).
Please, make sure that you download the example from the default branch (`oct2022 at the time this is written).
