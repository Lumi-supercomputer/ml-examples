## [lumi-eap] BERT finetuning on the SquadQA dataset

Here we use the same image we saw in the [PyTorch distributed example](https://github.com/Lumi-supercomputer/ml-examples/tree/main/pytorch/ptdist).
Please, have a look to the example before continuing reading this, as it has important information on how to run PyTorch on LUMI that's relevant for runnnig DeepSpeed as well.

### Installing the packages 

That image doesn't include DeepSpeed. We will install it on a virtual environment along with other packages that we need for this example:
```bash
# For BERT
singularity exec $SCRATCH/pytorch_rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1.sif bash
python -m venv ds-env --system-site-packages
. ds-env/bin/activate
MPICC=mpicc pip install mpi4py  # probably `MPICC=mpicc` is not needed here
pip install datasets transformers
pip install rich   # only for formatting the testing at the end of the run
pip install deepspeed
```

We need to do a small change on one DeepSpeed module, so it properly gets the names of the LUMI's compute nodes:
```bash
sed -i 's/hostname -I/hostname -i/g' ds-env/lib/python3.7/site-packages/deepspeed/comm/comm.py
```


### Running

Deepspeed programs must be run with a rank per GPU and in LUMI, they need to be launched with OpenMPI's `mpirun`.
OpenMPI can be installed with EasyBuild:
```bash
module load LUMI/22.08 partition/G
module load EasyBuild-user
eb OpenMPI-4.1.3-cpeGNU-22.08.eb -r
```

We will run the script [`3_squad_bert_deepspeed.py`](https://github.com/eth-cscs/pytorch-training/blob/oct2022/bert_squad/3_squad_bert_deepspeed.py).
Other files there are needed so it's easier to clone the repo to have all the
content in the directory [`pytorch-training/bert_squad`](https://github.com/eth-cscs/pytorch-training/blob/oct2022/bert_squad).
Please, make sure that you download the example from the default branch (`oct2022` at the time this is written).

Once the code has been downloaded, we can use [`run.sh`](run.sh) to submit the job.
There we use a few environment variables that are explained in the [PyTorch distributed example](https://github.com/Lumi-supercomputer/ml-examples/tree/main/pytorch/ptdist). The only difference here is that we need to use OpenMPI's `mpirun` to submit the job instead of `srun`.
