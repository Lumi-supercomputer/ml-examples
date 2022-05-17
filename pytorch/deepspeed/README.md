## [lumi-eap] DeepSpeed

### The image
Here we used the image [rocm/deepspeed:rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0](https://hub.docker.com/layers/deepspeed/rocm/deepspeed/rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0/images/sha256-69798ed9488ae84a47ce256196953c74de1fdf75e75854d72b9afa27143a3129?context=explore).

### Updating DeepSpeed
DeepSpeed can be updated (as long as it's compatible with the PyTorch version installed on the image) by simple installing it with pip within the container
```bash
pip install --user --upgrade deepspeed
```

### Running a BERT SquadQA fine-tuning with DeepSpeed
Deepspeed scripts are run with a rank per GPU and they need to be launched with OpenMPI's `mpirun` (an OpenMPI installation is required in the system).
