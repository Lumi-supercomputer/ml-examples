# Notes

## [grenoble] Running a horovod example within the container 
We are using the image [rocm/tensorflow:rocm4.1-tf2.3-dev](https://hub.docker.com/layers/rocm/tensorflow/rocm4.1-tf2.3-dev/images/sha256-0f369142a95872bef829fc61256a628828e0427284ff8f2f8d1f821023aa5b4c?context=explore) and runnig with singularity.

>> Before running it's necessary to install `psutil` and `cloudpickle` inside the container:
```bash
pip install psutil cloudpickle
```

On the grenoble nodes, [the horovod example](https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py) can be run interactively - login in the container interactively and run the script as follows: 
```bash
mpirun -n 2 python tensorflow2_keras_synthetic_benchmark.py --model=ResNet50 --batch-size=256 --num-batches-per-iter=10
```
Doing this, each rank will get a GPU.

Running singularity with `srun` requires probably some setup on the grenoble nodes. It's not working now.
