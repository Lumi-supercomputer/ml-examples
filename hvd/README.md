# Notes

## [grenoble] Running a horovod example within the container 
First it's necessary to install `psutil` and `cloudpickle` inside the container:
```bash
pip install psutil cloudpickle
```

On the grenoble nodes, [the horovod example](https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py) can be run interactively - login in the container interactively and run the script as follows: 
```bash
mpirun -n 2 python tensorflow2_keras_synthetic_benchmark.py --model=ResNet50 --batch-size=256 --num-batches-per-iter=10
```
Running singularity with `srun` requires probably some setup on the grenoble nodes. It's not working now.
