export HOROVOD_WITHOUT_MXNET=1
export HOROVOD_WITHOUT_PYTORCH=1
export HOROVOD_GPU=ROCM
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_WITHOUT_GLOO=1
export HOROVOD_WITH_TENSORFLOW=1
export HOROVOD_ROCM_PATH=/opt/rocm
export HOROVOD_RCCL_HOME=/opt/rocm/rccl
export RCCL_INCLUDE_DIRS=/opt/rocm/rccl/include
export HOROVOD_RCCL_LIB=/opt/rocm/rccl/lib
export HCC_AMDGPU_TARGET=gfx90a

pip install --no-cache-dir --force-reinstall horovod==0.26.1

pip install psutil
