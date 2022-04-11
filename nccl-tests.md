## [lumi-eap] Testing RCCL with [rccl-test](https://github.com/ROCmSoftwarePlatform/rccl-tests)

Here we use the same image that we use for [tensorflow+hvd](hvd/README.md).

### Build
Run the container interactively
```bash
singularity exec $SCRATCH/tensorflow_rocm4.2-tf2.5-dev.sif bash
``` 
and then within the container
```bash

cd ~/rccl-tests
make MPI=1 MPI_HOME=/openmpi HIP_HOME=/opt/rocm-4.2.0/hip RCCL_HOME=/opt/rocm-4.2.0
```

### Run
First it needs some exports:
```bash
module swap PrgEnv-cray/8.0.0 PrgEnv-gnu
export PATH=$HOME/software/openmpi-4.1.2-install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/software/openmpi-4.1.2-install/lib:$LD_LIBRARY_PATH
export SINGULARITY_BIND='/opt/cray/libfabric/1.11.0.4.106/lib64/libfabric.so.1:/ext_cray/libfabric.so.1,/opt/cray/pe/lib64/libpmi2.so.0:/ext_cray/libpmi2.so.0,/opt/cray/pe/mpich/8.1.8/ofi/gnu/9.1/lib/libmpi_gnu_91.so.12:/ext_cray/libmpi_gnu_91.so.12,/usr/lib64/liblustreapi.so:/ext_cray/liblustreapi.so,/usr/lib64/libatomic.so.1:/usr/lib64/libatomic.so.1,/usr/lib64/libpals.so.0:/usr/lib64/libpals.so.0,/etc/libibverbs.d:/etc/libibverbs.d,/usr/lib64/libibverbs.so.1:/usr/lib/libibverbs.so.1,/var/opt/cray:/var/opt/cray,/appl:/appl,/opt/cray:/opt/cray,/usr/lib64/librdmacm.so.1:/ext_cray/librdmacm.so.1,/lib64/libtinfo.so.6:/ext_cray/libtinfo.so.6,$HOME/software/openmpi-4.1.2-install:/ext_openmpi'
export SINGULARITYENV_LD_LIBRARY_PATH='/etc/libibverbs.d:/var/opt/cray/pe/pe_images/aocc-compiler/usr/lib64/libibverbs:/usr/lib64:/opt/cray/pe/lib64:/opt/gcc/10.2.0/snos/lib64:/ext_cray:/usr/lib64:/opt/cray/pe/lib64:/opt/cray/xpmem/2.2.40-2.1_3.9__g3cf3325.shasta/lib64:/ext_openmpi/lib:$LD_LIBRARY_PATH'
```

For two nodes with one GPU per node, it can be run like this:
```bash
$> salloc -peap -N2 -Aproject_462000002 --gres=gpu:mi100:1 --time 1:00:00 --ntasks-per-node=1
$> mpirun singularity exec $SCRATCH/tensorflow_rocm4.2-tf2.5-dev.sif ~/rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
# nThread: 1 nGpus: 1 minBytes: 8 maxBytes: 134217728 step: 2(factor) warmupIters: 5 iters: 20 validation: 1
#
# Using devices
#   Rank  0 Pid 106347 on  nid000013 device  0 [0000:c9:00.0] Device 738c
#   Rank  1 Pid  97185 on  nid000014 device  0 [0000:c9:00.0] Device 738c
#
#                                                       out-of-place                       in-place
#       size         count      type   redop     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                       (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum    41.99    0.00    0.00  0e+00    42.18    0.00    0.00  0e+00
          16             4     float     sum    42.61    0.00    0.00  0e+00    42.09    0.00    0.00  0e+00
          32             8     float     sum    40.75    0.00    0.00  0e+00    40.07    0.00    0.00  0e+00
          64            16     float     sum    40.39    0.00    0.00  0e+00    40.35    0.00    0.00  0e+00
         128            32     float     sum    40.65    0.00    0.00  0e+00    41.08    0.00    0.00  0e+00
         256            64     float     sum    40.81    0.01    0.01  0e+00    40.77    0.01    0.01  0e+00
         512           128     float     sum    42.12    0.01    0.01  0e+00    42.36    0.01    0.01  0e+00
        1024           256     float     sum    44.06    0.02    0.02  0e+00    43.74    0.02    0.02  0e+00
        2048           512     float     sum    45.78    0.04    0.04  0e+00    45.65    0.04    0.04  0e+00
        4096          1024     float     sum    45.98    0.09    0.09  0e+00    45.97    0.09    0.09  0e+00
        8192          2048     float     sum    49.93    0.16    0.16  0e+00    49.58    0.17    0.17  0e+00
       16384          4096     float     sum    51.94    0.32    0.32  0e+00    51.99    0.32    0.32  0e+00
       32768          8192     float     sum    61.29    0.53    0.53  0e+00    61.36    0.53    0.53  0e+00
       65536         16384     float     sum    82.53    0.79    0.79  0e+00    83.46    0.79    0.79  0e+00
      131072         32768     float     sum    136.4    0.96    0.96  0e+00    131.6    1.00    1.00  0e+00
      262144         65536     float     sum    213.9    1.23    1.23  0e+00    212.1    1.24    1.24  0e+00
      524288        131072     float     sum    129.7    4.04    4.04  0e+00    126.1    4.16    4.16  0e+00
     1048576        262144     float     sum    211.8    4.95    4.95  0e+00    214.8    4.88    4.88  0e+00
     2097152        524288     float     sum    383.2    5.47    5.47  0e+00    384.5    5.45    5.45  0e+00
     4194304       1048576     float     sum    735.6    5.70    5.70  0e+00    731.2    5.74    5.74  0e+00
     8388608       2097152     float     sum   1442.7    5.81    5.81  0e+00   1352.4    6.20    6.20  0e+00
    16777216       4194304     float     sum   2666.9    6.29    6.29  0e+00   2719.2    6.17    6.17  0e+00
    33554432       8388608     float     sum   5111.9    6.56    6.56  0e+00   5129.2    6.54    6.54  0e+00
    67108864      16777216     float     sum   9538.0    7.04    7.04  0e+00   9405.5    7.14    7.14  0e+00
   134217728      33554432     float     sum    17524    7.66    7.66  0e+00    17502    7.67    7.67  0e+00
# Errors with asterisks indicate errors that have exceeded the maximum threshold.
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 2.31741
#
```
