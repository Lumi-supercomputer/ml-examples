# Notes


## Input pipeline and MIOpen
With the example [`imagenet_cnn_synthetic_benchmark.py`](tensorflow/imagenet_cnn_synthetic_benchmark.py) We have seen that if `.repeat()` is not included on the pipeline, later on the training (`module.fit()`), tensorflow will stop to compile code before every epoch. This seems to be due to the way in which keras interacts with `tf.data`. The compilation time is quite long (several minutes).

In the current version of the script:
```python
AUTO = tf.data.experimental.AUTOTUNE
dataset = (tf.data.TFRecordDataset(list_of_files, num_parallel_reads=AUTO)
.map(decode, num_parallel_calls=AUTO)
.repeat()
.batch(batch_size)
.prefetch(AUTO)
)
```
there is not such a thing as tensorflow knows since the beginning the code that it's going to be run. The code will be compiled at the beginning of the run and cached for later runs.

Some github issues about the long compilation times with MIOpen:
 * https://github.com/ROCmSoftwarePlatform/MIOpen/issues/337
 * https://github.com/ROCmSoftwarePlatform/MIOpen/issues/130#issuecomment-612304331

