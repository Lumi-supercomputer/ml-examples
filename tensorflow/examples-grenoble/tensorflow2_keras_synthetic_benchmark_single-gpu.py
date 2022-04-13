# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import numpy as np
from timeit import default_timer as timer

import tensorflow as tf
# import horovod.tensorflow.keras as vd
from tensorflow.keras import applications

#from tensorflow.keras import mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)

# Benchmark settings
parser = argparse.ArgumentParser(description='TensorFlow Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='ResNet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
device = 'GPU' 

# Synthetic dataset
data = tf.random.uniform([args.batch_size, 224, 224, 3])
target = tf.random.uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((data, target)).cache().repeat().batch(args.batch_size)

# Set up standard model.
model = getattr(applications, args.model)(weights=None)
opt = tf.optimizers.SGD(0.01)

model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
              optimizer=opt)
              # experimental_run_tf_function=False)

 
class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.img_secs = []

    def on_train_end(self, logs=None):
        img_sec_mean = np.mean(self.img_secs)
        img_sec_conf = 1.96 * np.std(self.img_secs)
        print('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))

    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs=None):
        time = timer() - self.starttime
        img_sec = args.batch_size * args.num_batches_per_iter / time
        print('Iter #%d: %.1f img/sec per %s' % (epoch, img_sec, device))
        # skip warm up epoch
        if epoch > 0:
            self.img_secs.append(img_sec)

callbacks = []
timing = TimingCallback()
callbacks.append(timing)

# Train the model.
model.fit(
    dataset,
    # batch_size=args.batch_size,
    steps_per_epoch=args.num_batches_per_iter,
    callbacks=callbacks,
    epochs=args.num_iters,
    verbose=0,
)
