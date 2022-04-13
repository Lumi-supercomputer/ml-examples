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
import timeit
import time

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import applications

# Benchmark settings
parser = argparse.ArgumentParser(description='TensorFlow Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

args = parser.parse_args()

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = ds_train.repeat(2)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                                                    activation='relu',
                                                    input_shape=(28, 28, 1)),
                             tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                             tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                             tf.keras.layers.Dropout(0.25),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(10, activation='softmax')])
opt = tf.optimizers.Adam(0.001)

# @tf.function
def benchmark_step(data, target):
    with tf.GradientTape() as tape:
        probs = model(data, training=True)
        loss = tf.losses.sparse_categorical_crossentropy(target, probs)

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def log(s, nl=True):
    print(s, end='\n' if nl else '')


log('Batch size: %d' % args.batch_size)
device = 'GPU'
log('Number of %ss: %d' % (device, 1))


with tf.device(device):
    log('Running benchmark...')
    img_secs = []
    for data, target in ds_train:
        loss = benchmark_step(data, target)
        print(tf.math.reduce_sum(loss))
