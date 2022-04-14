import argparse
import os
import time
import numpy as np
import tensorflow as tf
tf.version.VERSION


# Benchmark settings
parser = argparse.ArgumentParser(description='TensorFlow Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='ResNet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size per GPU')
parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

rank = int(os.environ['SLURM_NODEID'])
mpi_world_size = int(os.environ['SLURM_NPROCS'])
local_size = len(gpus)
size = local_size * mpi_world_size

communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    cluster_resolver=tf.distribute.cluster_resolver.SlurmClusterResolver(),
    communication_options=communication_options
)

data = tf.random.uniform([args.batch_size * size, 224, 224, 3])
target = tf.random.uniform([args.batch_size * size, 1], minval=0, maxval=999, dtype=tf.int64)

dataset = tf.data.Dataset.from_tensor_slices((data, target))
# dataset = dataset.cache()
dataset = dataset.shard(size, rank)
dataset = dataset.repeat(args.num_iters)
dataset = dataset.batch(args.batch_size)

# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
# dataset = dataset.with_options(options)

with strategy.scope():
    model = getattr(tf.keras.applications, args.model)(weights=None)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    # from_logits=True,
    reduction=tf.keras.losses.Reduction.NONE)

def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss,
                                      global_batch_size=(args.batch_size * size))

def train_step(inputs):
    features, labels = inputs

    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def distributed_train_step(dist_inputs):
    per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
    strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


if rank == 0:
    print(f'Model: {args.model}')
    print(f'Batch size per GPU: {args.batch_size}')
    print(f'Number of GPUs: {size} [{local_size} GPUs/node]')

# training
for step, dist_inputs in enumerate(dataset):
    start = time.time()
    distributed_train_step(dist_inputs)
    dt = time.time() - start
    
    if rank == 0:
        img_sec = args.batch_size * size / dt
        print(f'Iter #{step}: {img_sec:.1f} img/sec')
