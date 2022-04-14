import numpy as np
import tensorflow as tf


# distribution strategy
communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    cluster_resolver=tf.distribute.cluster_resolver.SlurmClusterResolver(),
    communication_options=communication_options
)

# create a linear function with noise as our data
nsamples = 1000
ref_slope = 2.0
ref_offset = 0.0
noise = np.random.random((nsamples, 1)) - 0.5    # -0.5 to center the noise
x_train = np.random.random((nsamples, 1)) - 0.5  # -0.5 to center x around 0
y_train = ref_slope * x_train + ref_offset + noise

# dataset
dataset = tf.data.Dataset.from_tensor_slices((x_train.astype(np.float32),
                                              y_train.astype(np.float32)))
dataset = dataset.shuffle(1000)
dataset = dataset.batch(100)
dataset = dataset.repeat(150)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,), activation='linear'),
])

with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,), activation='linear'),
    ])
    opt = tf.keras.optimizers.SGD(learning_rate=0.5)
    model.compile(loss='mse', optimizer=opt)


fit = model.fit(dataset, verbose=2)

print(f'\n{model.trainable_variables}\n')
