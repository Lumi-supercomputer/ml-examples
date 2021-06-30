import os
import glob
import numpy as np
import tensorflow as tf
from datetime import datetime
from timeit import default_timer as timer


image_shape = (229, 229)
batch_size = 128

def decode(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
        })
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.image.resize(image, image_shape, method='bicubic')
    label = tf.cast(features['image/class/label'], tf.int64) - 1  # [0-999]
    return image, label


list_of_files = glob.glob('/home/sarafael/imagenet/train-000*')

AUTO = tf.data.experimental.AUTOTUNE
dataset = (tf.data.TFRecordDataset(list_of_files, num_parallel_reads=AUTO)
           .map(decode, num_parallel_calls=AUTO)
           .repeat()
           .batch(batch_size)
           .prefetch(AUTO)
          )

model = tf.keras.applications.InceptionV3(weights=None,
                                          input_shape=(*image_shape, 3),
                                          classes=1000)

optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join('inceptionv3_logs',
                                                                  datetime.now().strftime("%d-%H%M")),
                                             histogram_freq=1,
                                             profile_batch='80,100')

class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.img_secs = []

    def on_train_end(self, logs=None):
        img_sec_mean = np.mean(self.img_secs)
        img_sec_conf = 1.96 * np.std(self.img_secs)
        print('Img/sec per %s: %.1f +-%.1f' % ('device', img_sec_mean, img_sec_conf))
        print('Total img/sec on %d %s(s): %.1f +-%.1f' %
             (1, 'device', img_sec_mean, img_sec_conf))

    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs=None):
        time = timer() - self.starttime
        img_sec = 128 * 410 / time
        print('Iter #%d: %.1f img/sec per %s' % (epoch, img_sec, 'device'))
        # skip warm up epoch
        if epoch > 0:
            self.img_secs.append(img_sec)


fit = model.fit(dataset,
                epochs=10,
                steps_per_epoch=78,  # for train-000*
                # callbacks=[TimingCallback()]
                )
