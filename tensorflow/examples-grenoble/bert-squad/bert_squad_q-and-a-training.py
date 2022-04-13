import os
import json
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import dataset_utils as du
from tokenizers import BertWordPieceTokenizer
from transformers import TFBertModel, BertTokenizer


batch_size = 16
max_len = 384


slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                               cache_dir="/home/sarafael/tf-examples/bert-squad/_bert_tockenizer")

save_path = f"/home/sarafael/tf-examples/bert-squad/bert_tockenizer"
if not os.path.exists(save_path):
    os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer(f"{save_path}/vocab.txt", lowercase=True)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer(f"{save_path}/vocab.txt", lowercase=True)

train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
train_path = keras.utils.get_file("train.json", train_data_url, cache_dir="./")
eval_path = keras.utils.get_file("eval.json", eval_data_url, cache_dir="./")

with open(train_path) as f:
    raw_train_data = json.load(f)

with open(eval_path) as f:
    raw_eval_data = json.load(f)

print(f"{len(raw_train_data['data'])} training items loaded.")
print(f"{len(raw_eval_data['data'])} evaluation items loaded.")

train_squad_examples = du.create_squad_examples(raw_train_data, max_len, tokenizer)
x_train, y_train = du.create_inputs_targets(train_squad_examples, shuffle=True, seed=42)
print(f"{len(train_squad_examples)} training points created.")

eval_squad_examples = du.create_squad_examples(raw_eval_data, max_len, tokenizer)
x_eval, y_eval = du.create_inputs_targets(eval_squad_examples)
print(f"{len(eval_squad_examples)} evaluation points created.")


encoder = TFBertModel.from_pretrained("bert-base-uncased",
                                      cache_dir=f"/home/sarafael/tf-examples/bert-squad/bert_model")

input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)

embedding = encoder(input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask)[0]

start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
start_logits = layers.Flatten()(start_logits)
start_probs = layers.Activation(keras.activations.softmax)(start_logits)

end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
end_logits = layers.Flatten()(end_logits)
end_probs = layers.Activation(keras.activations.softmax)(end_logits)

model = keras.Model(inputs=[input_ids, token_type_ids, attention_mask],
                    outputs=[start_probs, end_probs])

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(lr=5e-5)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.summary()

def get_dataset(x, y, batch_size=batch_size):
    dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(x),
        tf.data.Dataset.from_tensor_slices(y),
    ))
    dataset = dataset.shuffle(2048, seed=42)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset

# batch shapes
for X, Y in get_dataset(x_train, y_train).take(1):
    print([i.shape for i in X],
          [i.shape for i in Y])


fit = model.fit(get_dataset(x_train, y_train),
                epochs=1,
                steps_per_epoch=50,
                validation_data=get_dataset(x_eval, y_eval),
                validation_steps=len(y_eval[0]) // batch_size)


