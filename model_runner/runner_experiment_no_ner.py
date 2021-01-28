import os

import tf_model_modified as tf_model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


from official.modeling import tf_utils
from official import nlp
from official.nlp import bert
from cubert_tokenizer import python_tokenizer, code_to_subtokenized_sentences

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks
from tensor2tensor.data_generators import text_encoder
import pandas as pd

import json
from sklearn import preprocessing
import itertools

tf.get_logger().setLevel('WARNING')

MAX_ARG_LENGTH =20
DS_PATH = "./data/_all_data.csv"
EPOCHS = 3
shuffle_buffer_size = 1000
SEQ_LENGTH = 512
MODEL_PATH = "bert2"

# bert_tokenizer = bert.tokenization.FullTokenizer(
#     vocab_file="cuvocab.txt",
#      do_lower_case=True)

with open(MODEL_PATH+"/cubert_config.json") as conf_file:
    config_dict = json.loads(conf_file.read())
    bert_config = bert.configs.BertConfig.from_dict(config_dict)

bert_encoder = bert.bert_models.get_transformer_encoder(
    bert_config, sequence_length=SEQ_LENGTH)

tf.keras.utils.plot_model(bert_encoder, show_shapes=True, dpi=48)
# tf.keras.utils.plot_model(bert_classifier, show_shapes=True, dpi=48)


checkpoint = tf.train.Checkpoint(model=bert_encoder)
checkpoint.restore(MODEL_PATH+'/bert1-1').assert_consumed()


if ".json" in DS_PATH:
    data = pd.read_json(DS_PATH)
else:
    data = pd.read_csv(DS_PATH)

tokenizer = python_tokenizer.PythonTokenizer()
subword_tokenizer = text_encoder.SubwordTextEncoder(MODEL_PATH+"/cuvocab.txt")

data_body1 = data['body.1']

## Preprocessign arg

data['arg_types'] = data['arg_types'].apply(eval)
df_labels = pd.DataFrame(data['arg_types'].values.tolist())

print(df_labels.head())

max_label_length=10
df_labels2=df_labels.iloc[:,0:max_label_length]
df_labels2[pd.isnull(df_labels2)]  = 'NaN'

enc = preprocessing.LabelEncoder()
all_types = df_labels.apply(pd.Series).stack().values
enc.fit(all_types)

df3 = df_labels2.apply(enc.transform)
data['labels'] = df3.values.tolist()

print(data.head())

def transform(code_text):
    return [2]+sum(code_to_subtokenized_sentences.code_to_cubert_sentences(
        code=code_text,
        initial_tokenizer=tokenizer,
        subword_tokenizer=subword_tokenizer),[])

MAX_BODY_LENGTH = 512

batch_size = 1

# def gen():
#     for idx, data_batch in data.iterrows():
#         sentence1 = tf.ragged.constant([transform(data_batch['body.1'])])
#         sentence1 = tf.reshape(sentence1.to_tensor(default_value=0, shape=[1,8,MAX_BODY_LENGTH]), [1,8*MAX_BODY_LENGTH])
#         labels = tf.convert_to_tensor(data_batch['labels'])
#         type_s1 = tf.zeros_like(sentence1)
#         yield ({'input_word_ids': sentence1,
#             'input_mask': tf.ones_like(sentence1),
#             'input_type_ids': type_s1,
#             'lens':len(data['arg_types'])},labels)


# dataset = tf.data.Dataset.from_generator(
#         gen,
#         ({"input_word_ids": tf.int32, "input_mask": tf.int32, "input_type_ids": tf.int32, "lens": tf.int32}, tf.int64),
#         (
#             {
#                 "input_word_ids": tf.TensorShape([1, 8*MAX_BODY_LENGTH]),
#                 "input_mask": tf.TensorShape([1, 8*MAX_BODY_LENGTH]),
#                 "input_type_ids": tf.TensorShape([1, 8*MAX_BODY_LENGTH]),
#                 "lens": None,
#             },
#             tf.TensorShape([max_label_length]),
#         ),
#     ) 


def gen():
    for _, data_batch in data.groupby(np.arange(len(data))//batch_size):
        if len(data_batch) < batch_size:
            continue
        ids = []
        sentence1 = []
        for _,data_batch_i in data_batch.iterrows():
            id_list = []
            sentence2 = np.array(transform(data_batch_i['body.1'])[:SEQ_LENGTH-1]+[3])
            for label in eval(data_batch_i['arg_names']):
                id_list.append( np.where(sentence2 == subword_tokenizer.encode(label)[0])[0][0])
            sentence1.append(sentence2)
        sentence1 = tf.ragged.constant(sentence1)
        sentence1 = sentence1.to_tensor(default_value=0, shape=[batch_size, SEQ_LENGTH])
        id_list =  tf.convert_to_tensor(id_list)
        # sentence1 = tf.reshape(sentence1, [batch_size,8*MAX_BODY_LENGTH])
        labels = tf.convert_to_tensor([np.array(data_batch_i['labels']) for _,data_batch_i in data_batch.iterrows()])
        type_s1 = tf.zeros_like(sentence1)
        yield (bert_encoder({'input_word_ids': sentence1,
            'input_mask': tf.ones_like(sentence1),
            'input_type_ids': type_s1})[0][0][id_list[0]],labels) #TODO change 0s for batches


dataset = tf.data.Dataset.from_generator(
        gen,
        (tf.float32, tf.int64),
        (   
            tf.TensorShape([1024,]),
            tf.TensorShape([batch_size, 1]),
        ),
    )

DATASET_SIZE = len(df3)
train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

full_dataset = dataset
# full_dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)

print(next(gen()))

# for i in full_dataset.take(1):
#     print(i)

# N_CLASSES = len(enc.classes_)
# N_CLASSES = 10
# model = tf_model.TypePredictor(bert_encoder, num_classes=N_CLASSES)
# print(tf_model.train(model, train_dataset, test_dataset, epochs=EPOCHS))

inputs = tf.keras.layers.Input(shape=(1024,))
outputs = tf.keras.layers.Dense(1)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss= 'sparse_categorical_crossentropy', metrics=['mae', "acc"])
model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)