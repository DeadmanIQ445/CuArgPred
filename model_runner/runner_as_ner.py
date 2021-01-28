import os
import tf_model_modified as tf_model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.metrics import precision_recall_fscore_support
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

tf.get_logger().setLevel('ERROR')

MAX_ARG_LENGTH =20
DS_PATH = "./data/_all_data.csv"
EPOCHS = 3
shuffle_buffer_size = 1000
SEQ_LENGTH = 512
BATCH_SIZE = 1
MODEL_PATH = "bert2"


with open(MODEL_PATH+ "/cubert_config.json") as conf_file:
    config_dict = json.loads(conf_file.read())
    bert_config = bert.configs.BertConfig.from_dict(config_dict)

bert_encoder = bert.bert_models.get_transformer_encoder(
    bert_config, sequence_length=SEQ_LENGTH)

checkpoint = tf.train.Checkpoint(model=bert_encoder)
checkpoint.restore(MODEL_PATH+'/bert1-1').assert_consumed()

if ".json" in DS_PATH:
    data = pd.read_json(DS_PATH)
else:
    data = pd.read_csv(DS_PATH)

tokenizer = python_tokenizer.PythonTokenizer()
subword_tokenizer = text_encoder.SubwordTextEncoder(MODEL_PATH + "/cuvocab.txt")

data_body1 = data['body.1']

## Preprocessign arg and labels

data['arg_types'] = data['arg_types'].apply(eval)
df_labels = pd.DataFrame(data['arg_types'].values.tolist())


max_label_length=10
df_labels2=df_labels.iloc[:,0:max_label_length]
df_labels2[pd.isnull(df_labels2)]  = 'NaN'

enc = preprocessing.LabelEncoder()
all_types = df_labels.apply(pd.Series).stack().values
enc.fit(all_types)

df3 = df_labels2.apply(enc.transform)
data['labels'] = df3.values.tolist()

# Dataset generator setup

def transform(code_text):
    return [2]+sum(code_to_subtokenized_sentences.code_to_cubert_sentences(
        code=code_text,
        initial_tokenizer=tokenizer,
        subword_tokenizer=subword_tokenizer),[])


def process_batch(data_batch):
    def process_elem(data_batch_i):
        id_list = np.zeros((SEQ_LENGTH))
        sentence_line = np.array(transform(data_batch_i['body.1'])[:SEQ_LENGTH-1]+[3])
        for labels, l_types in zip(eval(data_batch_i['arg_names']), data_batch_i['arg_types']):
            for label in labels: 
                label_sub = sum([subword_tokenizer.encode_without_tokenizing(word) for word in tokenizer.tokenize(label)],[])
                # for j in label_sub[:-1]: id_list[tuple(np.where(sentence_line == j))] = enc.transform([l_types])[0]
                id_list[tuple(np.where(sentence_line == label_sub[0]))] = enc.transform([l_types])[0]
        return sentence_line, id_list
    
    ids = [] # ner labels for sequence
    full_sentence = [] # here will be the end result of method tokenization
    for _,data_batch_i in data_batch.iterrows():
        sentence_line, id_list = process_elem(data_batch_i)
        full_sentence.append(sentence_line)
        ids.append(id_list)
    return full_sentence, ids



def gen():
    for _, data_batch in data.groupby(np.arange(len(data))//BATCH_SIZE):
        if len(data_batch) < BATCH_SIZE: continue # just an edge case
        full_sentence, ids = process_batch(data_batch)
        full_sentence = tf.ragged.constant(full_sentence)
        full_sentence = full_sentence.to_tensor(default_value=0, shape=[BATCH_SIZE, SEQ_LENGTH])
        ids = tf.convert_to_tensor(ids)
        yield ({'input_word_ids': full_sentence,
            'input_mask': ids > 0,
            'input_type_ids': tf.zeros_like(full_sentence),
            'lens': SEQ_LENGTH
        },ids)



dataset = tf.data.Dataset.from_generator(
        gen,
        ({"input_word_ids": tf.int32, "input_mask": tf.int32, "input_type_ids": tf.int32, "lens": tf.int32}, tf.int32),
        (
            {
                "input_word_ids": tf.TensorShape([BATCH_SIZE, SEQ_LENGTH]),
                "input_mask": tf.TensorShape([BATCH_SIZE, SEQ_LENGTH]),
                "input_type_ids": tf.TensorShape([BATCH_SIZE, SEQ_LENGTH]),
                "lens": None,
            },
            tf.TensorShape([BATCH_SIZE, SEQ_LENGTH]),
        ),
    )

DATASET_SIZE = len(df3)
train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

full_dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)

N_CLASSES = len(enc.classes_)
model = tf_model.TypePredictor(bert_encoder, num_classes=N_CLASSES)
print(tf_model.train(model, train_dataset, test_dataset, epochs=EPOCHS, scorer=precision_recall_fscore_support))

