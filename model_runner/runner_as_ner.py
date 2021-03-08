import os
import tf_model_modified as tf_model

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from official.nlp import bert
from cubert_tokenizer import python_tokenizer, code_to_subtokenized_sentences
import official.nlp.bert.bert_models
import official.nlp.bert.configs
from tensor2tensor.data_generators import text_encoder
import pandas as pd
import json
from sklearn import preprocessing

tf.get_logger().setLevel('ERROR')

DS_PATH = "./data/_all_data2.csv"
EPOCHS = 3
shuffle_buffer_size = 10000
SEQ_LENGTH = 512
BATCH_SIZE = 1
MODEL_PATH = "./bert2"
FREQ_LIMIT = 100
FREQ_CUT_SYMBOL = "<UNK>"
NaN_symbol = ''
with open(MODEL_PATH + "/cubert_config.json") as conf_file:
    config_dict = json.loads(conf_file.read())
    bert_config = bert.configs.BertConfig.from_dict(config_dict)

bert_encoder = bert.bert_models.get_transformer_encoder(
    bert_config, sequence_length=SEQ_LENGTH)
bert_encoder.trainable = False
checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
checkpoint.restore(MODEL_PATH + '/bert2_2-1').assert_consumed()

data = pd.read_csv(DS_PATH)

tokenizer = python_tokenizer.PythonTokenizer()
# This tokenizer is not used for splitting but only for encoding, PythonTokenizer does subword tokenization
subword_tokenizer = text_encoder.SubwordTextEncoder(MODEL_PATH + "/cuvocab.txt")

CLS = subword_tokenizer.encode_without_tokenizing("[CLS]")
SEP = subword_tokenizer.encode_without_tokenizing("[SEP]")

# Preprocessing arg and labels
data['arg_types'] = data['arg_types'].apply(eval)
data = data[data.arg_types.astype(bool)]
df_labels = pd.DataFrame(data['arg_types'].values.tolist())

df_labels[pd.isnull(df_labels)]  = NaN_symbol
df_labels = df_labels.apply(lambda x: x.mask(x.map(x.value_counts())<FREQ_LIMIT, FREQ_CUT_SYMBOL))
enc = preprocessing.LabelEncoder()
all_types = df_labels.apply(pd.Series).stack().values
enc.fit(all_types)

FREQ_CUT_ENC = enc.transform([FREQ_CUT_SYMBOL])
NaN_enc = enc.transform([NaN_symbol])
print(f'Enc for "NaN" {NaN_enc}, Enc for FREQ_CUT_SYMBOL {FREQ_CUT_ENC}')
df3 = df_labels.apply(enc.transform)
data['labels'] = df3.values.tolist()

def la(data_batch_i):
  r = []
  for i in data_batch_i:
        if not ((i == NaN_enc[0] or i==FREQ_CUT_ENC[0]) and len(data_batch_i)==1):
            r.append(i)
  if len(r) == 0:
    return pd.NA
  return r

data['labels'] = data['labels'].apply(la)
data = data.dropna(subset=['labels'], axis=0)

def train_test_by_repo(data, split=0.75):
    train_l = []
    test_l = []
    c = 0
    train_len = split * len(data)
    for name, i in data.groupby(['repo']).count().sample(frac=1).iterrows():
        if train_len > c:
            train_l.append(name)
            c += i['author']
        else:
            test_l.append(name)
    return data.loc[data['repo'].isin(train_l)], data.loc[data['repo'].isin(train_l)]


train_ds, test_ds = train_test_by_repo(data)

print(train_ds.shape)
# Dataset generator setup

def transform(code_text, labels):
    code_encoded, respective_labels = code_to_subtokenized_sentences.code_to_cubert_sentences(
        code=code_text,
        initial_tokenizer=tokenizer,
        subword_tokenizer=subword_tokenizer, labels=labels)
    return CLS + code_encoded, [0] + respective_labels



def process_elem(data_batch_i):
    labels = dict(zip(eval(data_batch_i['arg_names']), data_batch_i['labels']))
    line_encoded, id_list = transform(data_batch_i['body'], labels)
    sentence_line = line_encoded[:SEQ_LENGTH - 1] + SEP
    le = len(sentence_line)
    sentence_line = np.array(sentence_line + [0]*(SEQ_LENGTH-len(sentence_line)))
    id_list = np.array(id_list[:SEQ_LENGTH - 1] + [0]*(SEQ_LENGTH-le+1))
    return sentence_line, id_list, le

def create_dataset(dataset):
    def gen():
        for _,data_batch in dataset.iterrows():
            full_sentence, ids, le = process_elem(data_batch)
            full_sentence = tf.ragged.constant(full_sentence)
            ids = tf.ragged.constant(ids)
            yield ({'input_word_ids': full_sentence,
                    'input_mask': ids > 0,
                    'input_type_ids': tf.zeros_like(full_sentence),
                    }, ids, le)

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_word_ids": tf.int32, "input_mask": tf.int32, "input_type_ids": tf.int32}, tf.int32, tf.int32),
        (
            {
                "input_word_ids": tf.TensorShape([SEQ_LENGTH]),
                "input_mask": tf.TensorShape([SEQ_LENGTH]),
                "input_type_ids": tf.TensorShape([SEQ_LENGTH])
            },
            tf.TensorShape([SEQ_LENGTH]),
            None
        ),
    )

train_dataset = create_dataset(train_ds)
test_dataset = create_dataset(test_ds)
train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
# train_dataset = train_dataset.shuffle(buffer_size)
# test_dataset = test_dataset.shuffle(buffer_size)
N_CLASSES = len(enc.classes_)

model = tf_model.TypePredictor(bert_encoder, num_classes=N_CLASSES)
print(tf_model.train(model, train_dataset, test_dataset, epochs=EPOCHS, scorer=precision_recall_fscore_support, learning_rate=0.00075, decrease_every=2500, learning_rate_decay=0.5,report_every=250, lower_bound = 1.5e-5))
