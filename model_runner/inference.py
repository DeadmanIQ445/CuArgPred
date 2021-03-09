import os
import tf_model_modified as tf_model

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import ast
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from official.nlp import bert
from cubert_tokenizer import python_tokenizer, code_to_subtokenized_sentences
import official.nlp.bert.bert_models
import official.nlp.bert.configs
from tensor2tensor.data_generators import text_encoder
import pandas as pd
import numpy as np
import json
from sklearn import preprocessing

tf.get_logger().setLevel('ERROR')

DS_PATH = "./data/_all_data2.csv"
EPOCHS = 3
shuffle_buffer_size = 10000
SEQ_LENGTH = 512
MODEL_PATH = "./bert2"
FREQ_LIMIT = 100
FREQ_CUT_SYMBOL = "<UNK>"
NaN_symbol = ''
with open(MODEL_PATH + "/cubert_config.json") as conf_file:
    config_dict = json.loads(conf_file.read())
#     bert_config = bert.configs.BertConfig.from_dict(config_dict)

# bert_encoder = bert.bert_models.get_transformer_encoder(
#     bert_config, sequence_length=SEQ_LENGTH)
# bert_encoder.trainable = False
# checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
# checkpoint.restore(MODEL_PATH + '/bert2_2-1').assert_consumed()

model = tf.saved_model.load('./model')


tokenizer = python_tokenizer.PythonTokenizer()
# This tokenizer is not used for splitting but only for encoding, PythonTokenizer does subword tokenization
subword_tokenizer = text_encoder.SubwordTextEncoder(MODEL_PATH + "/cuvocab.txt")

CLS = subword_tokenizer.encode_without_tokenizing("[CLS]")
SEP = subword_tokenizer.encode_without_tokenizing("[SEP]")

enc = preprocessing.LabelEncoder()
enc.classes_ = np.load('classes.npy', allow_pickle=True)
FREQ_CUT_ENC = enc.transform([FREQ_CUT_SYMBOL])
NaN_enc = enc.transform([NaN_symbol])

print(f'Enc for "NaN" {NaN_enc}, Enc for FREQ_CUT_SYMBOL {FREQ_CUT_ENC}')

# Dataset generator setup

def transform(code_text, labels):
    code_encoded, respective_labels = code_to_subtokenized_sentences.code_to_cubert_sentences(
        code=code_text,
        initial_tokenizer=tokenizer,
        subword_tokenizer=subword_tokenizer, labels=labels)
    return CLS + code_encoded, [0] + respective_labels


def get_names(src):
    ret = []
    try:
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.arg):
                ret.append(node.arg)
        return ret
    except:
        print("Could Not process the code")
        return ret

def process_elem(code):
    args = get_names(code)
    labels = dict(zip(args, [1]*len(args)))
    line_encoded, id_list = transform(code, labels)
    sentence_line = line_encoded[:SEQ_LENGTH - 1] + SEP
    le = len(sentence_line)
    sentence_line = np.array(sentence_line + [0]*(SEQ_LENGTH-len(sentence_line)))
    id_list = np.array(id_list[:SEQ_LENGTH - 1] + [0]*(SEQ_LENGTH-le+1))
    return sentence_line, id_list, le

def gen(code):
    full_sentence, ids, le = process_elem(code)
    full_sentence = tf.ragged.constant(full_sentence)
    ids = tf.ragged.constant(ids)
    return ({'input_word_ids': tf.cast(tf.reshape(full_sentence,[1, SEQ_LENGTH]), dtype=tf.int32),
            'input_mask': tf.cast(tf.reshape(ids > 0,[1, SEQ_LENGTH]),dtype=tf.int32),
            'input_type_ids': tf.cast(tf.reshape(tf.zeros_like(full_sentence),[1, SEQ_LENGTH]),dtype=tf.int32),
            }, ids, le)

def predict(code):
    encoded = gen(code)
    logits = model.call(encoded[0])
    mask = tf.sequence_mask(encoded[2], SEQ_LENGTH)
    mask = tf.math.logical_and(mask, encoded[0]['input_mask']>0)
    masked_pred = tf.boolean_mask(logits, mask)
    argmax = tf.math.argmax(logits, axis=-1)
    estimated_labels = tf.cast(tf.boolean_mask(argmax, mask), tf.int32)
    print(enc.inverse_transform(estimated_labels.numpy()),  [ enc.inverse_transform(i.numpy()) for i in  tf.nn.top_k(masked_pred, k=5).indices])
N_CLASSES = len(enc.classes_)
print(predict("def abc(a,b):\n return a+b"))
# model = tf_model.TypePredictor(bert_encoder, num_classes=N_CLASSES)
# print(tf_model.train(model, train_dataset, test_dataset, epochs=EPOCHS, scorer=precision_recall_fscore_support, learning_rate=0.00001, decrease_every=2000, learning_rate_decay=1.0,report_every=250, lower_bound = 1.5e-5))
