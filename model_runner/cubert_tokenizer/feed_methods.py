import json
import code_to_subtokenized_sentences
from tensor2tensor.data_generators import text_encoder
from cubert_tokenizer import python_tokenizer
from pathlib import Path
import pandas as pd


DS_PATH = "data/_all_data.csv"
if ".json" in DS_PATH:
    is_json=True
elif ".csv" in DS_PATH:
    is_json=False
else:
    is_json=False

tokenizer = python_tokenizer.PythonTokenizer()
subword_tokenizer = text_encoder.SubwordTextEncoder("cubert_model_load/cuvocab.txt")

output_path = "data/resulting_ds/embed.jsonl"
filename = Path("/".join(output_path.split('/')[:-1]))
filename.mkdir(parents=True, exist_ok=True)

if not is_json:
    data = pd.read_csv(DS_PATH)
else:
    data = pd.read_json(DS_PATH)

with open(output_path,'w+') as output_file:
    for code in data['body.1']:
        subtokenized_sentences = code_to_subtokenized_sentences.code_to_cubert_sentences(
            code=code,
            initial_tokenizer=tokenizer,
            subword_tokenizer=subword_tokenizer)
        output_file.write(json.dumps(subtokenized_sentences, indent=2))

# with open(DS_PATH, 'r') as dataset_jsonl, open(output_path,'w+') as output_file:
#     for i in dataset_jsonl:
#         code = json.loads(i)['text']
#         subtokenized_sentences = code_to_subtokenized_sentences.code_to_cubert_sentences(
#             code=code,
#             initial_tokenizer=tokenizer,
#             subword_tokenizer=subword_tokenizer)
#         output_file.write(json.dumps(subtokenized_sentences, indent=2))