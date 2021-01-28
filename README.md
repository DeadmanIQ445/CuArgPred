This is based on TF2.

The model lies [here](https://www.dropbox.com/s/pmjvf1jmg8ih087/bert2.zip?dl=0). This is CuBERT model that was transformed from tf1 to tf2.
The dataset is located [here](https://www.dropbox.com/s/pmjvf1jmg8ih087/bert2.zip?dl=0).

'model_runner' folder contains the scripts that train the model. 'runner_as_ner.py' preprocesses and prepares the dataset and loads CuBERT. 'tf_model_modified.py' trains the model and was based on Vitaly Romanov's work. Now it is somewhat changed and might have been broken in the process of changing.