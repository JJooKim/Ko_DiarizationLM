
import pandas as pd
from datasets import Dataset, DatasetDict

import json
import os
import re
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('train_path', 'data_D60.csv', '')
flags.DEFINE_string('test_path', 'data_D61.csv', '')
flags.DEFINE_string('dict_name', 'jjookim/kodiarization_lm', '')

def make_hf_dataset_dict(train_df, test_df, dict_name):
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    data_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    data_dict.push_to_hub(dict_name, private=True)

def dataset_maker(train_path, test_path, dict_name):
    # train data
    train = pd.read_csv(train_path, encoding='utf-8')
    #train['prompt'] = train['prompt'].str.replace('\\\\n', '\n')
    train['chat_sample'] = train['prompt'] + " " + train['completion']

    test = pd.read_csv(test_path, encoding='utf-8')
    #test['prompt'] = test['prompt'].str.replace('\\\\n', '\n')
    test['chat_sample'] = test['prompt'] + " " + test['completion']

    make_hf_dataset_dict(train, test, dict_name)

    
def main(argv):
    del argv  # Unused
    train_path = FLAGS.train_path
    test_path = FLAGS.test_path
    dict_name = FLAGS.dict_name

    dataset_maker(train_path, test_path, dict_name)


if __name__ == '__main__':
    app.run(main)

##### Usage
#python prompt_maker.py --train_path .....