import json
import os
import re
from absl import app
from absl import flags
from utils_orch import *
from tqdm import tqdm
FLAGS = flags.FLAGS

flags.DEFINE_string('input', 'data.json', 'Root directory containing conversation data')

# def drop_spk_3(data):
#     for utter in data:
#         if '3' in utter['hyp_spk']: 
#             data.remove(utter)

def get_oracle_hyp_spk(data):
    for utter in tqdm(data):
        utter["hyp_spk_oracle"] = transcript_preserving_speaker_transfer(\
            utter['ref_text'], utter['ref_spk'], utter['hyp_text'], utter['hyp_spk'])


def get_diarized_text(data):
    for utter in tqdm(data):
        utter["ref_diarized_text"] = create_diarized_text(utter['ref_text'].split(), utter['ref_spk'].split())
        utter["hyp_diarized_text"] = create_diarized_text(utter['hyp_text'].split(), utter['hyp_spk'].split())
        utter["hyp_diarized_text_oracle"] = create_diarized_text(utter['hyp_text'].split(), utter['hyp_spk_oracle'].split())
        

def orchestration(input='data.json'):

    with open(input, 'r', encoding='utf-8-sig') as file:
        json_data = json.load(file)
    data = json_data['utterances']

    #drop_spk_3(data)
    get_oracle_hyp_spk(data)
    get_diarized_text(data)


    json_data['utterances'] = data # 없어도 될 듯
    with open(input, 'w', encoding='utf-8-sig') as file:
        json.dump(json_data, file, indent=2)
        
    return 0


def main(argv):
    del argv  # Unused
    input = FLAGS.input
    orchestration(input)

if __name__ == '__main__':
    app.run(main)

##### Usage
#python orchestration.py --input data_D60.json