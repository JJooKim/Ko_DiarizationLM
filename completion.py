import json
import os
import re
from absl import app
from absl import flags
from utils_orch import *
from tqdm import tqdm
FLAGS = flags.FLAGS

flags.DEFINE_string('input', 'data.json', 'Root directory containing conversation data')

def get_completion_spk(data):
    for utter in tqdm(data):
        utter["1_epoch_spk_completion"] = transcript_preserving_speaker_transfer(\
            utter['1_epoch_text'], utter['1_epoch_spk'], utter['hyp_text'], utter['hyp_spk'])



def get_diarized_text_llm(data):
    for utter in tqdm(data):
        utter["1_epoch_diarized_text"] = create_diarized_text(utter['hyp_text'].split(), utter['1_epoch_spk_completion'].split())
        
def completion(input='data.json'):

    with open(input, 'r', encoding='utf-8-sig') as file:
        json_data = json.load(file)
    data = json_data['utterances']

    get_completion_spk(data)
    get_diarized_text_llm(data)

    json_data['utterances'] = data # 없어도 될 듯
    with open(input, 'w', encoding='utf-8-sig') as file:
        json.dump(json_data, file, indent=2)
        
    return 0


def main(argv):
    del argv  # Unused
    input = FLAGS.input
    completion(input)

if __name__ == '__main__':
    app.run(main)

##### Usage
#python completion.py --input data_D60.json