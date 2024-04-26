import json
import os
import re
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input', 'Data/Validation/Management_Male_new', 'Root directory containing conversation data')
flags.DEFINE_string('output', 'datainterview.json', 'Output JSON file path')


def get_ref_text(input_text):
    # # Extract the latter part after "/"
    # processed_text = re.sub(r'\([^)]+\)/\(([^)]+)\)', r'\1', input_text)

    # Remove non-Korean characters
    processed_text = re.sub(r'[^가-힣\s]', ' ', input_text)

    # Removing extra whitespaces and trimming
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()

    return processed_text


def get_ref_data(json_path): # eg.directory_path = Data/Training/D60/J91/S12345678
    print(json_path)
    with open(json_path, 'r', encoding='utf-8-sig') as i_file:
        i_json = json.load(i_file) # i_json: conversation Information Json

    question = get_ref_text(i_json['dataSet']['question']['raw']['text'])
    answer = get_ref_text(i_json['dataSet']['answer']['raw']['text'])


    ref_spks = []
    ref_spks.extend(['1'] * len(question.split()))
    ref_spks.extend(['2'] * len(answer.split()))
    ref_spk = ' '.join(ref_spks)
    ref_text = question + ' ' + answer

    duration = i_json['rawDataInfo']['question']['duration'] + i_json['rawDataInfo']['answer']['duration']

    # Error Case 
    if len(ref_text.split()) != len(ref_spk.split()):
        raise ValueError("Number of words in the reference text does not match the number of speakers in the reference speaker list.")
    return {
        "utterance_id": json_path[:-5],
        "ref_text": ref_text,
        "ref_spk": ref_spk,
        "duration": duration,
    }





def get_ref_json(input='InterviewData/Validation/Management_Male_new', output = 'data.json'):
    data = {"utterances": []}

    for root, dirs, files in os.walk(input):
        for file in files:

            if file.endswith('.json'):
                print(os.path.join(root, file))
                s_json = get_ref_data(os.path.join(root, file)) # this is a json file innit?

                data["utterances"].append(s_json)



    with open(output, 'w', encoding='utf-8-sig') as data_file:
        json.dump(data, data_file, indent=2, ensure_ascii=False)

    return 0 # used to be error case


def main(argv):
    del argv  # Unused
    input = FLAGS.input
    output = FLAGS.output
    error_case = get_ref_json(input, output)



if __name__ == '__main__':
    app.run(main)

##### Usage
#python get_ref_json_inter.py --input InterviewData/Validation/Management_Male_new --output datainterview.json