import json
import os
import re
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input', 'data.json', 'Root directory containing conversation data')




def has_multiple_spaces(input_string):
    pattern = re.compile(r'\s{2,}')  # Match two or more consecutive whitespaces
    
    return bool(pattern.search(input_string))

def dropcase5(input='data.json'):

    with open(input, 'r', encoding='utf-8-sig') as file:
        json_data = json.load(file)
    data = json_data['utterances']

    del_li = []
    i = 0
    for utter in data:
        if has_multiple_spaces(utter['ref_text']):
            del_li.append(i)
        i +=1

    for i in sorted(del_li, reverse=True):
        data.pop(i)

    json_data['utterances'] = data # 없어도 될 듯
    with open(input, 'w', encoding='utf-8-sig') as file:
        json.dump(json_data, file, indent=2)
        
    return del_li


def main(argv):
    del argv  # Unused
    input = FLAGS.input
    error_case = dropcase5(input)
    print("Error cases:", error_case)


if __name__ == '__main__':
    app.run(main)

##### Usage
#python dropcase_5.py --input YourInputFile.json