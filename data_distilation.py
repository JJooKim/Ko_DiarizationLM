from Levenshtein import distance

from tqdm import tqdm




import json
import os
import re
from absl import app
from absl import flags
from utils_orch import *
from tqdm import tqdm
FLAGS = flags.FLAGS

flags.DEFINE_string('input', 'data.json', 'Root directory containing conversation data')
flags.DEFINE_string('output', 'data.json', 'Root directory containing conversation data')

# Threshold for similarity


# Function to calculate similarity
def similarity(ref_text1, ref_text2):
    # Normalize the Levenshtein distance to a value between 0 and 1
    max_length = max(len(ref_text1), len(ref_text2))
    if max_length == 0:
        return 1.0  # Both strings are empty, so they are similar
    return 1 - (distance(ref_text1, ref_text2) / max_length)

def get_distilation(data):
    threshold = 0.8
    # Group similar pairs
    similar_groups = []
    checked_indices = set()  # To keep track of indices that have been checked
    for i in tqdm(range(len(data))):
        if i not in checked_indices:
            similar_group = [i]  # Initialize a new group with the current index
            for j in range(i+1, len(data)):
                if j not in checked_indices:  # Check if the index has not been already checked
                    ref_text1 = data[i]['ref_text']
                    ref_text2 = data[j]['ref_text']
                    similarity_score = similarity(ref_text1, ref_text2)
                    if similarity_score >= threshold:
                        similar_group.append(j)
                        checked_indices.add(j)
            similar_groups.append(tuple(similar_group))

    # Print similar groups
    print("Similar groups:")
    for group in similar_groups:
        print(group)

    new_data = [data[group[0]] for group in similar_groups]
    return new_data
    

        
def distilation(input='data.json', output='data_distillation.json'):

    with open(input, 'r', encoding='utf-8-sig') as file:
        json_data = json.load(file)
    data = json_data['utterances']

    new_data = get_distilation(data)

    json_data['utterances'] = new_data # 없어도 될 듯
    with open(output, 'w', encoding='utf-8-sig') as file:
        json.dump(json_data, file, indent=2)
        
    return 0


def main(argv):
    del argv  # Unused
    input = FLAGS.input
    output = FLAGS.output
    distilation(input, output)

if __name__ == '__main__':
    app.run(main)

##### Usage
#python data_distilation.py --input data_D60.json --output data_D60_distil.json