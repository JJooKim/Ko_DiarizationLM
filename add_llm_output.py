import json
import os
import re
from absl import app
from absl import flags
from datasets import load_dataset
FLAGS = flags.FLAGS

flags.DEFINE_string('input', 'data.json', 'Root directory containing conversation data')


def extract_text_and_spk(
    completions: str
) -> tuple[str, str]:
  """Extract the text and spk from the completions string."""
  spk = "1"
  previous_spk = "1"
  result_text = []
  result_spk = []
  for word in completions.replace("<", " <") .split():
    if word.startswith('<화자:'):
      if not word.endswith('>'):
        word += '>'
      spk = word[len('<화자:'):-len('>')]
      # Handle undefined behaviors of non-recognizable spk with a placeholder.
      try:
        spk_int = int(spk)
        if not spk or spk_int < 1 or spk_int > 10:
          raise ValueError("Seeing unexpected word: ", word)
        previous_spk = spk
      except ValueError:
        print("Skipping meaningless speaker token:", word)
        spk = previous_spk
    else:
      result_text.append(word)
      result_spk.append(spk)
  return " ".join(result_text), " ".join(result_spk)


def get_1epoch_output(input='data.json'):

    with open(input, 'r', encoding='utf-8-sig') as file:
        json_data = json.load(file)
    data = json_data['utterances']

    llm_output = load_dataset("jjookim/kodiarization_test")['test']


    decoded_llm_dict = {}

    for example in llm_output:
        utterance_id = example['utterance_id']
        decoded_llm = example['decoded_llm']

        # Extract the 'utterance_id' without segment information
        base_utterance_id = '_'.join(utterance_id.split('_')[:-1])

        # Append the 'decoded_llm' value to the string in the dictionary
        if base_utterance_id in decoded_llm_dict:
            decoded_llm_dict[base_utterance_id] += decoded_llm
        else:
            decoded_llm_dict[base_utterance_id] = decoded_llm
     
    for utter in data:
        utter['1_epoch_text'], utter['1_epoch_spk'] = extract_text_and_spk(decoded_llm_dict[utter['utterance_id']])



    json_data['utterances'] = data # 없어도 될 듯
    with open(input, 'w', encoding='utf-8-sig') as file:
        json.dump(json_data, file, indent=2)
        
    return 0

def main(argv):
    del argv  # Unused
    input = FLAGS.input
    get_1epoch_output(input)



if __name__ == '__main__':
    app.run(main)
