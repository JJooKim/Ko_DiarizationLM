import json
import os
import re


from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input', 'Data/Training/D60/J91', 'Root directory containing conversation data')
flags.DEFINE_string('output', 'data.json', 'Output JSON file path')


def get_ref_text(input_text):
    # Extract the latter part after "/"
    processed_text = re.sub(r'\([^)]+\)/\(([^)]+)\)', r'\1', input_text)

    # Remove non-Korean characters
    processed_text = re.sub(r'[^가-힣\s]', '', processed_text)

    # Removing extra whitespaces and trimming
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()

    return processed_text


def get_ref_data(directory_path): # eg.directory_path = Data/Training/D60/J91/S12345678

    json_file = os.path.basename(directory_path) + '.json'
    json_path = os.path.join(directory_path, json_file)

    with open(json_path, 'r', encoding='utf-8-sig') as i_file:
        i_json = json.load(i_file) # i_json: conversation Information Json


    ref_texts = []
    ref_spks = []
    spk_ids = [speaker["id"] for speaker in i_json["dataSet"]["typeInfo"]["speakers"]] #화자 고유 id

    # Conversation Drop Case 1 (크라우드 소싱과정에서 음성 정보 누락)
    ##################################################################################################
    # GroundTruth Data의 화자 전환 정보이다.(정확히는 001.wav~마지막.wav 까지 각각 파일의 화자를 담은 리스트)
    # spk_turn의 비율이 각각 0.4 보다 낮으면 온전한 데이터가 아니라 판단하고 Drop 한다.
    spk_turns = [spk_ids.index(dialog["speaker"]) + 1 for dialog in i_json["dataSet"]["dialogs"]]
    # Count occurrences of each value
    count_1 = spk_turns.count(1)
    count_2 = spk_turns.count(2)
    # Calculate the ratio
    ratio_1 = count_1 / len(spk_turns)
    ratio_2 = count_2 / len(spk_turns)
    if ratio_1 <= 0.4 or ratio_2 <= 0.4:
        print("1. Unbalanced ratio")
        return 1
    ###################################################################################################

    # Conversation Drop Case 2 (업체에서 실제 음성 파일, 텍스트 파일을 직접 누락 시킨 경우)
    ###################################################################################################
    text_num = []
    for dialog in i_json["dataSet"]["dialogs"]:
        text_num.append(dialog["textPath"][-8:-4])
    if len(text_num) != int(text_num[-1]):
        print(f"2. Text file missing")
        return 2
    ###################################################################################################


    for dialog in i_json["dataSet"]["dialogs"]:
        text_path = os.path.join(directory_path, dialog["textPath"][-8:])
        if os.path.exists(text_path):
            text = open(text_path, 'r', encoding='utf-8-sig').read()
        else:
            print(f"3. Text path not found: {text_path}")
            return 3 # Conversation Drop Case 3 (존재 해야 할 파일이 존재하지 않는 경우)
        
        text = get_ref_text(text)
        ref_texts.append(text)
        ref_spks.extend([str(spk_ids.index(dialog["speaker"]) + 1)] * len(text.split()))

    ref_text = ' '.join(ref_texts)
    ref_spk = ' '.join(ref_spks)

    # Error Case 4번인데 있을 수 없는 경우임 
    if len(ref_text.split()) != len(ref_spk.split()):
        print(f"4. ref_text & ref_spk length missmatch")
        return 4
    return {
        "utterance_id": directory_path,
        "ref_text": ref_text,
        "ref_spk": ref_spk,
    }





def get_ref_json(input='Data/Training/D60/J91', output = 'data.json'):

    data = {"utterances": []}
    error_case = {
        1: 0,
        2: 0,
        3: 0,
        4: 0
    }
    for root, dirs, files in os.walk(input):
        if dirs == []: break
        for directory in dirs:
            s_json = get_ref_data(os.path.join(root, directory)) # json data for 1 conversation
            
            if s_json in [1, 2, 3, 4]:
                error_case[s_json] += 1
                continue

            data["utterances"].append(s_json)

    with open(output, 'w', encoding='utf-8-sig') as data_file:
        json.dump(data, data_file, indent=2, ensure_ascii=False)

    return error_case


def main(argv):
    del argv  # Unused
    input = FLAGS.input
    output = FLAGS.output
    error_case = get_ref_json(input, output)
    print("Error cases:", error_case)


if __name__ == '__main__':
    app.run(main)

##### Usage
#python get_ref_json.py --input YourRootDirectory --output YourOutputFile.json