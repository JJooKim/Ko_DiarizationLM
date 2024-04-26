import whisperx
import os
import json
from tqdm import tqdm
from utils_hyp import * 


ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)


def get_hyp_json(audio_path, whisper_model, alignment_model, metadata,batch_size=8, language="ko", device="cuda"):
    enable_stemming = False

    rename_file(audio_path)

    ## 1. Separating music from speech using Demucs
    if enable_stemming:
        # Isolate vocals from the rest of the audio

        return_code = os.system(
            f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "temp_outputs"'
        )

        if return_code != 0:
            logging.warning("Source splitting failed, using original audio file.")
            vocal_target = audio_path
        else:
            vocal_target = os.path.join(
                "temp_outputs",
                "htdemucs",
                os.path.splitext(os.path.basename(audio_path))[0],
                "vocals.wav",
            )
    else:
        vocal_target = audio_path

    ## 2.Transcriping audio using Whisper and realligning timestamps using Wav2Vec2
    whisper_results, language = transcribe_batched(
        vocal_target,
        language,
        batch_size,
        device,
        whisper_model
    )


    ## 3.Aligning the transcription with the original audio using Wav2Vec2
    result_aligned = whisperx.align(
        whisper_results, alignment_model, metadata, vocal_target, device
    )
    word_timestamps = filter_missing_timestamps(result_aligned["word_segments"])

    # clear gpu vram
    #del alignment_model
    torch.cuda.empty_cache()

    # 4.
    sound = AudioSegment.from_file(vocal_target).set_channels(1)
    output_file_path = os.path.join(temp_path, "mono_file.wav")
    sound.export(output_file_path, format="wav")
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path, DOMAIN_TYPE="telephonic")).to("cuda")
    msdd_model.diarize()
    os.remove(output_file_path)
    del msdd_model
    torch.cuda.empty_cache()


    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    #wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    #ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
    return wsm



import os
import wave
from moviepy.editor import concatenate_audioclips, AudioFileClip

def concatenate_wav_files(wave_path, output_file, whisper_model, alignment_model, metadata):

    wave_files = [f"{wave_path.replace('_d_', '_q_')}.wav", f"{wave_path.replace('_d_', '_a_')}.wav"]


    # if 'full.wav' in wave_files:
    #     wave_files.remove('full.wav')
    # # wav_files.sort()  # Sort files in ascending order
    clips = [AudioFileClip('./'+wav_file) for wav_file in wave_files]
    final_clip = concatenate_audioclips(clips)
    final_clip.write_audiofile(output_file)
    wsm = get_hyp_json(output_file, whisper_model, alignment_model, metadata)
    os.remove(output_file)
    return wsm





def add_hyp_values(input = 'datainterview.json'):

    with open(input, 'r', encoding='utf-8-sig') as file:
        json_data = json.load(file)

    # whisper model
    whisper_model = whisperx.load_model(
        "large-v3",
        device="cuda",
        compute_type="float16",
        asr_options={"suppress_numerals": True},
        language="ko"
    )

    # allignment_model
    alignment_model, metadata = whisperx.load_align_model(
    language_code="ko", device="cuda"
    )


    utterances = json_data['utterances']
    i = 0
    for utterance in utterances: # 너무 길면 잘라서
        print(i)
        i += 1
        wav_path = utterance["utterance_id"] # eg. "Data/Training/D60/J91/S00009134" so the wav and txt file were together
        output_file = 'full.wav'
        wsm = concatenate_wav_files(wav_path, output_file, whisper_model, alignment_model, metadata)
        # hyp_texts = [re.sub(r'[^가-힣]', '', entry['word']) for entry in wsm]
        # hyp_spks = [str(int(entry['speaker']) + 1) for entry in wsm]
        hyp_texts = []
        hyp_spks = []
        for entry in wsm:
            txt = re.sub(r'[^가-힣]', '', entry['word'])
            if txt == '':
                continue
            else:
                hyp_texts.append(txt)
                hyp_spks.append(str(int(entry['speaker']) + 1))
                

        utterance['hyp_text'] = ' '.join(hyp_texts)
        utterance['hyp_spk'] = ' '.join(hyp_spks)
        if (i % 50) == 0:
            with open(input, 'w', encoding='utf-8-sig') as file:
                json.dump(json_data, file, indent=2)
        
    with open(input, 'w', encoding='utf-8-sig') as file:
        json.dump(json_data, file, indent=2)
add_hyp_values('datainterview.json')
