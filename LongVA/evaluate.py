import json
import re
import random
from tqdm import tqdm
from pathlib import Path
from run import *


def prepare_model(args):
    #! Load Model, Tokenizer, Video processor
    model_path = "lmms-lab/LongVA-7B-DPO" # you can also set the device map to auto to accomodate more frames
    model, tokenizer, image_processor = load_model(model_path, device_map= "auto")
    return model, tokenizer, image_processor


def timestamp_to_seconds(timestamp):
    h, m, s = timestamp.split(":")
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds

def insert_subtitles_into_frames(frame_timestamps, subtitles, starting_timestamp_for_subtitles, duration):
    interleaved_list = []
    cur_i = 0

    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration

            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]

        for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
            if frame_timestamp <= subtitle_timestamp:
                # print("frame:", frame_timestamp)
                interleaved_list.append("<image>")
                cur_i += 1
            else:
                break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame_timestamp in frame_timestamps:
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break

        if covering_frames:
            # print("subtitle:", subtitle_timestamp, start, end)
            interleaved_list.append(subtitle_text)
        else:
            pass
            # print("leaving out subtitle:", start, end)

    for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
        interleaved_list.append("<image>")

    return "\n".join(interleaved_list)



def compute_frame_timestamps(duration, max_num_frames=16):
    if duration > max_num_frames:
        return [duration / max_num_frames * i for i in range(max_num_frames)]
    else:
        return [i for i in range(int(duration))]


def longvideobench_doc_to_text(meta, max_num_frames, data_path, subtitle=True):
    candidates = meta["candidates"]
    
    pre_prompt = ""
    post_prompt = "Answer with the option's letter from the given choices directly.\n"
    system_prompt = lambda x: f"""
        <|im_start|>
        system\n
        You are a helpful assistant.
        <|im_end|>\n
        <|im_start|>user\n
        <image>\n 
        {x} <|im_end|>\n
        <|im_start|>assistant\n
        """

    if subtitle:
        subtitle_path = Path(data_path) / "subtitles" / meta["subtitle_path"]
        with open(subtitle_path, "r") as f:
            subtitles = json.load(f)

        frame_timestamps = compute_frame_timestamps(meta["duration"], max_num_frames)
        interleaved_prefix = insert_subtitles_into_frames(
            frame_timestamps, 
            subtitles,
            meta["starting_timestamp_for_subtitles"],
            meta["duration"]
            )
        question = meta["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
        return f"{pre_prompt}{interleaved_prefix}\n{question}\n{post_prompt}"
    else:
        question = meta["question_wo_referring_query"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
        return system_prompt(f'{pre_prompt}\n{question}\n{post_prompt}')
 
def evaluate_multiple(model, tokenizer, image_processor, metadata, max_frames_num, json_name):
    
    for meta in tqdm(metadata):
        video_path = Path(DATA_PATH) / 'videos' / f"{meta['video_id']}.mp4"
        video_tensor = video_embed(
            str(video_path), image_processor, max_frames_num = max_frames_num, 
            ).to(model.device, dtype=model.dtype) # 57200
        text_info = longvideobench_doc_to_text(meta, max_num_frames=max_frames_num, data_path=DATA_PATH, subtitle=False)
        input_ids = tokenizer_image_token(text_info, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(video_tensor.device)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, images=[video_tensor], modalities=["video"], 
                do_sample=True, temperature=0.5, top_p=None, 
                num_beams=1, use_cache=True, max_new_tokens=1024
                )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        score_dict = longvideobench_process_results(meta, outputs)
        write_or_append_json(json_name, score_dict['lvb_acc'])


def write_or_append_json(file_path, new_data):
    """
    Appends a list of new data to a JSON file if it exists, or creates the file if it doesn't.

    Args:
        file_path (str): The path to the JSON file.
        new_data (dict): A list of data to write or append.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:            
            existing_data = json.load(f)
    else:
        existing_data = []
    existing_data.append(new_data)
    
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)
        
def evaluate_longvideobench(samples):
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        correct = eval_multi_choice(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}

def eval_multi_choice(gold_i, pred_i):
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct

def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Changed from MMMU-style complex parsing into simple parsing.
    Fixed to avoid 'D. A book' be parsed as A.
    Same as original LongVideoBench paper (from author Haoning Wu), if parsing failed, it will assign a random choice to model.
    """
    s = response.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return random.choice(all_choices)

    matches = re.search(r"[ABCDE]", s)
    if matches is None:
        return random.choice(all_choices)
    return matches[0]


def longvideobench_process_results(doc, results):
    pred = results[0]
    all_choices = []
    index2ans = {}
    candidates = doc["candidates"]
    for i, option in enumerate(candidates):
        index2ans[chr(ord("A") + i)] = option
        all_choices.append(chr(ord("A") + i))

    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    id = doc["id"]
    lvb_acc = {
        "id": id, 
        "duration_group": doc["duration_group"], 
        "question_category": doc["question_category"], 
        "answer": chr(ord("A") + doc["correct_choice"]), 
        "parsed_pred": parsed_pred
        }
    return {
        "lvb_acc": lvb_acc,
        "submission": {
            id: pred,
        },
    }
    
if __name__ == "__main__":
    DATA_PATH = "/home/server08/yoonjeon_workspace/VideoPrefill/LongVA/dataset/LongVideoBench"
    metadata_path = Path(DATA_PATH) / "lvb_val.json"
    max_frames_num = 100
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    model_path = "lmms-lab/LongVA-7B-DPO" # you can also set the device map to auto to accomodate more frames
    attn_type = "flexprefill" # None "minference" "flexprefill"
    model, tokenizer, image_processor = load_model(model_path, device_map= "cuda")
    
    json_name = "answers_vanilla.json" if attn_type is None else f"answers_{attn_type}.json"
    if os.path.exists(json_name):
        os.remove(json_name)
    evaluate_multiple(model, tokenizer, image_processor, metadata, max_frames_num, json_name)
    print("Evaluation done. Samples are saved in samples.json")
    with open("samples.json", "r") as f:
        samples = json.load(f)
    judge_dict, acc = evaluate_longvideobench(samples)
    print(judge_dict)
    print(acc)