import os
import re
import json
import random
from pathlib import Path
import torch
from decord import VideoReader, cpu
from PIL import Image

def timestamp_to_seconds(timestamp):
    h, m, s = timestamp.split(":")
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds

def insert_subtitles_into_frames(frames, frame_timestamps, subtitles, 
                                 starting_timestamp_for_subtitles, duration):
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

        
        for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
                if frame_timestamp <= subtitle_timestamp:
                    #print("frame:", frame_timestamp)
                    interleaved_list.append(frame)
                    cur_i += 1
                else:
                    break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame, frame_timestamp in zip(frames, frame_timestamps):
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break
        #
        if covering_frames:
            #print("subtitle:", subtitle_timestamp, start, end)
            interleaved_list.append(subtitle_text)
        else:
            pass
            #print("leaving out subtitle:", start, end)
        
    for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
        #print(frame_timestamp)
        interleaved_list.append(frame)
        
    return interleaved_list



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
        
def evaluate_longvideobench(samples, ids=None):
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        if ids is not None:
            if sample["lvb_acc"]["id"] not in ids:
                continue
        sample = sample["lvb_acc"]
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
        return ""
    return matches[0]


def longvideobench_process_results(doc, pred):
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

def summarize_results():
    for attn_json in ["answers_vanilla.json", "answers_minference.json", "answers_flexprefill.json"]:
        with open(attn_json, "r") as f:
            samples = json.load(f)
        judge_dict, acc = evaluate_longvideobench(samples)
        print(f"------------{attn_json}------------")
        # print(judge_dict)
        print(acc) #!
        