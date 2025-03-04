import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

def egoschema_run(args):
    data_path = "/home/server08/.cache/huggingface/egoschema"
    video_path = os.path.join(data_path, "videos")
    metadata_path = os.path.join(data_path, "Subset/test-00000-of-00001.parquet")
    metadata = pd.read_parquet(metadata_path)
    data = metadata.iloc[4]
    q_id = data.question_idx
    question = data.question
    video_idx = data.video_idx
    option = data.option
    answer = data.answer
    video_at = os.path.join(video_path, f"{video_idx}.mp4")
    frames, timesteps, total_frames = load_video(video_at, target_fps=None, max_num_frames=args.num_frames)
    save_at = f"examples/egoschema/{q_id}"
    if not os.path.exists(save_at):
        os.makedirs(f"examples/egoschema/{q_id}")
        for frame_idx, frame in enumerate(frames):
            frame.save(f"examples/egoschema/{q_id}/{frame_idx:04d}.png")
    full_prompt = f"{question}\n{option}\nAnswer with the option's letter from the given choices directly."
    reasoning = f"Summarize the content of the video. Identify the objects in the given question \n{question} \n. Then, find the objects from the video frames. Think about problem solving process with the given question and the video. Finally, choose one of the following options {option}"
    return frames, full_prompt, reasoning, reasoning, answer, video_idx

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

# We will unzip all the zip files
# To HF HOME cache dir
# And load it here
HF_HOME = os.environ["HF_HOME"] if "HF_HOME" in os.environ else os.path.expanduser("~/.cache/huggingface/hub")
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, cache_dir)
cache_dir = os.path.join(cache_dir, "videos")

from loguru import logger as eval_logger


# Pass in video path here
# Can only work correctly with video llm
def egoschema_doc_to_visual(doc):
    video_path = doc["video_idx"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# This is the place where you format your question
def egoschema_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    question = doc["question"]
    if "option" in doc:
        for op in doc["option"]:
            question += "\n" + op
        post_prompt = "\nAnswer with the option's letter from the given choices directly."

    return f"{pre_prompt}{question}{post_prompt}"


def egoschema_doc_to_answer(doc):
    return doc["answer"]


# Process result for mc_ppl
def egoschema_process_results(doc, result):
    # Initialize minimum value and index
    min_value = float("inf")
    min_index = -1

    # Iterate through the results to find the index of the lowest value
    for i, (value, _) in enumerate(result):
        if value < min_value:
            min_value = value
            min_index = i

    # Return the result with the index of the lowest value
    return {"submission": {doc["video_idx"]: min_index}, "score": {"pred": min_index, "ground_truth": doc["answer"]}}


def get_multi_choice_info(doc):
    all_choices = []
    index2ans = {}
    OPTIONS = ["A", "B", "C", "D", "E"]
    for i in range(5):
        # import pdb;pdb.set_trace()
        index2ans[OPTIONS[i]] = doc["option"][i].strip()
        all_choices.append(OPTIONS[i])

    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    ans_with_space = False
    ans_with_dot = False
    candidates = []
    # import pdb; pdb.set_trace()
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(f"({choice})")
            ans_with_brack = True

    # if len(candidates) == 0:
    for choice in all_choices:  # e.g., A B C D
        if f"{choice} " in response:
            candidates.append(f"{choice} ")
            ans_with_space = True

    # if len(candidates) == 0:
    for choice in all_choices:  # e.g., A. B. C. D.
        if f"{choice}." in response:
            candidates.append(f"{choice}.")
            ans_with_dot = True

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:

        start_indexes = []
        if index_ans:
            for can in candidates:
                index = response.rfind(can)
                start_indexes.append(index)  # -1 will be ignored anyway
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the first one
        pred_index = candidates[np.argmin(start_indexes)]
        pred_index = pred_index.replace("(", "").replace(")", "").replace(".", "").strip()
    else:  # if only one candidate, use it.
        pred_index = candidates[0]
        pred_index = pred_index.replace("(", "").replace(")", "").replace(".", "").strip()

    return pred_index, len(candidates) > 0


# Process result for mcq answer generation
def egoschema_process_results_generation(doc, result):
    # import pdb;pdb.set_trace()
    pred = result[0]

    index2ans, all_choices = get_multi_choice_info(doc)
    parsed_pred, matched_tag = parse_multi_choice_response(pred, all_choices, index2ans)

    pred_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    index = pred_to_index.get(parsed_pred, -1)  # Default to -1 if the prediction is not found

    return {"submission": {doc["video_idx"]: index}, "score": {"pred": index, "ground_truth": doc["answer"]}}

def egoschema_aggregate_score(results, args):
    yes_count = 0

    # results is a list of dict
    for answer_dict in results:
        if str(answer_dict["ground_truth"]) == str(answer_dict["pred"]):
            yes_count = yes_count + 1

    accuracy = yes_count / len(results)

    return accuracy


def egoschema_doc_to_choice(doc):
    return [op.split(".")[1].strip() for op in doc["option"]]