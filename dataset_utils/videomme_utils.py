
import os
import re
import cv2
import numpy as np
from datasets import load_dataset
from utils.video_utils import load_video

VIDEO_TYPE = ["short", "medium", "long"]
CATEGORIES = ["Knowledge", "Film & Television", "Sports Competition", "Artistic Performance", "Life Record", "Multilingual"]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual",
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]

replace_prompt = " Please answer yes or no."

def videomme_run(args):
    
    data_path = "/home/server08/.cache/huggingface/videomme"
    video_path = os.path.join(data_path, "data")
    subtitle_path = os.path.join(data_path, "subtitle")
    dataset = load_dataset("lmms-lab/Video-MME", "videomme")
    data = dataset['test'][0]
    video_id = data['video_id']
    duration = data['duration']
    domain = data['domain']
    sub_category = data['sub_category']
    question = data['question']
    options = data['options']
    answer = data['answer']
    videoID = data['videoID']
    frames, timesteps, total_frames = load_video(os.path.join(video_path, f"{videoID}.mp4"), target_fps=None, max_num_frames=args.num_frames)
    save_at = f"examples/videomme/{video_id}"
    if not os.path.exists(save_at):
        os.makedirs(save_at)
        for frame_idx, frame in enumerate(frames):
            frame.save(f"examples/videomme/{video_id}/{frame_idx:04d}.png")
            
    if args.subtitle:
        #! NEED TO COMPLETE
        import numpy as np
        subtitles_prompt = "This video's subtitles are listed below: \n"
        subtitle_by_frame = extract_subtitles(video_path, os.path.join(subtitle_path, f"{videoID}.srt"))
        uniform_sampled_frames = np.linspace(0, total_frames - 1, args.num_frames, dtype=int).tolist()

        subtitle_by_frame_idx = []
        for frame_idx in uniform_sampled_frames:
            for idx, title in enumerate(subtitle_by_frame):
                if frame_idx < title[1] and frame_idx >= title[0]:
                    subtitle_by_frame_idx.append(idx)
        subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

        textlist = []
        for idx in subtitle_by_frame_idx:
            pattern = r'<font color="white" size=".72c">(.*?)</font>'
            raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
            try:
                textlist.append(raw_text[0])
            except:
                continue
        subtitle_text = "\n".join(textlist)
        
            
    # option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    option_prompt = ""
    option = "\n".join([f"{opt}" for i, opt in enumerate(options)])

    # post_prompt = "\nAnswer with the option's letter from the given choices directly."
    post_prompt = ""
    reasoning_homer_suffix = f"Does the frame contain modern Christmax tree with apples, candles, and berries? Answer yes or no. If yes, count the number of apples, candles, and berries."
    reasoning_full_prompt = reasoning_homer_suffix + "\n" + option_prompt + "\n" + question + "\n" + option + "\n" + post_prompt  if not args.subtitle else subtitles_prompt + subtitle_text + "\n" + reasoning_homer_suffix + "\n" + option_prompt + "\n" + question + "\n" + option + "\n" + post_prompt
    full_prompt = option_prompt + "\n" + question + "\n" + option + "\n" + post_prompt if not args.subtitle else subtitles_prompt + subtitle_text + "\n" + option_prompt + "\n" + question + "\n" + option + "\n" + post_prompt
    return frames, full_prompt, reasoning_full_prompt, reasoning_homer_suffix, answer

def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames


def videomme_doc_to_text(doc):
    option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    post_prompt = "The best answer is:"
    full_prompt = option_prompt + "\n" + question + "\n" + post_prompt
    return full_prompt


def videomme_doc_to_text_subtitle(doc, cache_dir, lmms_eval_specific_kwargs=None):
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, "data", video_path)
    subtitle_path = os.path.join(cache_dir, "subtitle", doc["videoID"] + ".srt")
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(subtitle_path):  # Denote have subtitle
        subtitle = open(subtitle_path).readlines()
    else:
        subtitle = ""
    subtitles_prompt = "This video's subtitles are listed below: \n"
    if subtitle == "":
        subtitle = "No subtitles available"
    else:
        if "gemini_api_flag" in lmms_eval_specific_kwargs:  # specific for gemini_api
            if lmms_eval_specific_kwargs["gemini_api_flag"] == "full subtitle":
                textlist = []
                for ele in subtitle:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    matches = re.findall(pattern, ele)
                    if matches:
                        textlist.append(matches[0])
                subtitle_text = "\n".join(textlist)
        else:
            if "frame_num" in lmms_eval_specific_kwargs:
                frame_num = lmms_eval_specific_kwargs["frame_num"]
                subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
                if frame_num == -1:
                    frame_num = total_frame
                uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

                subtitle_by_frame_idx = []
                for frame_idx in uniform_sampled_frames:
                    for idx, title in enumerate(subtitle_by_frame):
                        if frame_idx < title[1] and frame_idx >= title[0]:
                            subtitle_by_frame_idx.append(idx)
                subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

                textlist = []
                for idx in subtitle_by_frame_idx:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
                    try:
                        textlist.append(raw_text[0])
                    except:
                        continue
                subtitle_text = "\n".join(textlist)
        subtitle = subtitle_text

    option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    full_prompt = subtitles_prompt + subtitle + "\n" + option_prompt + "\n" + question + "\n" + "The best answer is:"
    return full_prompt


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]




def videomme_process_results(doc, pred):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """ 
    pred_ans = extract_characters_regex(pred)
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    category = doc["domain"]
    sub_category = doc["sub_category"]
    task_category = doc["task_type"]
    data_dict = {"question_id": doc["question_id"], "duration": doc["duration"], "category": category, "sub_category": sub_category, "task_category": task_category, "pred_answer": pred_ans, "answer": doc["answer"]}

    # return {f"videomme_perception_score": data_dict for metric in matrices}
    return {f"videomme_perception_score": data_dict}


def videomme_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}

    for video_type in VIDEO_TYPE:
        for category in CATEGORIES:
            for sub_category in SUB_CATEGORIES:
                for task_category in TASK_CATEGORIES:
                    key = f"{video_type}_{category}_{sub_category}_{task_category}"
                    category2score[key] = {"correct": 0, "answered": 0}

    for result in results:
        video_type = result["duration"]
        category = result["category"]
        sub_category = result["sub_category"]
        task_category = result["task_category"]
        key = f"{video_type}_{category}_{sub_category}_{task_category}"
        category2score[key]["answered"] += 1
        category2score[key]["correct"] += result["pred_answer"] == result["answer"]

    for video_type in VIDEO_TYPE:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if video_type in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        # eval_logger.info(f"Evaluation on video Type: {video_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for category in CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        # eval_logger.info(f"Evaluation on Categories: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for sub_cate in SUB_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if sub_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        # eval_logger.info(f"Evaluation on Video Sub Categories: {sub_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for task_cate in TASK_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        # eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    # eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0



if __name__ == "__main__":
    import pandas as pd

    # Load a Parquet file into a DataFrame
    df = pd.read_parquet("/home/server08/yoonjeon_workspace/VideoPrefill/dataset/VideoMME/videomme/test-00000-of-00001.parquet")
    # Display the first few rows
    print(df.head())
    breakpoint()
    print(df.iloc[0].videoID)
    