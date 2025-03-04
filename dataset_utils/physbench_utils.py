import os
import json
import torch
import pickle
from tqdm import tqdm
from torch.nn import functional as F
from collections import defaultdict
from qwen_vl_utils import process_vision_info
from utils.video_utils import load_video

task_type_order = ["dynamics", "relationships", "scene", "dynamics"]
ability_type_order = [
    "identify", "comparison", "static", "dynamic",
    "perception", "prediction", "judgment", "reasoning"
]

sub_task_order = [
    "number", "mass", "color", "attribute", "size", "location", "depth",
    "distance", "movement", "temperature", "camera", "gas", "light",
    "collision", "throwing", "manipulation", "fluid", "chemistry", "others"
]

def load_val_dataset(dataset_path, split="val"):
    with open(dataset_path +r'/test.json', 'r', encoding='utf-8') as file:
        dataset = json.load(file)
        
    val_dataset = []
    for item in dataset:
        if item['split'] == split:
            val_dataset.append(item)
    dataset = val_dataset
    return dataset

def load_visuals(dataset_path, file_name, test_frame=8):
    from utils.video_utils import load_video
    visuals = [_process_visual_path(dataset_path, f) for f in file_name]
    images = []
    for visual in visuals:
        if visual.endswith("mp4"):
            frames, timesteps, total_frames = load_video(visuals[0], target_fps=None, max_num_frames=test_frame)
            images.extend(frames)
        else:
            from PIL import Image
            img = Image.open(visual).convert("RGB")
            images.append(img)
    return images

def _process_cls_embedding(model, processor, dataset_path):
    from tqdm import trange
    
    # dataset_path = "/home/server08/hdd1/yoonjeon_workspace/ood_qa/"
    meta_dataset = load_val_dataset(dataset_path)
    
    # bos, system_text, eos, bos, text x M, eos, vision_start, video x N, vision_end, eos
    for idx in trange(len(meta_dataset)):
        item = meta_dataset[idx]
        # prompt = item["question"] + end_prompt
        prompt = item["question"].strip()
        query = prompt[:prompt.find("A. ")]
        options = prompt[prompt.find("A. "):].split("\n")

        images = load_visuals(dataset_path, item["file_name"], test_frame=8)
        content = [
            {
                "type": "text",
                "text": query
            },
            {
                "type": "video",
                "video": images
            }
        ]
        messages = [
            {
                "role": "user",
                "content": content,
            },
        ]
        max_token = 32
        
        # Preparation for inference
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs)
        outputs = model.generate(**inputs, max_new_tokens=max_token, return_dict_in_generate=True, output_hidden_states=True)
        hidden_states = outputs["hidden_states"]
        prefill_latens = hidden_states[0][-1]
        vs = (model.config.vision_start_token_id == inputs.input_ids).nonzero()
        ve = (model.config.vision_end_token_id == inputs.input_ids).nonzero()
        eos = (model.config.eos_token_id == inputs.input_ids).nonzero()
        text_states = prefill_latens[:, eos[0,1]+5:vs[0,1]]
        video_states = prefill_latens[:, vs[0,1]:ve[0,1]+1]
        output = processor.batch_decode(
                inputs.input_ids[:, eos[0,1]+5:vs[0,1]], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        # save hidden states
        save_at = os.path.join(dataset_path, "qwen2_5", f"{item['idx']}.pkl")
        os.makedirs(os.path.dirname(save_at), exist_ok=True)
        hidden = {
            "text": text_states.cpu().detach(),
            "image": video_states.cpu().detach(),
            
        }
        with open(save_at, "wb") as f:
            pickle.dump(hidden, f)
            
def _process_cossim(dataset_path):
    dataset = load_val_dataset(dataset_path)
    saved_at = [os.path.join(dataset_path, "qwen2_5", f"{item['idx']}.pkl") for item in dataset]
    pkl_files = [pickle.load(open(f, "rb")) for f in saved_at]
    
    for file_idx, (curr_filename, file) in tqdm(enumerate(zip(saved_at, pkl_files))):
        text = file["text"].cuda()
        text = F.normalize(text, p=2, dim=-1)
        video = file["image"].cuda()
        video = F.normalize(video, p=2, dim=-1)
        other_files = pkl_files[:file_idx] + pkl_files[file_idx+1:]
        other_filenames = saved_at[:file_idx] + saved_at[file_idx+1:]
        sim = {}
        for other_file, file_name in zip(other_files, other_filenames):
            other_text = other_file["text"]
            other_video = other_file["image"]
            other_text = F.normalize(other_text, p=2, dim=-1)
            other_video = F.normalize(other_video, p=2, dim=-1)
            text_sim = text[0, -1] @ other_text[0, -1].cuda().T
            video_sim = video[0, -1] @ other_video[0, -1].cuda().T
            sim[file_name] = {
                'text_sim': text_sim.mean().item(),
                'video_sim': video_sim.mean().item()
            }
        file["similiarty"] = sim
        with open(curr_filename, "wb") as f:
            pickle.dump(file, f)

def filter_topk(dataset_path, curr_file, dataset, topk):
    """
    For physbench, select topk videos based on the similarity between the hidden representation of [EOV] token
    topk: int, number of topk videos to select
    """
    from PIL import Image
    end_prompt = "\nAnswer with the option's letter from the given choices directly. You can only answer one letter from A, B, C, or D."
    vid_sims = [(k, x["video_sim"]) for k, x in curr_file['similiarty'].items()]
    vid_sims.sort(key=lambda x: x[1], reverse=True)
        
    topk_idxs = [int(pkl_path[0].split("/")[-1].strip(".pkl")) for pkl_path in vid_sims[:topk]]

    image_files = [(datum['idx'], datum['file_name']) for datum in dataset if datum['idx'] in topk_idxs]
    with open("logs/PhysBenchmark/val_answer.json", "r", encoding="utf-8") as f:
        answer_file = json.load(f)
    contents = []
    for idx, file_names in image_files:
        item = [data for data in dataset if data["idx"] == idx][0]
        data_idx = item["idx"]
        prompt = item["question"] + end_prompt
        answers = [x for x in answer_file if x['idx'] == data_idx][0]
        answer = answers["answer"]
        
        visuals = [_process_visual_path(dataset_path, f) for f in item["file_name"]]
        images = []
        frames = None
        for visual_path in visuals:
            
            if visual_path.endswith("mp4"):
                frames, timesteps, total_frames = load_video(visual_path, target_fps=None, max_num_frames=8)
            else:
                img = Image.open(visual_path).convert("RGB")
                images.append(img)
        content = physbench_content(prompt, images, frames, answer) 
        contents.append(content)
    return contents


def physbench_content(prompt, images, frames, answer=None, img_hw=(280, 280)):
    """
    Process a single question with images and videos (optionally with answer pair)
    """
    end_idx = 0
    substr = "<image>"
    img_idx = 0
    content = []
    video_idx = prompt.find("<video>")
    if video_idx > 0 and frames is not None: # if <video> is present in the question
        content.append(
            {
                "type": "text",
                "text": prompt[:video_idx]
            }
        ) # question in text
        content.append(
            {
                "type": "video",
                "video": frames,
                "resized_height": img_hw[0], 
                "resized_width": img_hw[1]
            }
        ) # question in video
        end_idx = video_idx + len("<video>")
    while True: # find all images in the prompt e.g.) A. <image> B. <image> C. <image> D. <image>
        start_idx = prompt.find(substr, end_idx)
        if start_idx == -1:
            break
        else:
            if start_idx > end_idx:
                content.append(
                    {
                        'type': 'text',
                        'text': prompt[end_idx:start_idx]
                    }
                    )
            end_idx = start_idx + len(substr)
            content.append(
                    {
                        "type": "image",
                        "image": images[img_idx],
                        "resized_height": img_hw[0], 
                        "resized_width": img_hw[1]
                    }
                )
            img_idx += 1
    # append the remaining text (select option blabla)
    content.append(
        {
            "type": "text",
            "text": prompt[end_idx:]
        }
    )
    if answer is not None: # If answer is provided, this means in-context samples are provided. Append answers to the question.
        d = {"A": 0, "B": 1, "C": 2, "D": 3}
        map_ans_to_idx = lambda k: d[k]
        opt_idxs = [prompt.find(opt) for opt in ["A.", "B.", "C.", "D."]] + [-1]
        options = [prompt[idx:opt_idxs[i+1]].split("\n")[0] for i, idx in enumerate(opt_idxs[:-1])]
        assert options[0].startswith("A.")
        assert options[1].startswith("B.")
        assert options[2].startswith("C.")
        assert options[3].startswith("D.")
        selected_option = options[map_ans_to_idx(answer)]
        
        if "<image>" in selected_option: # A. <image> B. <image> C. <image> D. <image> -> The answer in option C, which is the following image.
            ans_img = images[map_ans_to_idx(answer)]
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"The answer is option {answer}, which is the following image\n"
                    },
                    {
                        "type": "image",
                        "image": ans_img,
                        "resized_height": img_hw[0], 
                        "resized_width": img_hw[1]
                    }
                ]
            )
        else: # A. text B. text C. text D. text -> The answer is C. text
            content.append(
                {
                    "type": "text",
                    "text": f"The answer is {selected_option}"
                }
            )
    return content

def _process_visual_path(dataset_path, file_name):
    if file_name.endswith(".mp4"):
        file_path = dataset_path + '/video/' + file_name
    elif file_name.endswith(".jpg") or file_name.endswith(".JPG") or file_name.endswith(".png"):
        file_path = dataset_path + '/image/' + file_name
    else:
        raise NotImplementedError
    return file_path        

def answer_true(item, gt_item):
    return item['answer'][0] == gt_item['answer'] or item['answer'][0].startswith(gt_item['answer'])

def calculate_accuracy(val_annotation_file, user_submission_file):
    with open(user_submission_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    with open(val_annotation_file, 'r', encoding='utf-8') as file:
        gt_data = json.load(file)
    total_questions = len(data)
    correct_answers = 0

    task_type_counts = defaultdict(lambda: {'correct': -1, 'total': -1})
    sub_type_counts = defaultdict(lambda: {'correct': -1, 'total': -1})
    ability_type_counts = defaultdict(lambda: {'correct': -1, 'total': -1})

    for item in data:
        if item['answer'] is None:
            continue

        gt_item = next((gt for gt in gt_data if gt['idx'] == item['idx']), None)

        if gt_item is None:
            print(f'Unknown idx : {item["idx"]}')
            continue

        if answer_true(item, gt_item):
            correct_answers += 1
            task_type_counts[gt_item['task_type']]['correct'] = max(task_type_counts[gt_item['task_type']]['correct'], 0) + 1
            sub_type_counts[gt_item['sub_type']]['correct'] = max(sub_type_counts[gt_item['sub_type']]['correct'], 0) + 1
            ability_type_counts[gt_item['ability_type']]['correct'] = max(ability_type_counts[gt_item['ability_type']]['correct'], 0) + 1

        task_type_counts[gt_item['task_type']]['total'] = max(task_type_counts[gt_item['task_type']]['total'], 0) + 1
        sub_type_counts[gt_item['sub_type']]['total'] = max(sub_type_counts[gt_item['sub_type']]['total'], 0) + 1
        ability_type_counts[gt_item['ability_type']]['total'] = max(ability_type_counts[gt_item['ability_type']]['total'], 0) + 1

    overall_accuracy = correct_answers / total_questions * 100 if total_questions > 0 else 0

    def calculate_specific_accuracy(counts):
        return {
            k: {'accuracy': max(v['correct'] / v['total'] * 100, 0) if v['total'] > 0 else -1, 'correct': v['correct'],
                'total': v['total']} for k, v in counts.items()}

    task_type_accuracy = calculate_specific_accuracy(task_type_counts)
    sub_type_accuracy = calculate_specific_accuracy(sub_type_counts)
    ability_type_accuracy = calculate_specific_accuracy(ability_type_counts)

    return {
        'overall_accuracy': overall_accuracy,
        'overall_correct': correct_answers,
        'overall_total': total_questions,
        'task_type_accuracy': task_type_accuracy,
        'sub_type_accuracy': sub_type_accuracy,
        'ability_type_accuracy': ability_type_accuracy
    }

def calculate_weighted_avg(accuracy_data):
    total_correct = sum(data['correct'] for data in accuracy_data.values() if data['total'] > 0)
    total_questions = sum(data['total'] for data in accuracy_data.values() if data['total'] > 0)
    if total_questions > 0:
        return total_correct / total_questions * 100
    return -1

def print_accuracies(accuracies, name='Unknown'):
    print(f"Overall Accuracy: {accuracies['overall_accuracy']:.2f}% ({accuracies['overall_correct']} correct out of {accuracies['overall_total']})")

    print("\nTask Type Accuracies:")
    for task_type, data in accuracies['task_type_accuracy'].items():
        print(f"{task_type}: {data['accuracy']:.2f}% ({data['correct']} correct out of {data['total']})")

    print("\nSub Type Accuracies:")
    for sub_type in sub_task_order:
        data = accuracies['sub_type_accuracy'].get(sub_type, {'accuracy': -1, 'correct': -1, 'total': -1})
        print(f"  {sub_type}: {data['accuracy']:.2f}% ({data['correct']} correct out of {data['total']})")

    print("\nAbility Type Accuracies:")
    for ability_type, data in accuracies['ability_type_accuracy'].items():
        print(f"  {ability_type}: {data['accuracy']:.2f}% ({data['correct']} correct out of {data['total']})")

    # Generate markdown tables
    def generate_markdown_table(title, accuracy_data, order=None):
        if order:
            accuracy_data = {k: accuracy_data.get(k, {'accuracy': -1, 'correct': -1, 'total': -1}) for k in order}
        headers = '| ' + ' | '.join(accuracy_data.keys()) + ' | avg |'
        separator = '| ' + ' | '.join(['---'] * len(accuracy_data)) + ' | --- |'
        values = '| ' + ' | '.join([f"{data['accuracy']:.2f}" for data in accuracy_data.values()]) + f' | {calculate_weighted_avg(accuracy_data):.2f} |'

        headers = '| model '+ headers
        separator = '| --- ' + separator
        values = f'| {name} ' + values
        return f"**{title}**\n\n{headers}\n{separator}\n{values}\n"

    print("\nMarkdown Tables:\n")
    print(generate_markdown_table("Task Type Accuracy", accuracies['task_type_accuracy'], task_type_order))
    print(generate_markdown_table("Sub Type Accuracy", accuracies['sub_type_accuracy'], sub_task_order))
    print(generate_markdown_table("Ability Type Accuracy", accuracies['ability_type_accuracy'], ability_type_order))

def measure_acc():
    gt_file_path = os.path.join(log_path, 'val_answer.json')
    user_file_path = os.path.join(log_path, 'qwen2-5_vl_8_icl100_val.json')
    accuracies = calculate_accuracy(gt_file_path, user_file_path)
    print_accuracies(accuracies)
