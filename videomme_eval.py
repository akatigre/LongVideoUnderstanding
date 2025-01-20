
import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from decord import VideoReader, cpu

from videomme_utils import videomme_doc_to_text, videomme_process_results, write_or_append_json
from tokenizers import AutoTokenizer
from minference import MInference
from longva.model.language_model.llava_qwen import LlavaQwenForCausalLM
from longva.mm_utils import tokenizer_image_token
from longva.constants import IMAGE_TOKEN_INDEX
from longva.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


#! Process video
def video_embed(video_path, image_processor, max_frames_num):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()

    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    return video_tensor


def load_model(model_path, attn_type = None, device_map = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaQwenForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, 
        attn_implementation="flash_attention_2", 
        device_map = device_map, 
        torch_dtype = "auto",
        )
    if attn_type == "minference":
        BASE_DIR = "../MInference/minference-0.1.6b1-py3.10-linux-x86_64.egg/minference/configs/"
        config_path = os.path.join(
            BASE_DIR, "Qwen2_7B_Instruct_128k_instruct_kv_out_v32_fit_o_best_pattern.json"
        )
        minference_patch = MInference("minference", model_path, config_path=config_path)
        model = minference_patch(model)
    elif attn_type == "flexprefill":
        minference_patch = MInference("flexprefill", model_path, attn_kwargs={"gamma": 0.9, "tau": 0.1, "min_budget": None, "max_budget": None, "block_size": 128})
        model = minference_patch(model)
    
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != "auto":
        vision_tower.to(device=device_map, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    return model, tokenizer, image_processor

def evaluate_multiple(model, tokenizer, image_processor, metadata, max_frames_num, json_name, undone_ids, data_path):
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
        
    for meta in tqdm(metadata):
        video_path = meta["videoID"] + ".mp4"
        video_path = os.path.join(args.data_path, "data", video_path)
        if os.path.exists(video_path):
            video_path = video_path
        elif os.path.exists(video_path.replace("mp4", "MP4")):
            video_path = video_path.replace("mp4", "MP4")
        elif os.path.exists(video_path.replace("mp4", "mkv")):
            video_path = video_path.replace("mp4", "mkv")
        else:
            sys.exit(f"video path:{video_path} does not exist, please check")
        video_tensor = video_embed(
            str(video_path), image_processor, max_frames_num = max_frames_num, 
            ).to(model.device, dtype=model.dtype) # 57200
        text_info = videomme_doc_to_text(meta, system_prompt=system_prompt)
        input_ids = tokenizer_image_token(text_info, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(video_tensor.device)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, images=[video_tensor], modalities=["video"], 
                do_sample=True, temperature=0.5, top_p=None, 
                num_beams=1, use_cache=True, max_new_tokens=1024
                )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip() # could be empty string
        score_dict = videomme_process_results(meta, pred=outputs)
        write_or_append_json(json_name, score_dict)
        
def main(args):
    metadata_path = Path(args.data_path) / "lvb_val.json"
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    json_name = f"answers_{args.attn_type}.json"
    max_frames_num = 100
    if os.path.exists(json_name):
        print(f"{json_name} already exists. Skipping evaluation.")
        with open(json_name, "r") as f:
            samples = json.load(f)
        done_ids = [sample["id"] for sample in samples]
        metadata_ids = [sample["id"] for sample in metadata]
        undone_ids = [id for id in metadata_ids if id not in done_ids]
    else:   
        undone_ids = [sample["id"] for sample in metadata]
        
    if len(undone_ids) > 0:
        model_path = "lmms-lab/LongVA-7B-DPO" # you can also set the device map to auto to accomodate more frames
        model, tokenizer, image_processor = load_model(model_path, device_map= "cuda")
        evaluate_multiple(model, tokenizer, image_processor, metadata, max_frames_num, json_name, undone_ids, data_path=args.data_path)
        print("Evaluation done. Samples are saved in samples.json")  
           
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--attn-type", type=str, default="vanilla", choices=["vanilla", "minference", "flexprefill", "inf-llm"])
    parser.add_argument("--data-path", type=str, default="./dataset/LongVideoBench")
    parser.add_argument("--save-dir", type=str, default="../lvb_answers")
    args = parser.parse_args()
    
    main(args)