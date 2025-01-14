import os
from transformers import AutoTokenizer
from minference import MInference
from functools import partial
from minference.minference_configuration import MInferenceConfig
from minference.patch import prepare_cache, prepare_inputs_for_generation_kvcompression
from minference.modules.forward import attn_forward, decoding_forwards, prefill_forwards
from longva.model.language_model.llava_qwen import LlavaQwenForCausalLM
from longva.mm_utils import tokenizer_image_token
from longva.constants import IMAGE_TOKEN_INDEX
from longva.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from decord import VideoReader, cpu
import torch
import numpy as np


def load_model(model_path, attn_type = None, device_map = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaQwenForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, 
        attn_implementation="flash_attention_2", 
        device_map = device_map, 
        torch_dtype = "auto",
        )
    if attn_type == "minference":
        BASE_DIR = "/home/server44/anaconda3/envs/longva/lib/python3.10/site-packages/minference-0.1.6b1-py3.10-linux-x86_64.egg/minference/configs/"
        config_path = os.path.join(
            BASE_DIR, "Qwen2_7B_Instruct_128k_instruct_kv_out_v32_fit_o_best_pattern.json"
        )
        minference_patch = MInference("minference", model_path, config_path=config_path)
        model = minference_patch(model)
    elif attn_type == "flexprefill":
        minference_patch = MInference("flexprefill", model_path, attn_kwargs={"gamma": 0.9, "tau": 0.1, "min_budget": None, "max_budget": None, "block_size": 64})
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
    



def model_patch(model, config=None):
    
    model.config.starting_layer = config.starting_layer
    model.config.config_path = config.config_path

    Attention = model.model.layers[0].self_attn.__class__ # Qwen2FlashAttention2
    Model = model.model.__class__ # LlavaQwenModel 
    DecoderLayer = model.model.layers[0].__class__ # Qwen2DecoderLayer

    prefill_forward = prefill_forwards[config.attn_type]
    decoding_forward = decoding_forwards[config.kv_type]

    custom_rope_func = None  # apply custom rope func if needed
    forward = partial(
        attn_forward,
        prefill_forward=prefill_forward,
        decoding_forward=decoding_forward,
        attn_forward_config=config.attn_kwargs,
        customized_rope_func=custom_rope_func,
    )

    def update_module(m):
        if isinstance(m, Attention):
            m.forward = (
                lambda self, *args, **kwargs: forward(self, *args, **kwargs)
            ).__get__(m, Attention)

    model.apply(update_module)
    prepare_cache_func = prepare_cache(config.kv_type, config)
    model._prepare_cache_for_generation = prepare_cache_func.__get__(
        model, model.__class__
    )

    prepare_inputs_func = prepare_inputs_for_generation_kvcompression(
        config.kv_type, config, model.prepare_inputs_for_generation
    )
    model.prepare_inputs_for_generation = prepare_inputs_func.__get__(
        model, model.__class__
    )
    return model

#! Process video
def video_embed(video_path, image_processor, max_frames_num):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()

    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    return video_tensor

if __name__ == "__main__":
    # fix seed
    torch.manual_seed(0)

    #! Load Model, Tokenizer, Video processor
    model_path = "lmms-lab/LongVA-7B-DPO" # you can also set the device map to auto to accomodate more frames
    video_path = "/home/server08/yoonjeon_workspace/VideoPrefill/LongVA/dataset/VideoMME/videos_chunked_20/data/zbvamKv81o0.mp4"
    
    model, tokenizer, image_processor = load_model(model_path, device_map= "auto")

    video_tensor = video_embed(video_path, image_processor, max_frames_num=50).to(model.device, dtype=model.dtype) # 57200
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nGive a detailed caption of the video as if I am blind.<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(video_tensor.device)
    
    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=[video_tensor], modalities=["video"], do_sample=True, temperature=0.5, top_p=None, num_beams=1, use_cache=True, max_new_tokens=1024)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print("*"*50)
    print(outputs)