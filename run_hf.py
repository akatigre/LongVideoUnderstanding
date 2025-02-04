# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import warnings
warnings.filterwarnings("ignore")

import os
import torch
import numpy as np
from minference import MInference
from decord import VideoReader, cpu
from transformers import AutoTokenizer
from longva.model.language_model.llava_qwen import LlavaQwenForCausalLM
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

def prepare_models(args, load_fn):
    model_path_map = {
        "longVA": "lmms-lab/LongVA-7B-DPO", 
        "qwen2-vl": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen2-5-vl": "Qwen/Qwen2.5-VL-7B-Instruct",
        "longvila-1.5b": "Efficient-Large-Model/qwen2-1.5b-longvila-256f",
        "longvila-7b": "Efficient-Large-Model/qwen2-7b-longvila-256f",
    }
    model_name = model_path_map[args.model_name]
    model, tokenizer, image_processor = load_fn(model_name, args)
    
    if args.homer:
        max_position_id = model.config.max_position_embeddings // args.scale
        homer_args = {
                "max_chunk_len": 4096, # Qwen2.5VL: 128000
                "max_initial_chunk_len": args.max_initial_chunk_len,
                "reduction_mode": "power_max_last_calibrated",
                "layers_warmup": args.layer_warmup,
                "target_len": max_position_id // 2,
                "bias_path": args.bias_path,
            }

        from homer.merge import MergeManager
        manager_settings = {
            "max_chunk_len": 2048,
            "max_initial_chunk_len": -1,
            "reduction_mode": "power_max_last_calibrated",
            "layers_warmup": 12,
            "target_len": 2048,
            "bias_path": None,
            "visualize": False,
        }

        for k in homer_args:
            assert k in manager_settings, f"Unknown argument: {k}"

        manager_settings.update(homer_args)

        model.merge_manager = MergeManager(**manager_settings)

        for idx, layer in enumerate(model.model.layers):
            layer.layer_idx = idx
            layer.merge_manager = model.merge_manager

    if args.attn_type == "minference":
        BASE_DIR = "./MInference/minference/configs/"
        config_path = os.path.join(
            BASE_DIR, "Qwen2_7B_Instruct_128k_instruct_kv_out_v32_fit_o_best_pattern.json"
        )
        minference_patch = MInference("minference", model_name, config_path=config_path, kv_cache_cpu=True,)
        model = minference_patch(model)
      
    elif args.attn_type == "flexprefill":
        minference_patch = MInference(
            "flexprefill", 
            model_name, 
            attn_kwargs = {
                "gamma": 0.9, 
                "tau": 0.1, 
                "min_budget": None,
                "max_budget": None,
                "block_size": 32
                }
            )
        model = minference_patch(model)

    
    return model, tokenizer, image_processor

def load_qwen2_model(model_path, device_map):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        attn_implementation="flash_attention_2",
        torch_dtype="auto", 
        device_map = device_map,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, tokenizer, processor

def load_qwen2_5_model(model_path, args):
    from homer.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    if args.homer:
        max_position_id = args.max_position_embeddings * args.scale
        homer_args = {
                "max_chunk_len": 4096,
                "max_initial_chunk_len": args.max_initial_chunk_len,
                "reduction_mode": "power_max_last_calibrated",
                "layers_warmup": args.layer_warmup,
                "target_len": max_position_id // 2,
                "bias_path": args.bias_path,
            }
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto", 
        device_map="auto",
        attn_implementation="flash_attention_2",
        homer_args=homer_args,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, None, processor

def qwen2_prepare_embeds(video_path, text, vision_processor: Qwen2VLProcessor, max_frames_num, **kwargs):
    from qwen_vl_utils import process_vision_info
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "nframes": max_frames_num,
                },
                {"type": "text", "text": text},
                ],
            }
        ]
    
    images, videos = process_vision_info(messages)
    text = vision_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = vision_processor(text=[text], videos=videos, images = images, padding=True, return_tensors="pt")
    return inputs

def qwen2_5_prepare_embeds(video_path, text, vision_processor: Qwen2VLProcessor, fps, **kwargs):
    from qwen_vl_utils import process_vision_info  
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "fps": fps,
                },
                {
                    "type": "text", 
                    "text": text
                },
            ],
        }
    ]
        
    # Preparation for inference
    text = vision_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = vision_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    return inputs

def qwen2_generate_answer(model: Qwen2VLForConditionalGeneration, input_ids, attention_mask, pixel_values_videos, video_grid_thw, tokenizer, **kwargs):
    kwargs.update(
    {
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 128,
            "use_cache": False
        }
    )
    output = model.generate(
        input_ids = input_ids, 
        attention_mask = attention_mask, 
        pixel_values_videos = pixel_values_videos, 
        video_grid_thw = video_grid_thw, 
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.eos_token_id,
        do_sample=True if kwargs["temperature"] > 0 else False,
        **kwargs,
    )
    return output
    
def qwen2_5_generate_answer(model, inputs, processor, args):
    if args.homer:
        input_ids = inputs.input_ids
        video_token_ids = model.config.video_token_id
        mask = (input_ids == video_token_ids)
        video_start_idx = mask.nonzero()[0][1].item()
        video_end_idx = mask.nonzero()[-1][1].item()
        homer_prefix = model.create_homer_prefix(
            prefix_ids=inputs.input_ids[:, :video_start_idx],
            context_ids=inputs.input_ids[:, video_start_idx:video_end_idx+1],
            suffix_ids=inputs.input_ids[:, video_end_idx+1:],
            pixel_values_videos=inputs["pixel_values_videos"],
            video_grid_thw=inputs["video_grid_thw"],
            attention_mask = inputs["attention_mask"],
        )
        generated_ids, seq_len = model.generate(homer_prefix, max_new_tokens=128) # homer_prefix should be the same format with inputs = processor()
        generated_ids_trimmed = generated_ids[0][seq_len-1 :]
        output_text = processor.batch_decode(
            [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    else:
        
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    return output_text

def load_longva_model(model_path, device_map, dtype=torch.float16):
    from longva.model.language_model.llava_qwen import LlavaQwenForCausalLM
    from longva.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaQwenForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map = device_map,
        torch_dtype = dtype,
        )
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


def longva_prepare_embeds(video_path, text, tokenizer, vision_processor, model: LlavaQwenForCausalLM, max_frames_num, device, dtype):
    from longva.mm_utils import tokenizer_image_token
    from longva.constants import IMAGE_TOKEN_INDEX
    frames = video_extract_frames(video_path, max_frames_num)
    
    video_embeds = vision_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(device, dtype=dtype)
    input_ids = tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    position_ids = None
    attention_mask = None
    images = [video_embeds]
    modalities = ["video"]
    image_sizes = None
    if images is not None:
        (inputs, position_ids, attention_mask, _, inputs_embeds, inputs_ids) = model.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes
            )
    else:
        inputs_embeds = model.get_model().embed_tokens(input_ids)
    return {
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "inputs_embeds": inputs_embeds,
        "inputs_ids": inputs_ids
        }

def longva_generate_answer(position_ids, attention_mask, inputs_embeds, model_generate, model, **kwargs):

    with torch.inference_mode():
        output_ids = model_generate(
            model,
            position_ids=position_ids, 
            attention_mask=attention_mask, 
            inputs_embeds=inputs_embeds,
            **kwargs
            )
    return output_ids

def load_longvila_model(model_path, device_map):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path, 
        model_name, 
        device_map=device_map, 
        attn_implementation="flash_attention_2",
        )
    model.image_processor = image_processor
    tokenizer.pad_token_id = 151643
    return model, tokenizer, image_processor
    
def longvila_prepare_embeds(video, text, tokenizer, vision_processor, model, max_frames_num):
    from PIL import Image
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        process_images,
        tokenizer_image_token,
    )
    
    frames = video_extract_frames(video, max_frames_num)
    video = [Image.fromarray(img) for img in frames]
    video_embeds = [process_images(video, vision_processor, model.config).half().cuda()]
    
    qs = f"<video>\n {text}"
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(frames) + qs

    conv_template = "hermes-2"
    conv = conv_templates[conv_template].copy()

    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    return {"input_ids": input_ids, "images": video_embeds, "stopping_criteria": [stopping_criteria]}


def longvila_generate_answer(model, tokenizer, inputs):
    pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    attention_masks = inputs["input_ids"].ne(pad_token_ids).long().cuda()
    gen_kwargs = {
        'max_new_tokens': 1024,
        'temperature': 0.2,
        'top_p': None,
        'num_beams': 1,
        'use_cache': False,
        'do_sample': True if gen_kwargs["temperature"] > 0 else False,
    }

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            attention_mask=attention_masks,
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
        )
    return output_ids


def video_extract_frames(video_path, max_frames_num):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames

def model_system_prompt(model_name):
    longva_prompt = lambda x: f"""
        <|im_start|>
        system\n
        You are a helpful assistant.
        <|im_end|>\n
        <|im_start|>user\n
        <image>\n 
        {x} <|im_end|>\n
        <|im_start|>assistant\n
        """
    
    qwen_prompt = lambda x: x
        
    longvila_prompt = lambda x: f"USER: <video>\n{x} ASSISTANT:"
    prompt_map = {
        "longVA": longva_prompt,
        "qwen2-vl": qwen_prompt,
        "qwen2-5-vl": qwen_prompt,
        "longvila-1.5b": longvila_prompt,
        "longvila-7b": longvila_prompt,
    }
    return prompt_map[model_name]
