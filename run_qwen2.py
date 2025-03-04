import os
import math
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info       

from utils.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from utils.merge import MergeManager
from utils.chunk import Chunk
from dataset_utils.physbench_utils import load_val_dataset, physbench_content

def load_model():
    #! Load qwen2.5 VL
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto", 
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def qwen2_prune_generate(model, processor, in_context_samples, query, max_token = 32):
    """
    token pruning by attention weights
    """
    if len(in_context_samples):
        content = [context[0] for context in in_context_samples] + query
    else:
        content = query
    messages = [
            {
                "role": "user",
                "content": content,
            },
        ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs)
    
    generated_ids = model.generate(**inputs.to("cuda"), max_new_tokens=max_token)
    generated_ids_trimmed = generated_ids[0, inputs.input_ids.shape[1] :]
    del inputs
    torch.cuda.empty_cache()
    output = processor.batch_decode(
        [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output
    

@torch.no_grad()
def qwen2_homer_generate(model, processor, in_context_samples, query, chunk_len, warmup_layer=8, max_token = 32):
    device = model.device
    manager_settings = {
            "max_chunk_len": chunk_len, # final chunk limit
            "max_initial_chunk_len": -1,
            "reduction_mode": "power_max_last_calibrated", # using the attention score between last query token and all the other tokens
            "layers_warmup": warmup_layer,
            "target_len": chunk_len, # max_position_ids = 128000
            "visualize": False,
        }
    merge_manager = MergeManager(**manager_settings) # manages layer-wise prune mask, target len, token-wise significance, etc.

    for idx, layer in enumerate(model.model.layers):
        layer.layer_idx = idx
        layer.merge_manager = merge_manager
        
    ######################################################################
    if len(in_context_samples):
        content = in_context_samples + [query]
        messages = []
        for sample in content:
            messages.append(
                [{
                "role": "user",
                "content": sample
                }]
            )
    else:
        content = query
        messages = [
                {
                    "role": "user",
                    "content": content,
                },
            ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs)
    
    # Derive a merging schedule
    input_ids = inputs['input_ids']
    pixel_values = inputs["pixel_values_videos"]
    grid_thw = inputs["video_grid_thw"]
    vision_start_token_id = model.config.vision_start_token_id
    vision_end_token_id = model.config.vision_end_token_id
    start_idxs = (input_ids == vision_start_token_id).nonzero()[:]
    end_idxs = (input_ids == vision_end_token_id).nonzero()[:]
    context_ids = input_ids[:, start_idxs[0][1] : end_idxs[-1][1] + 1]
    prefix_ids = input_ids[:, :start_idxs[0][1]]
    suffix_ids = input_ids[:, end_idxs[-1][1] + 1:]
    
    prefix_len = prefix_ids.shape[1]
    suffix_len = suffix_ids.shape[1]
    context_len = context_ids.shape[1]
    total_len = prefix_len + suffix_len + context_len
    
    # Compute "effective" chunk lengths (lengths without affix)
    eff_max_chunk_len = merge_manager.max_chunk_len - (prefix_len + suffix_len)
    num_merging_layers = len(model.model.layers) - merge_manager.layers_warmup
    if merge_manager.max_initial_chunk_len > 0:
        eff_max_initial_chunk_len = merge_manager.max_initial_chunk_len - (
            prefix_len + suffix_len
        )
        num_chunks = math.ceil(context_len / eff_max_initial_chunk_len)
    else:
        num_chunks = math.ceil(context_len / eff_max_chunk_len)

    tree_height = math.ceil(math.log2(num_chunks))

    num_chunks = (
        2 ** tree_height
    )
    layers_per_chunk = math.floor(num_merging_layers / (tree_height + 1)) # Each chunk goes through early warmup layers, then through layers_per_chunk layers, then leftover layers
    layers_leftover = num_merging_layers - layers_per_chunk * (tree_height + 1) 

    merge_manager.set_sample_info(
        prefix_len,
        suffix_len,
        eff_max_chunk_len,
        layers_per_chunk,
        layers_leftover,
    )
    ############# MODIFIED FROM ORIGINAL CODE #############
    pixel_values = pixel_values.type(model.visual.get_dtype())
    #! batchify
    image_embeds = model.visual(pixel_values.to(device), grid_thw = grid_thw.to(device)).to("cpu")
    torch.cuda.empty_cache()
    spatial_merge_size = model.config.vision_config.spatial_merge_size
    t, h, w = grid_thw[0][0], int(grid_thw[0][1] / spatial_merge_size), int(grid_thw[0][2] / spatial_merge_size)
    eff_chunk_len = math.ceil(context_len / num_chunks)
    #######################################################
    assert num_chunks > 1, "HOMER only supports context with more than 1 chunk"
    chunks = Chunk.make_chunks(
        prefix_ids,
        context_ids,
        suffix_ids,
        num_chunks=num_chunks,
        visualize=merge_manager.visualize,
        image_embeds=rearrange(image_embeds, "(t h w) c -> t h w c", t=t, h=h, w=w), # t x h x w, 3584
        get_rope_index=model.get_rope_index,
        video_grid_thw=grid_thw,
    ) # list of prefix + partial context + suffix

    final_target_len = min(merge_manager.target_len, total_len)

    # Recursively merge the chunks
    chunk = model._merge(
        chunks=chunks,
        height=tree_height,
        target_len=final_target_len,
    )

    # Set visualization info
    model.visualization_info = chunk.get_visualization_info()
    model.context_ids = context_ids[0].cpu()

    # assert self.config.pretraining_tp <= 1
    hidden_states = model.model.norm(chunk.hidden_states.to(device))
    logits = model.lm_head(hidden_states)
    logits = logits.float()

    # Reset merge_manager states
    merge_manager.set_layer_reduction_info()
    prefix_cache = chunk.cache
    suffix_len  = chunk.suffix_len
    last_position_id = chunk.position_ids[0, :, -1]
    logits = logits.to(device)
    torch.cuda.empty_cache()
    batch_size, _, prefix_len, _ = prefix_cache.key_cache[0].shape
    
    prefix_cache.crop(prefix_len - suffix_len)
    last_position_id -= suffix_len
    # Used in prepare_inputs_for_generation() to adjust position ids
    prefix_cache.pos_diff = (
        prefix_len - last_position_id.to(device)
    )
    # Get last token's output
    new_token_idx = logits[0, -1].argmax()
    new_input_ids = new_token_idx.unsqueeze(0).unsqueeze(0).to(device)
    input_ids = torch.cat(
        [
            torch.ones((batch_size, prefix_len - suffix_len), device=device).long(),
            new_input_ids,
        ],
        dim=-1,
    )
    ######################################################################################
    generated_ids = model.generate(
                input_ids=input_ids, 
                past_key_values=prefix_cache.to(device), 
                use_cache=True,
                max_new_tokens=max_token,
                output_attentions=True,
            )
    generated_ids_trimmed = generated_ids[0][input_ids.shape[1]:] 
    output = processor.batch_decode(
        [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output
    
@torch.no_grad()
def qwen2_original_generate(model, processor, in_context_samples, query, max_token = 32):
    """
    Generation text from message without token pruning
    """
    if len(in_context_samples):
        
        content = [context[0] for context in in_context_samples] + query
    else:
        content = query
    messages = [
            {
                "role": "user",
                "content": content,
            },
        ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs)
    
    generated_ids = model.generate(**inputs.to("cuda"), max_new_tokens=max_token)
    generated_ids_trimmed = generated_ids[0, inputs.input_ids.shape[1] :]
    del inputs
    torch.cuda.empty_cache()
    output = processor.batch_decode(
        [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output

if __name__=="__main__":
    from pathlib import Path
    import pickle
    from utils.video_utils import load_video
    from dataset_utils.physbench_utils import _process_visual_path, filter_topk
    
    model, processor = load_model()
    dataset_path = "/home/server08/hdd1/yoonjeon_workspace/ood_qa"
    
    if not (Path(dataset_path) / "qwen2_5").exists():
        """
        dataset_path 
            |- assets  image  qwen2_5  README.md  test.json  video
        """
        print("Process embeddings and measure similarity to run ICL")
        from dataset_utils.physbench_utils import _process_cls_embedding, _process_cossim 
        _process_cls_embedding(model, processor, dataset_path)
        _process_cossim(dataset_path)
        
    answer_path = '/home/server08/yoonjeon_workspace/VideoPrefill/logs/PhysBenchmark/val_answer.json'
    with open(answer_path, "r") as f:
        answer = json.load(f)
    
    dataset = load_val_dataset(dataset_path) #* evaluate with validation set (200 samples, gt answers are provided)
    idx = 0
    data = dataset[idx]
    end_prompt = "\nAnswer with the option's letter from the given choices directly. You can only answer one letter from A, B, C, or D."
    prompt = data["question"] + end_prompt
    visuals = [_process_visual_path(dataset_path, f) for f in data["file_name"]]
    images = []
    frames = None
    for visual in visuals:
        if visual.endswith("mp4"):
            frames, timesteps, total_frames = load_video(visuals[0], target_fps=None, max_num_frames=8) # physbench sets default frames to 8
        else:
            from PIL import Image
            img = Image.open(visual).convert("RGB")
            images.append(img)

    query = physbench_content(prompt, images, frames)
    
    #! If ICL is enabled, we will append the topk videos by similarity
    
    pkl_file_name = Path(dataset_path) / f"qwen2_5/{data['idx']}.pkl"
    with open(pkl_file_name, "rb") as f:
        curr_file = pickle.load(f)
    topk = 16
    in_context_samples = filter_topk(dataset_path, curr_file, dataset, topk)
    
    infer_type = "homer"
    max_token = 32
    if infer_type=="homer":
        chunk_len = 2048 # final token length limit
        warmup_layer = 8 # forward through first 8 layers before pruning
        output = qwen2_homer_generate(model, processor, in_context_samples, query, chunk_len, warmup_layer=warmup_layer, max_token = max_token) #! Needs debugging
    elif infer_type=="token_prune":
        prune_ratio = ["0.5"] * topk # per-sample pruning ratio
        output = qwen2_prune_generate(model, processor, in_context_samples, query, prune_ratio, max_token = max_token) #! Needs debugging
    elif infer_type=="original":
        output = qwen2_original_generate(model, processor, in_context_samples, query, max_token = max_token)
    print(output)