import os
import math
import json
import torch
from pathlib import Path
from einops import rearrange
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info       


import pickle
from utils.video_utils import load_video
from dataset_utils.physbench_utils import _process_visual_path, filter_topk
# from utils.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from utils.merge import MergeManager
from utils.chunk import Chunk
from dataset_utils.physbench_utils import load_val_dataset, physbench_content

def load_model():
    #! Load qwen2.5 VL
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left") # for batched inference, set padding onto left side
    return model, processor

def prepare_batched_icl(data, dataset, dataset_path, topk, processor):
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

    in_context_samples = filter_topk(dataset_path, curr_file, dataset, topk)
    
    if len(in_context_samples):
        content = in_context_samples + [query]
        messages = []
        for sample in content:
            messages.append(
                [
                    {
                    "role": "user",
                    "content": sample
                    }
                ]
            )
        assert isinstance(messages[0], list), ""
    else:
        content = query
        messages = [
                {
                    "role": "user",
                    "content": content,
                },
            ]
    
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    images, videos = process_vision_info(messages)
    return texts, images, videos

def qwen2_prune_generate(model, processor, texts, images, videos, ratios, max_token): #! Needs debugging
    """
    token pruning by attention weights
    """
    inputs = processor(text=texts, images=images, videos=videos, padding=True, return_tensors="pt")
    #! for each in-context sample, prune tokens with dynamic ratio
    
    generated_ids = model.generate(**inputs.to("cuda"), max_new_tokens=max_token) # attention_mask: right padding -> wrong
    generated_ids_trimmed = generated_ids[0, inputs.input_ids.shape[1] :]
    del inputs
    torch.cuda.empty_cache()
    output = processor.batch_decode(
        [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output


@torch.no_grad()
def qwen2_homer_generate(model, processor, texts, images, videos, chunk_len, warmup_layer=8, max_token = 32):
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
    inputs = processor(text=texts, images=images, videos=videos, padding=True, return_tensors="pt")
    
    num_merging_layers = len(model.model.layers) - merge_manager.layers_warmup
    num_chunks = len(inputs.input_ids) - 1
    n_vision_tokens = (inputs.input_ids == model.config.image_token_id).sum().item() + (inputs.input_ids == model.config.video_token_id).sum().item()
    n_text_tokens = inputs.input_ids.size(1) - n_vision_tokens
    n_to_reduce = chunk_len - n_text_tokens
    tree_height = math.ceil(math.log2(num_chunks))

    num_chunks = (
        2 ** tree_height
    )
    layers_per_chunk = math.floor(num_merging_layers / (tree_height + 1)) # Each chunk goes through early warmup layers, then through layers_per_chunk layers, then leftover layers
    layers_leftover = num_merging_layers - layers_per_chunk * (tree_height + 1) 

    merge_manager.set_sample_info(
        prefix_len,
        suffix_len,
        n_to_reduce,
        layers_per_chunk,
        layers_leftover,
    )
    assert num_chunks > 1, "HOMER only supports context with more than 1 chunk"
    chunks = Chunk.make_chunks(
        num_chunks=num_chunks,
        visualize=merge_manager.visualize,
        get_rope_index=model.get_rope_index,
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
def qwen2_original_generate(model, processor, texts, images, videos, max_token): #! Needs debugging
    """
    token pruning by attention weights
    """
    inputs = processor(text=texts, images=images, videos=videos, padding=True, return_tensors="pt")
    
    generated_ids = model.generate(**inputs.to("cuda"), max_new_tokens=max_token)
    generated_ids_trimmed = generated_ids[0, inputs.input_ids.shape[1] :]
    del inputs
    torch.cuda.empty_cache()
    output = processor.batch_decode(
        [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output

if __name__=="__main__":
    
    model, processor = load_model()
    dataset_path = "/home/server08/hdd1/yoonjeon_workspace/ood_qa/physbench"
    
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
    topk = 16
    infer_type = "homer"
    max_token = 32
        
    for data in dataset:
        texts, images, videos = prepare_batched_icl(data, dataset, dataset_path, topk, processor)
        
        if infer_type=="homer":
            chunk_len = 2048 # final token length limit
            warmup_layer = 8 # forward through first 8 layers before pruning
            output = qwen2_homer_generate(model, processor, texts, images, videos, chunk_len = chunk_len, warmup_layer = warmup_layer, max_token = max_token) #! Needs debugging
        elif infer_type=="token_prune":
            prune_ratio = ["0.5"] * topk # per-sample pruning ratio
            output = qwen2_prune_generate(model, processor, texts, images, videos, prune_ratio, max_token = max_token) #! Needs debugging
        elif infer_type=="original":
            output = qwen2_original_generate(model, processor, texts, images, videos, max_token = max_token)
        print(output)