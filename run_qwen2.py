import os
import math
import json
import torch
from pathlib import Path
from einops import rearrange
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.cache_utils import DynamicCache
import pickle
from utils.video_utils import load_video
from dataset_utils.physbench_utils import _process_visual_path, filter_topk
from modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from homer_utils.merge import MergeManager
from homer_utils.chunk import Chunk
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

    in_context_samples = filter_topk(dataset_path, curr_file, dataset, topk) # batchify is only supported with "image" type. do not use "video" type
    
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
        assert isinstance(messages[0], list), f"messages should be nested list but got {type(messages[0])}"
    else:
        content = query
        messages = [
                {
                    "role": "user",
                    "content": content,
                },
            ]
    
    # texts = [
    #     processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    #     for msg in messages
    # ]
    texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    return texts, images, videos

@torch.no_grad()
def qwen2_prune_generate(
    model: Qwen2_5_VLForConditionalGeneration, 
    processor, 
    texts, 
    images, 
    videos, 
    prune_ratios, 
    max_token, 
    ): #! Needs debugging
    """
    token pruning by attention weights
    """
    device = model.device
    inputs = processor(text=texts, images=images, videos=videos, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    attention_mask = inputs["attention_mask"]
    ############# MODIFIED FROM ORIGINAL CODE #############
    output_attentions, output_hidden_states, return_dict = False, False, False
    inputs_embeds, pixel_values_videos, video_grid_thw, position_ids, cache_position, second_per_grid_ts, rope_deltas = None, None, None, None, None, None, None
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = model.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(model.visual.dtype)
            image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == model.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
            video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == model.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (cache_position is not None and cache_position[0] == 0) or model.rope_deltas is None:
            position_ids, rope_deltas = model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            model.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + model.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    ######################################################################
    pad_masks = (input_ids == model.config.bos_token_id)
    merged_sample = model._forward_and_prune(
        inputs_embeds = inputs_embeds,
        position_ids = position_ids,
        ratio = prune_ratios,
        past_key_value = DynamicCache(),
        reduction_mode="query_text_key_image",
        prune_layer = 1,
        v_mask = mask,
        pad_masks = pad_masks,
    )
    torch.cuda.empty_cache()
    #######################################################
    kv_cache = merged_sample["past_key_values"]
    batch_size, _, prefix_len, _ = kv_cache.key_cache[0].shape

    # Get last token's output
    new_token_idx = merged_sample["logits"][0, -1].argmax()
    new_input_ids = new_token_idx.unsqueeze(0).unsqueeze(0).to(device)
    input_ids = torch.cat(
        [
            torch.ones((batch_size, prefix_len), device=device).long(),
            new_input_ids,
        ], dim=-1
    )
    ##################################################################################
    generated_ids = model.generate(
                input_ids=input_ids, 
                past_key_values=kv_cache.to(device), 
                use_cache=True,
                max_new_tokens=max_token,
            )
    
    generated_ids_trimmed = generated_ids[0][input_ids.size(1)-1:] # [-100:] 
    output = processor.batch_decode(
        [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # if visualize:
    #     from utils.vis_utils import visualize_attention
    #     total_layers = len(model.model.layers)
    #     for batch_attn in range(0, generated_ids.size(0)):
    #         for layer_attn in range(0, total_layers):
    #             v_mask = vision_mask[batch_attn].cpu()
    #             attn = output_attn[layer_attn][batch_attn].cpu() # n_heads x n_tokens x n_tokens
    #             top5_attention, average_attentions = visualize_attention( # plot attention score with y axis = query: text tokens & x axis = key: visual tokens
    #                 attn, 
    #                 output_path = f"logs/attn_maps/attn_layer{layer_attn}_batch{batch_attn}.png",
    #                 v_mask = v_mask
    #                 )
    
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
    image_embeds = model.visual(pixel_values.to(device), grid_thw = grid_thw.to(device)).to("cpu") #! batchify
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

    # assert model.config.pretraining_tp <= 1
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
    
    input_ids, attn_mask = [], []
    for b_idx, sample in enumerate(inputs.input_ids):
        mask = (sample == model.config.bos_token_id) # since batch produces padding (w/ bos), remove them before concat
        input_ids.append(sample[~mask]) # 1, seq_len
        attn_mask.append(inputs.attention_mask[b_idx][~mask])
    input_ids = torch.cat(input_ids).unsqueeze(0)
    attn_mask = torch.cat(attn_mask).unsqueeze(0)

    generated_ids = model.generate(
        input_ids=input_ids, 
        attention_mask=attn_mask, 
        pixel_values = inputs.pixel_values,
        image_grid_thw = inputs.image_grid_thw,
        max_new_tokens=max_token
        )
    input_len = input_ids.size(1)
    generated_ids_trimmed = generated_ids[0, input_len-100:]
    del inputs
    torch.cuda.empty_cache()
    output = processor.batch_decode(
        [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output

if __name__=="__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--prune_ratio", 
                        nargs='+',             # Accept one or more values
                        type=float,
                        default=None
                        )
    parser.add_argument("--infer_type", type=str, default="token_prune")
    parser.add_argument("--max_token", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8, choices=[4, 8, 16])
    args = parser.parse_args()
    log_path = f"/home/server08/yoonjeon_workspace/VideoPrefill/logs/PhysBenchmark/{args.prune_ratio}/"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
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
    
    model_answers = []
    dataset = load_val_dataset(dataset_path) #* evaluate with validation set (200 samples, gt answers are provided)
    for data in dataset:
        texts, images, videos = prepare_batched_icl(data, dataset, dataset_path, args.topk, processor)
        
        if args.infer_type=="homer": # hierarchical merging [https://github.com/alinlab/HOMER/tree/main/src/homer]
            chunk_len = 2048 # final token length limit
            warmup_layer = 8 # forward through first 8 layers before pruning
            output = qwen2_homer_generate(model, processor, texts, images, videos, chunk_len = chunk_len, warmup_layer = warmup_layer, max_token = args.max_token) #! Needs debugging
        elif args.infer_type=="token_prune":
            output = qwen2_prune_generate(model, processor, texts, images, videos, args.prune_ratio, max_token = args.max_token) #* operating seamlessly
        elif args.infer_type=="original":
            output = qwen2_original_generate(model, processor, texts, images, videos, max_token = args.max_token) #* operating seamlessly
        print(output)
        model_answers.append({
				"idx": data["idx"],
				"answer": output,
			})
			
		
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(model_answers, f, ensure_ascii=False, indent=4)

