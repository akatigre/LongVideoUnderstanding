import torch
import argparse
from qwen_vl_utils import process_vision_info       
from transformers import AutoProcessor
from homer.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from pathlib import Path
import json
import os
from tqdm import tqdm
from video_utils import load_video

def eval_lvb(args):
    from lvb_utils import longvideobench_doc_to_text, write_or_append_json, longvideobench_process_results
    metadata_path = Path(args.data_path) / "LongVideoBench/lvb_val.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    json_name = f"log/LVB/{args.model_name}_{args.attn_type}_nframes_{args.max_frames_num}"
    json_name += "_subtitle" if args.subtitle else ""
    
    if os.path.exists(json_name):
        print(f"{json_name} already exists. Skipping evaluation.")
        with open(json_name, "r") as f:
            samples = json.load(f)
        done_ids = [sample["lvb_acc"]["id"] for sample in samples]
        metadata_ids = [sample["id"] for sample in metadata]
        undone_ids = [id for id in metadata_ids if id not in done_ids]
    else:   
        undone_ids = [sample["id"] for sample in metadata]
        
    if len(undone_ids) > 0:
        args.chunk_len = 128000 // args.scale
        model, processor = load(args)
        for meta in tqdm(metadata):
            if meta["id"] not in undone_ids:
                continue
            
            video_path = Path(args.data_path) / 'videos' / f"{meta['video_id']}.mp4"
            frames, frame_timestamps = load_video(video_path, duration=meta["duration"], max_num_frames=args.max_frames_num)
            
            post_prompt = "Answer with the option's letter from the given choices directly.\n"
            if subtitle:
                subtitle_path = Path(args.data_path) / "subtitles" / meta["subtitle_path"]
                with open(subtitle_path, "r") as f:
                    subtitles = json.load(f)

                interleaved_video_subtitles = insert_subtitles_into_frames(
                    frames,
                    frame_timestamps, 
                    subtitles,
                    meta["starting_timestamp_for_subtitles"],
                    meta["duration"]
                    )
                question = meta["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
                return interleaved_video_subtitles, question, post_prompt
            else:
                question = meta["question_wo_referring_query"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
           
            output_text1, output_text2 = generate_text(model, processor, interleaved_video_subtitles, question, post_prompt)
            if output_text1 is None:
                score_dict_orig = None
            else:
                score_dict_orig = longvideobench_process_results(meta, pred=output_text1[0])
                write_or_append_json(f"{args.save_dir}/{json_name}.json", score_dict_orig)
            score_dict_homer = longvideobench_process_results(meta, pred=output_text2[0])
            write_or_append_json(f"{args.save_dir}/{json_name}_homer_chunklen_{args.chunk_len}.json", score_dict_homer)
        
        print(f"Evaluation done. Samples are saved in {json_name}")

def eval_videomme(args):
    import pandas as pd
    from videomme_utils import TASK_CATEGORIES, SUB_CATEGORIES, videomme_doc_to_text_subtitle, videomme_doc_to_text
    cache_dir = os.path.join(args.data_path, "VideoMME")
    metadata_path = os.path.join(cache_dir, "test-00000-of-00001.parquet")
    metadata = pd.read_parquet(metadata_path)
    
    json_name = f"log/VideoMME/{args.model_name}_{args.attn_type}_nframes_{args.max_frames_num}"
    json_name += "_subtitle" if args.subtitle else ""
    
    if os.path.exists(json_name):
        print(f"{json_name} already exists. Skipping evaluation.")
        with open(json_name, "r") as f:
            samples = json.load(f)
        done_ids = [sample["question_id"] for sample in samples]
        metadata_ids = [sample["question_id"] for sample in metadata]
        undone_ids = [id for id in metadata_ids if id not in done_ids]
    else:   
        undone_ids = [sample["question_id"] for sample in metadata]
        
    if len(undone_ids) > 0:
        args.chunk_len = 128000 // args.scale
        model, processor = load(args)
        
        matrices = []
        VIDEO_TYPE = ["long"]
        CATEGORIES = ["Knowledge"]
        for i in VIDEO_TYPE:
            for j in CATEGORIES:
                for k in SUB_CATEGORIES:
                    for l in TASK_CATEGORIES:
                        matrices.append(f"{i}_{j}_{k}_{l}")

        for idx in tqdm(range(len(metadata))):
            meta = metadata.iloc[idx]
            if meta["question_id"] not in undone_ids:
                continue
            
            if args.subtitle:
                prompt = videomme_doc_to_text_subtitle(meta, cache_dir = cache_dir)
            else:
                prompt = videomme_doc_to_text(meta)
            
            frames, frame_timestamps = load_video(
                os.path.join(cache_dir, "data", meta["videoID"] + ".mp4"), 
                duration=meta["duration"], 
                max_num_frames=args.max_frames_num
                )
            
            post_prompt = "Answer with the option's letter from the given choices directly.\n"
            if subtitle:
                subtitle_path = Path(args.data_path) / "subtitles" / meta["subtitle_path"]
                with open(subtitle_path, "r") as f:
                    subtitles = json.load(f)

                interleaved_video_subtitles = insert_subtitles_into_frames(
                    frames,
                    frame_timestamps, 
                    subtitles,
                    meta["starting_timestamp_for_subtitles"],
                    meta["duration"]
                    )
                question = meta["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
                return interleaved_video_subtitles, question, post_prompt
            else:
                question = meta["question_wo_referring_query"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
                return frames, question, post_prompt
            output_text1, output_text2 = generate_text(model, processor, interleaved_video_subtitles, question, post_prompt)
            if output_text1 is None:
                score_dict_orig = None
            else:
                score_dict_orig = longvideobench_process_results(meta, pred=output_text1[0])
                write_or_append_json(f"{args.save_dir}/{json_name}.json", score_dict_orig)
            score_dict_homer = longvideobench_process_results(meta, pred=output_text2[0])
            write_or_append_json(f"{args.save_dir}/{json_name}_homer_chunklen_{args.chunk_len}.json", score_dict_homer)
        
        print(f"Evaluation done. Samples are saved in {json_name}")


def load(args):
    homer_args = {
            "max_chunk_len": args.chunk_len, # final chunk limit
            "max_initial_chunk_len": args.max_initial_chunk_len,
            "reduction_mode": "power_max_last_calibrated",
            "layers_warmup": args.layer_warmup,
            "target_len": args.chunk_len, # max_position_ids = 128000
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
    
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

@torch.no_grad()
def generate_text(model, processor, list_of_pil_images, question, post_prompt, **video_kwargs):
    list_of_content = [
        {
            "type": "video", 
            "video": [ele for ele in list_of_pil_images],
            # "max_pixels": 360 * 420
        },
        {
            "type": "text",
            "text": f"\n{question}\n{post_prompt}",
        }
    ]

    
    messages = [
        {
            "role": "user",
            "content": list_of_content
        },
    ]
        
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    
    #! ORIGINAL Model
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs.to("cuda"), max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text1 = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"Original: {output_text1}")
    
    #! Reasoning globally: DEBUG
    output_text1_reason = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"Global Reasoning: {output_text1_reason}")
    
    #! HOMER
    vision_start_token_id = model.config.vision_start_token_id
    start_idxs = (inputs.input_ids == vision_start_token_id).nonzero()[:]
    vision_end_token_id = model.config.vision_end_token_id
    end_idxs = (inputs.input_ids == vision_end_token_id).nonzero()[:]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    prefix_ids = inputs.input_ids[:, :start_idxs[0][1]]
    context_ids = inputs.input_ids[:, start_idxs[0][1]:end_idxs[-1][1] + 1]
    suffix_ids = inputs.input_ids[:, end_idxs[-1][1] + 1:]
    homer_prefix = model.create_homer_prefix(
        prefix_ids, context_ids, suffix_ids,
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        attention_mask = inputs["attention_mask"], # 1213 ~ 2408
    )
    torch.cuda.empty_cache()
    generated_ids, seq_len = model.generate(homer_prefix, max_new_tokens=128)
    generated_ids_trimmed = generated_ids[0][seq_len-1 :]
    output_text2 = processor.batch_decode(
        [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"Homer: {output_text2}")
    
    #! Reasoning per Chunk

    
    #! MINFERENCE
    # import os
    # from minference import MInference
    # BASE_DIR = "./MInference/minference/configs/"
    # config_path = os.path.join(
    #     BASE_DIR, "Qwen2_7B_Instruct_128k_instruct_kv_out_v32_fit_o_best_pattern.json"
    # )
    # minference_patch = MInference("minference", model_path, config_path=config_path, kv_cache_cpu=True,) # 
    # model = minference_patch(model)
    # generated_ids = model.generate(homer_prefix = None, max_new_tokens=128, **inputs)
    # generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[1] :]
    # output_text = processor.batch_decode(
    #     [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # print(output_text)
    
    #! MINFERENCE + HOMER
    # generated_ids, seq_len = model.generate(homer_prefix, max_new_tokens=128) # homer_prefix should be the same format with inputs = processor()
    # generated_ids_trimmed = generated_ids[0][seq_len - 1 :]
    # output_text = processor.batch_decode(
    #     [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # print(output_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=5)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--video-path", type=str, default="examples/dc_demo.mp4")
    parser.add_argument("--text", type=str, default="What is the video about?")
    parser.add_argument("--max-frames-num", type=int, default=100)
    parser.add_argument("--subtitle", action="store_true")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument(
        "--model_type",
        type=str,
        default="plain",
        choices=["plain", "yarn", "homer", "homer_yarn"],
    )
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="./dataset/")
    # HOMER arguments
    parser.add_argument("--max_initial_chunk_len", type=int, default=-1)
    parser.add_argument("--layer_warmup", type=int, default=12)
    parser.add_argument("--bias_path", type=str, default=None)
    # Task-specific arguments
    parser.add_argument("--num_test_samples", type=int, default=-1)
    args = parser.parse_args()

    args.model_name = "qwen2-5-vl"
    args.attn_type = "dense"
    # eval(args)

    eval_videomme(args)
    # test(args)
    