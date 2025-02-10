import torch
import argparse
from qwen_vl_utils import process_vision_info       
from transformers import AutoProcessor
from homer.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm
from longvideobench import LongVideoBenchDataset
from lvb_utils import longvideobench_process_results, write_or_append_json
from PIL import Image

def load_model(args):
    
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

def debug(task_name="LongVideoBench"):
    import os
    if task_name == "LongVideoBench":
        video_id = 'yn7oTvw8QRY'
        question = """
                There are two images here. One shows a girl in green clothing with braided hair, holding a clay container in front of a solid color background wall. 
                The other shows a girl in black and white floral clothing with loose hair. According to the video, which character appears first?
                """
        subtitle_path = os.path.join("/home/server08/hdd1/yoonjeon_workspace/long_video_understanding/LongVideoBench/subtitles/", video_id+"_en.json")
        video_path = os.path.join("/home/server08/hdd1/yoonjeon_workspace/long_video_understanding/LongVideoBench/videos/", video_id+".mp4")
        options = [
            "A. Boy with short hair and green stripes",
            "B. Boy with golden hair",
            "C. Girl in green clothing with loose hair",
            "D. Girl in green clothing with braided hair",
            "E. Girl in black and white floral clothing with loose hair"
            ]
        starting_timestep_for_subtitles = 0
        # subtitle_path = None
    elif task_name == "videomme":
        video_id = "QTA8j5wSTx4"
        question = "Why did the main character in the video put his hand in the coat?"
        options = [ "A. Because his hand was deformed.", "B. For public image.", "C. Because such action can released stomach pain.", "D. For warmth." ]
        answer = "B"
        subtitle_path = os.path.join("/home/server08/.cache/huggingface/videomme", "subtitle", f"{video_id}.srt")
        video_path = os.path.join("/home/server08/.cache/huggingface/videomme", "data", f"{video_id}.mp4")
    images_path = os.path.join("/home/server08/yoonjeon_workspace/VideoPrefill/examples", video_id)
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    
    # for frame_idx, frame in enumerate(frames):
    #     frame.save(os.path.join(images_path, f"{frame_idx:04d}.png"))
        
def load_data(task_name="LongVideoBench", max_frame=100):
    return
    from video_utils import load_video
    from videomme_utils import extract_subtitles
    
    if task_name == "LongVideoBench" and args.subtitle:
        import json
        from lvb_utils import insert_subtitles_into_frames
        starting_timestep_for_subtitles = 0
        with open(subtitle_path, "r") as f:
            subtitles = json.load(f)
        frame_timestamps = [subtitle["timestamp"] for subtitle in subtitles]
        duration = frame_timestamps[-1]["end"]
        interleaved_list = insert_subtitles_into_frames(frames, frame_timestamps, subtitles, starting_timestep_for_subtitles, duration)
        
    elif task_name == "videomme" and args.subtitle:
        import re
        from collections import defaultdict
        subtitle_by_frame = extract_subtitles(video_path, subtitle_path)
        sample_by = total_frames // max_frame
        per_frame_subtitles = defaultdict(list)
        for start, end, subt in subtitle_by_frame:
            # start, end is the frame index ranging between 0 and total_frames - 1
            start_scaled, end_scaled = start / sample_by, end / sample_by
            pattern = r'<font color="white" size=".72c">(.*?)</font>'
            raw_text = re.findall(pattern, subt) # clean text
            per_frame_subtitles[int(start_scaled)].append(raw_text[0])
            if int(start_scaled) != int(end_scaled):
                per_frame_subtitles[int(end_scaled)].append(raw_text[0])
    
    return question, options, frames, subtitle_by_frame, total_frames, per_frame_subtitles

@torch.no_grad()
def generate_text(model, processor, messages, max_token):
    
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
    generated_ids = model.generate(**inputs, max_new_tokens=max_token)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text1 = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    torch.cuda.empty_cache()
    #! HOMER
    from homer.merge import MergeManager
    #################### MODIFIED FROM ORIGINAL CODE #####################
    args.chunk_len = model.config.max_position_embeddings // args.scale
    manager_settings = {
            "max_chunk_len": args.chunk_len, # final chunk limit
            "max_initial_chunk_len": args.max_initial_chunk_len,
            "reduction_mode": "power_max_last_calibrated",
            "layers_warmup": args.layer_warmup,
            "target_len": args.chunk_len, # max_position_ids = 128000
            "bias_path": args.bias_path,
            "visualize": False,
        }
    model.merge_manager = MergeManager(**manager_settings)

    for idx, layer in enumerate(model.model.layers):
        layer.layer_idx = idx
        layer.merge_manager = model.merge_manager
    ######################################################################
    homer_prefix = model.create_homer_prefix(
        input_ids = inputs['input_ids'],
        pixel_values=inputs["pixel_values_videos"],
        grid_thw=inputs["video_grid_thw"],
        vision_start_token_id = model.config.vision_start_token_id,
        vision_end_token_id = model.config.vision_end_token_id,
    )
    torch.cuda.empty_cache()
    generated_ids, seq_len = model.generate(homer_prefix, max_new_tokens=max_token)
    generated_ids_trimmed = generated_ids[0][seq_len-1 :]
    output_text2 = processor.batch_decode(
        [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text1, output_text2
    

def make_video(max_frame, video_frames, per_frame_subtitles, subtitle_by_frame):
    
    from video_utils import save_video_from_pil
    from PIL import ImageDraw, ImageFont
    subtitle_images = []
    for frame_idx in range(max_frame):
        frame = video_frames[frame_idx]
        if subtitle_by_frame is not None:
            image_width, image_height = frame.size
            draw = ImageDraw.Draw(frame)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30) 
            
            text = "\n".join(per_frame_subtitles[frame_idx])
            bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (image_width - text_width) / 2
            y = image_height - text_height - 10  # 10 pixels from the bottom
            draw.text((x, y), text, font=font, fill="white")
        subtitle_images.append(frame)
    video_path = f"examples/video_frames{max_frame}.mp4"
    save_video_from_pil(subtitle_images, video_path, fps=1)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=5)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--video-path", type=str, default="examples/dc_demo.mp4")
    parser.add_argument("--text", type=str, default="What is the video about?")
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--subtitle", action="store_true")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument(
        "--model_type",
        type=str,
        default="plain",
        choices=["plain", "yarn", "homer", "homer_yarn"],
    )
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--task_name", type=str, default="LongVideoBench", choices=["LongVideoBench", "videomme"])
    parser.add_argument("--data_path", type=str, default="/home/server08/.cache/huggingface/hub/")
    parser.add_argument("--num_frames_per_chunk", type=int, default=16)
    parser.add_argument("--reason", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    # HOMER arguments
    parser.add_argument("--max_initial_chunk_len", type=int, default=-1)
    parser.add_argument("--layer_warmup", type=int, default=12)
    parser.add_argument("--bias_path", type=str, default=None)
    # Task-specific arguments
    parser.add_argument("--num_test_samples", type=int, default=-1)
    args = parser.parse_args()

    args.model_name = "qwen2-5-vl"
    args.attn_type = "dense"

    model, processor = load_model(args)
    
    if args.task_name == "LongVideoBench":
        
        data_path = "/home/server08/.cache/huggingface/longvideobench/"
        dataset = LongVideoBenchDataset(data_path, "lvb_val.json", target_fps=args.fps)
        failure_list = []
        save_dir = "./logs/LVB"
        json_name = f"qwen2-5-vl_fps{args.fps}"
        for data in tqdm(iter(dataset)):
            inputs = data["inputs"] # list of interleaved pil images and subtitles, followed by question and options
            suffix = '\n'.join(inputs[-6:])
            context = inputs[:-6]
            
            list_of_content = []
            #! Chunk context into multiple clips
            per_clip_frames = 0
            clip, subtitle = [], []
            if args.subtitle:
                for inp in context:
                    if isinstance(inp, Image.Image):
                        per_clip_frames += 1
                        clip.append(inp)
                        list_of_content.append({
                            "type": "video",
                            "video": [inp]
                        })
                    elif isinstance(inp, str):
                        subtitle.append(inp)
            else:
                list_of_content = [{
                    "type": "video",
                    "video": [inp for inp in context if isinstance(inp, Image.Image)]
                }]
                    
            #! Add question and options
            if args.reason:
                suffix = '\n'.join(inputs[-5:])
                list_of_content += [
                    {
                        "type": "text",
                        "text": f"Reason about the thinking process to answer. Does the following video contain the answer for {inputs[-6]}? {suffix}."
                    }
                ]
                max_token = 128
            else:
                list_of_content += [
                    {
                        "type": "text",
                        "text": suffix
                    }
                ]
                max_token = 32

            messages = [
                {
                    "role": "user",
                    "content": list_of_content
                },
            ]
            
            try:
                output1, output2 = generate_text(model, processor, messages, max_token)
            except:
                failure_list.append(data['id'])
            score_dict_orig = longvideobench_process_results(data, pred=output1[0])
            write_or_append_json(f"{save_dir}/{json_name}.json", score_dict_orig)
            score_dict_homer = longvideobench_process_results(data, pred=output2[0])
            write_or_append_json(f"{save_dir}/{json_name}_homer_chunklen_{args.chunk_len}.json", score_dict_homer)
        
        print(f"Evaluation done. Samples are saved in {json_name}")