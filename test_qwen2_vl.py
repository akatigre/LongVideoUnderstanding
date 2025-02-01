import torch
import argparse
from qwen_vl_utils import process_vision_info       
from transformers import AutoProcessor
from homer.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

@torch.no_grad()
def main(args):
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
    
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": args.video_path,
                    "fps": 10.0
                },
                {
                    "type": "text", 
                    "text": args.text
                },
            ],
        }
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
    inputs = inputs.to("cuda")
    #! ORIGINAL Model
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    
    #! HOMER
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
    generated_ids_trimmed = generated_ids[0][seq_len :]
    output_text = processor.batch_decode(
        [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    
    #! MINFERENCE
    import os
    from minference import MInference
    BASE_DIR = "./MInference/minference/configs/"
    config_path = os.path.join(
        BASE_DIR, "Qwen2_7B_Instruct_128k_instruct_kv_out_v32_fit_o_best_pattern.json"
    )
    minference_patch = MInference("minference_homer", model_path, config_path=config_path, kv_cache_cpu=True,) # 
    model = minference_patch(model)
    generated_ids = model.generate(homer_prefix = None, max_new_tokens=128)
    generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[1] :]
    output_text = processor.batch_decode(
        [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    
    #! MINFERENCE + HOMER
    generated_ids, seq_len = model.generate(homer_prefix, max_new_tokens=128) # homer_prefix should be the same format with inputs = processor()
    generated_ids_trimmed = generated_ids[0][seq_len :]
    output_text = processor.batch_decode(
        [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--video-path", type=str, default="examples/dc_demo.mp4")
    parser.add_argument("--text", type=str, default="What is the video about?")
    parser.add_argument("--max-frames-num", type=int, default=100)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument(
        "--model_type",
        type=str,
        default="plain",
        choices=["plain", "yarn", "homer", "homer_yarn"],
    )
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--max_position_embeddings", type=int, default=4096)
    parser.add_argument("--gen_length", type=int, default=20)
    # HOMER arguments
    parser.add_argument("--max_initial_chunk_len", type=int, default=-1)
    parser.add_argument("--layer_warmup", type=int, default=12)
    parser.add_argument("--bias_path", type=str, default=None)
    # Task-specific arguments
    parser.add_argument("--num_test_samples", type=int, default=-1)
    args = parser.parse_args()

    main(args)
    