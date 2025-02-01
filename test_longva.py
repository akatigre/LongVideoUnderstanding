from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
from minference import MInference
import os
import warnings
from run_hf import load_longva_model

warnings.filterwarnings("ignore")
torch.manual_seed(0)
attn_type = "minference_homer" # "flexprefill" "minference" "minference_homer"
model_path = "lmms-lab/LongVA-7B-DPO"
image_path = "examples/user_example_01.jpg"
video_path = "examples/dc_demo.mp4"
max_frames_num = 16 # you can change this to several thousands so long you GPU memory can handle it :)
gen_kwargs = {"do_sample": True, "temperature": 0.5, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}

model, tokenizer, image_processor = load_longva_model(model_path, "auto", dtype="auto")
dtype = model.dtype
if attn_type == "minference":
    BASE_DIR = "./MInference/minference/configs/"
    config_path = os.path.join(
        BASE_DIR, "Qwen2_7B_Instruct_128k_instruct_kv_out_v32_fit_o_best_pattern.json"
    )
    minference_patch = MInference("minference", model_path, config_path=config_path, kv_cache_cpu=True,) # 
    model = minference_patch(model)
elif attn_type == "flexprefill":
    minference_patch = MInference(
            "flexprefill", 
            model_path, 
            attn_kwargs = {
                "gamma": 0.9, 
                "tau": 0.1,
                "min_budget": None,
                "max_budget": None,
                "block_size": 128
                }
            )
    model = minference_patch(model)
elif attn_type == "minference_homer":
    BASE_DIR = "./MInference/minference/configs/"
    config_path = os.path.join(
        BASE_DIR, "Qwen2_7B_Instruct_128k_instruct_kv_out_v32_fit_o_best_pattern.json"
    )
    minference_patch = MInference("minference_homer", model_path, config_path=config_path, kv_cache_cpu=True,) # 
    model = minference_patch(model)
elif attn_type == "flexprefill_homer":
    minference_patch = MInference("flexprefill_homer", model_path, attn_kwargs = {
                "gamma": 0.9, 
                "tau": 0.1, 
                "min_budget": None,
                "max_budget": None,
                "block_size": 64
                })
    model = minference_patch(model)
    
#image input
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nDescribe the image in details.<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
image = Image.open(image_path).convert("RGB")

images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=dtype)

with torch.inference_mode():
    output_ids = model.generate(inputs=input_ids, images=images_tensor, image_sizes=[image.size], modalities=["image"], **gen_kwargs)
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(outputs)
print("-" * 50)

#video input
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nGive a detailed caption of the video as if I am blind.<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
vr = VideoReader(video_path, ctx=cpu(0))
total_frame_num = len(vr)
uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
frame_idx = uniform_sampled_frames.tolist()
frames = vr.get_batch(frame_idx).asnumpy()
video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=dtype)

with torch.inference_mode():
    output_ids = model.generate(inputs=input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(outputs)