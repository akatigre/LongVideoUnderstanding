
import json
import torch
from pathlib import Path
from transformers import AutoProcessor
from run_qwen2 import load_model, prepare_batched_icl
from dataset_utils.physbench_utils import load_val_dataset
from utils.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from dataset_utils.physbench_utils import _process_cls_embedding, _process_cossim 
from dataset_utils.physbench_utils import process_vision_info

images_path = "/home/server08/yoonjeon_workspace/VideoPrefill/logs/examples/physbenchmark/000"

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", padding_side="left")
images = [
    "/home/server08/yoonjeon_workspace/VideoPrefill/dataset_utils/LongVideoBench/leaderboard_paper.png",
    "/home/server08/yoonjeon_workspace/VideoPrefill/dataset_utils/LongVideoBench/logo.png",
    "/home/server08/yoonjeon_workspace/VideoPrefill/dataset_utils/LongVideoBench/lvb_teaser.png",
]
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", 
             "image": images[0], 
             "resized_height": 300,
             "resized_width": 300
            },
            {"type": "text", "text": "What are the common elements in these pictures?"},
        ],
    }
]
messages2 = [
    {"role": "user", "content": [
        {'type': 'image', 
         'image': images[2],
         "resized_height": 300,
         "resized_width": 300
         },
        {'type': 'text', 'text': 'Hey?'}
    ]
    }
]

messages3 = [
    {"role": "user", "content": [
        {'type': 'image', 
         'image': images[1],
         "resized_height": 300,
         "resized_width": 300
         },
        {'type': 'text', 'text': 'Which is the important elements in this image?'}
    ]
    }
]
# Combine messages for batch processing
messages = [messages1, messages2, messages3]

texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
breakpoint()
inputs = inputs.to("cuda")

# Batch Inference
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_texts)