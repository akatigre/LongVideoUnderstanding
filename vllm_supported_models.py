"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
from vllm import LLM
from transformers import AutoTokenizer



# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.


# Aria
def run_aria(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"
    model_name = "rhymes-ai/Aria"

    # NOTE: Need L40 (or equivalent) to avoid OOM
    llm = LLM(model=model_name,
              tokenizer_mode="slow",
              dtype="bfloat16",
              max_model_len=4096,
              max_num_seqs=2,
              trust_remote_code=True,
              disable_mm_preprocessor_cache=disable_mm_preprocessor_cache)

    prompt = (f"<|im_start|>user\n<fim_prefix><|img|><fim_suffix>\n{question}"
              "<|im_end|>\n<|im_start|>assistant\n")

    stop_token_ids = [93532, 93653, 944, 93421, 1019, 93653, 93519]
    return llm, prompt, stop_token_ids


# BLIP-2
def run_blip2(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompt = f"Question: {question} Answer:"
    llm = LLM(model="Salesforce/blip2-opt-2.7b",
              disable_mm_preprocessor_cache=disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Chameleon
def run_chameleon(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    prompt = f"{question}<image>"
    llm = LLM(model="facebook/chameleon-7b",
              max_model_len=4096,
              max_num_seqs=2,
              disable_mm_preprocessor_cache=disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Fuyu
def run_fuyu(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    prompt = f"{question}\n"
    llm = LLM(model="adept/fuyu-8b",
              max_model_len=2048,
              max_num_seqs=2,
              disable_mm_preprocessor_cache=disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# GLM-4v
def run_glm4v(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"
    model_name = "THUDM/glm-4v-9b"

    llm = LLM(model=model_name,
              max_model_len=2048,
              max_num_seqs=2,
              trust_remote_code=True,
              enforce_eager=True,
              disable_mm_preprocessor_cache=disable_mm_preprocessor_cache)
    prompt = question
    stop_token_ids = [151329, 151336, 151338]
    return llm, prompt, stop_token_ids


# H2OVL-Mississippi
def run_h2ovl(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    model_name = "h2oai/h2ovl-mississippi-2b"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # Stop tokens for H2OVL-Mississippi
    # https://huggingface.co/h2oai/h2ovl-mississippi-2b
    stop_token_ids = [tokenizer.eos_token_id]
    return llm, prompt, stop_token_ids


# Idefics3-8B-Llama3
def run_idefics3(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"
    model_name = "HuggingFaceM4/Idefics3-8B-Llama3"

    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        enforce_eager=True,
        # if you are running out of memory, you can reduce the "longest_edge".
        # see: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3#model-optimizations
        mm_processor_kwargs={
            "size": {
                "longest_edge": 3 * 364
            },
        },
        disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
    )
    prompt = (
        f"<|begin_of_text|>User:<image>{question}<end_of_utterance>\nAssistant:"
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# InternVL
def run_internvl(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    model_name = "OpenGVLab/InternVL2-2B"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return llm, prompt, stop_token_ids


# LLaVA-1.5
def run_llava(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    llm = LLM(model="llava-hf/llava-1.5-7b-hf",
              max_model_len=4096,
              disable_mm_preprocessor_cache=disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    prompt = f"[INST] <image>\n{question} [/INST]"
    llm = LLM(model="llava-hf/llava-v1.6-mistral-7b-hf",
              max_model_len=8192,
              disable_mm_preprocessor_cache=disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LlaVA-NeXT-Video
# Currently only support for video input
def run_llava_next_video(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "video"

    prompt = f"USER: <video>\n{question} ASSISTANT:"
    llm = LLM(model="llava-hf/LLaVA-NeXT-Video-7B-hf",
              max_model_len=8192,
              disable_mm_preprocessor_cache=disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLaVA-OneVision
def run_llava_onevision(question: str, modality: str, disable_mm_preprocessor_cache):

    if modality == "video":
        prompt = f"<|im_start|>user <video>\n{question}<|im_end|> \
        <|im_start|>assistant\n"

    elif modality == "image":
        prompt = f"<|im_start|>user <image>\n{question}<|im_end|> \
        <|im_start|>assistant\n"

    llm = LLM(model="llava-hf/llava-onevision-qwen2-7b-ov-hf",
              max_model_len=16384,
              disable_mm_preprocessor_cache=disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Mantis
def run_mantis(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'  # noqa: E501
    prompt = llama3_template.format(f"{question}\n<image>")

    llm = LLM(
        model="TIGER-Lab/Mantis-8B-siglip-llama3",
        max_model_len=4096,
        hf_overrides={"architectures": ["MantisForConditionalGeneration"]},
        disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
    )
    stop_token_ids = [128009]
    return llm, prompt, stop_token_ids


# MiniCPM-V
def run_minicpmv(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    # 2.0
    # The official repo doesn't work yet, so we need to use a fork for now
    # For more details, please see: See: https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630 # noqa
    # model_name = "HwwwH/MiniCPM-V-2"

    # 2.5
    # model_name = "openbmb/MiniCPM-Llama3-V-2_5"

    # 2.6
    model_name = "openbmb/MiniCPM-V-2_6"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
    )
    # NOTE The stop_token_ids are different for various versions of MiniCPM-V
    # 2.0
    # stop_token_ids = [tokenizer.eos_id]

    # 2.5
    # stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

    # 2.6
    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    messages = [{
        'role': 'user',
        'content': f'(<image>./</image>)\n{question}'
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return llm, prompt, stop_token_ids


# LLama 3.2
def run_mllama(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        enforce_eager=True,
        disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "image"
        }, {
            "type": "text",
            "text": f"{question}"
        }]
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=False)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Molmo
def run_molmo(question, modality:str, disable_mm_preprocessor_cache):
    assert modality == "image"

    model_name = "allenai/Molmo-7B-D-0924"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
    )

    prompt = question
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# NVLM-D
def run_nvlm_d(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    model_name = "nvidia/NVLM-D-72B"

    # Adjust this as necessary to fit in GPU
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        tensor_parallel_size=4,
        disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# PaliGemma
def run_paligemma(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    # PaliGemma has special prompt format for VQA
    prompt = "caption en"
    llm = LLM(model="google/paligemma-3b-mix-224",
              disable_mm_preprocessor_cache=disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# PaliGemma 2
def run_paligemma2(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    # PaliGemma 2 has special prompt format for VQA
    prompt = "caption en"
    llm = LLM(model="google/paligemma2-3b-ft-docci-448",
              disable_mm_preprocessor_cache=disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Phi-3-Vision
def run_phi3v(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"

    # num_crops is an override kwarg to the multimodal image processor;
    # For some models, e.g., Phi-3.5-vision-instruct, it is recommended
    # to use 16 for single frame scenarios, and 4 for multi-frame.
    #
    # Generally speaking, a larger value for num_crops results in more
    # tokens per image instance, because it may scale the image more in
    # the image preprocessing. Some references in the model docs and the
    # formula for image tokens after the preprocessing
    # transform can be found below.
    #
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct#loading-the-model-locally
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/main/processing_phi3_v.py#L194
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        mm_processor_kwargs={"num_crops": 16},
        disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Pixtral HF-format
def run_pixtral_hf(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    model_name = "mistral-community/pixtral-12b"

    # NOTE: Need L40 (or equivalent) to avoid OOM
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
    )

    prompt = f"<s>[INST]{question}\n[IMG][/INST]"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Qwen
def run_qwen_vl(question: str, modality: str, disable_mm_preprocessor_cache):
    assert modality == "image"

    llm = LLM(
        model="Qwen/Qwen-VL",
        trust_remote_code=True,
        max_model_len=1024,
        max_num_seqs=2,
        disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
    )

    prompt = f"{question}Picture 1: <img></img>\n"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Qwen2-VL
def run_qwen2_vl(question: str, modality: str, disable_mm_preprocessor_cache):

    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


model_example_map = {
    "aria": run_aria,
    "blip-2": run_blip2,
    "chameleon": run_chameleon,
    "fuyu": run_fuyu,
    "glm4v": run_glm4v,
    "h2ovl_chat": run_h2ovl,
    "idefics3": run_idefics3,
    "internvl_chat": run_internvl,
    "llava": run_llava,
    "llava-next": run_llava_next,
    "llava-next-video": run_llava_next_video,
    "llava-onevision": run_llava_onevision,
    "mantis": run_mantis,
    "minicpmv": run_minicpmv,
    "mllama": run_mllama,
    "molmo": run_molmo,
    "NVLM_D": run_nvlm_d,
    "paligemma": run_paligemma,
    "paligemma2": run_paligemma2,
    "phi3_v": run_phi3v,
    "pixtral_hf": run_pixtral_hf,
    "qwen_vl": run_qwen_vl,
    "qwen2_vl": run_qwen2_vl,
}

