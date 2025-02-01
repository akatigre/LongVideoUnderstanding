import json
from tqdm import tqdm
from pathlib import Path

from lvb_utils import longvideobench_doc_to_text, longvideobench_process_results, write_or_append_json
from run_hf import *
from tqdm import trange
from transformers.generation.utils import GenerationMixin

HOMER = True

def gather_logits(logits, input_ids):
    input_ids = input_ids[0, 1 : logits.size(1) + 1].unsqueeze(-1)
    return logits.log_softmax(dim=-1)[0].gather(1, input_ids).squeeze()

def evaluate_multiple(model, tokenizer, vision_processor, metadata, json_name, undone_ids, data_path, system_prompt):
    for meta in tqdm(metadata):
        if meta["id"] not in undone_ids:
            continue
        video_path = Path(data_path) / 'videos' / f"{meta['video_id']}.mp4"
        question = longvideobench_doc_to_text(meta, max_num_frames = args.num_frames, data_path = data_path, subtitle=False, system_prompt=system_prompt)
        device = next(model.parameters()).device
        dtype = model.dtype
        inputs = prepare_fn(
            video_path = video_path, 
            text = question, 
            tokenizer = tokenizer, 
            model = model,
            vision_processor = vision_processor, 
            max_frames_num = args.num_frames, 
            device=device, 
            dtype=dtype
            )
        output_ids = generate_fn(model=model, tokenizer=tokenizer, inputs=inputs,)
    
        decoder = tokenizer if hasattr(tokenizer, "batch_decode") else vision_processor
        outputs = decoder.batch_decode(
            output_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
            )[0].strip() # tokenizer OR processor
        score_dict = longvideobench_process_results(meta, pred=outputs)
        
        write_or_append_json(json_name, score_dict)
        
def main(args):
    metadata_path = Path(args.data_path) / "lvb_val.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    json_name = f"{args.save_dir}/answers_{args.model_name}_{args.attn_type}_nframes_{args.num_frames}.json"
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
        model, tokenizer, image_processor = prepare_models(args, load_fn)
        evaluate_multiple(model, tokenizer, image_processor, metadata, json_name, undone_ids, data_path=args.data_path, system_prompt=system_prompt)
        print(f"Evaluation done. Samples are saved in {json_name}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--attn-type", type=str, default="dense", choices=["dense", "flexprefill_homer", "minference_homer", "minference", "flexprefill", "inf_llm", "hf", "a_shape", "minference_with_dense"])
    parser.add_argument("--data-path", type=str, default="./dataset/LongVideoBench")
    parser.add_argument("--save-dir", type=str, default="./lvb_answers")
    parser.add_argument("--num-frames", type=int, default=256, choices=[64, 256, 400, 512])
    parser.add_argument("--device-map", type=str, default="auto", choices=["auto","cuda"])
    parser.add_argument("--model-name", type=str, choices=["longVA", "qwen2-vl", "longvila-1.5b", "longvila-7b"])
    args = parser.parse_args()
    
    load_fn_map = {
        "longVA": load_longva_model,
        "qwen2-vl": load_qwen2_model,
        "longvila-1.5b": load_longvila_model,
        "longvila-7b": load_longvila_model,
    }
    prepare_fn_map = {
        "longVA": longva_prepare_embeds,
        "qwen2-vl": qwen2_prepare_embeds,
        "longvila-1.5b": longvila_prepare_embeds,
        "longvila-7b": longvila_prepare_embeds,
    }
    generate_fn_map = {
        "longVA": longva_generate_answer,
        "qwen2-vl": qwen2_generate_answer,
        "longvila-1.5b": longvila_generate_answer,
        "longvila-7b": longvila_generate_answer,
    }
    system_prompt = model_system_prompt(args.model_name)
    load_fn = load_fn_map[args.model_name]
    prepare_fn = prepare_fn_map[args.model_name]
    generate_fn = generate_fn_map[args.model_name]
    main(args)