import os 
import sys
import json
from glob import glob
# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)
from lvb_utils import evaluate_longvideobench

def process_answers(attn_json):
    with open(attn_json, "r") as f:
        samples = json.load(f)
    judge_dict, acc = evaluate_longvideobench(samples)
    return judge_dict, acc, len(samples)

if __name__ == "__main__":
    answers = sorted(glob(os.path.join(os.path.dirname(__file__), "answers_qwen2-5*.json")))
    for answer in answers:
        judge_dict, acc, num_samples = process_answers(answer)
        print(f"------------{answer}------------")
        print(f"Num {num_samples}: {acc}")
