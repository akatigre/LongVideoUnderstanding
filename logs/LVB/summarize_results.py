import os 
import sys
import json
from glob import glob
# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, parent_dir)
from lvb_utils import evaluate_longvideobench

def process_answers(attn_json, list_ids=None):
    with open(attn_json, "r") as f:
        samples = json.load(f)
    judge_dict, acc = evaluate_longvideobench(samples, list_ids)
    return judge_dict, acc, len(judge_dict)

def summarize_results(results_dir):
    answers = sorted(glob(os.path.join(os.path.dirname(__file__), "answers_qwen2-5*.json")))
    for answer in answers:
        judge_dict, acc, num_samples = process_answers(answer)
        print(f"------------{answer}------------")
        print(f"Num {num_samples}: {acc}")

def compare_two_results(results_dir1, results_dir2):
    comp1 = []
    with open(results_dir1, "r") as f:
        answers1 = json.load(f)
    for ans in answers1:
        comp1.append(ans["lvb_acc"]["id"])
    comp2 = []
    with open(results_dir2, "r") as f:
        answers2 = json.load(f)
    for ans in answers2:
        comp2.append(ans["lvb_acc"]["id"])
        
    both = list(set(comp1) & set(comp2))
    judge_dict1, acc1, num_samples1 = process_answers(results_dir1, both)
    judge_dict2, acc2, num_samples2 = process_answers(results_dir2, both)
    print(f"------------{results_dir1}------------")
    print(f"Num {num_samples1}: {acc1}")
    print(f"------------{results_dir2}------------")
    print(f"Num {num_samples2}: {acc2}")

if __name__ == "__main__":
    compare_two_results(
        "/home/server08/yoonjeon_workspace/VideoPrefill/logs/LVB/qwen2-5-vl_dense_nframes_100_homer_chunklen_25600.json",
        "/home/server08/yoonjeon_workspace/VideoPrefill/logs/LVB/qwen2-5-vl_dense_nframes_100.json"
        )