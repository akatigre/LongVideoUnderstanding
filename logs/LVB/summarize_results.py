import os 
import sys
import json
from glob import glob
# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, parent_dir)
from lvb_utils import evaluate_longvideobench, eval_multi_choice

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
        
def compare_correct_wrong(samplesA, samplesB, list_ids=None):
    
    # judge_dict, acc = evaluate_longvideobench(samples, list_ids)
    pred_correct = 0
    judge_dict = dict()
    for sampleA, sampleB in zip(samplesA, samplesB):
        # if list_ids is not None:
        #     if sampleA["lvb_acc"]["id"] not in list_ids:
        #         continue
        assert sampleA["lvb_acc"]["id"] == sampleB["lvb_acc"]["id"]
        gold_i = sampleA["lvb_acc"]["answer"]
        pred_i_A = sampleA["lvb_acc"]["parsed_pred"]
        pred_i_B = sampleB["lvb_acc"]["parsed_pred"]
        correctA = eval_multi_choice(gold_i, pred_i_A)
        correctB = eval_multi_choice(gold_i, pred_i_B)
        if correctA != correctB:
            print(f"------------{sampleA['lvb_acc']['id']}------------")
            print(f"Correct: {correctA}, {correctB}")
            print(f"Pred: {pred_i_A}, {pred_i_B}")
        

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
        "/home/server08/yoonjeon_workspace/VideoPrefill/logs/LVB/qwen2-5-vl_fps0.5.json",
        "/home/server08/yoonjeon_workspace/VideoPrefill/logs/LVB/qwen2-5-vl_fps0.5_homer_chunklen_3200.json"
        )