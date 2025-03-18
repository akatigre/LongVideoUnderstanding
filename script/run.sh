#!/usr/bin/env bash

TOPK=4  # Adjust as needed

# 1) Generate lines of space-separated ratios, e.g. "0.1 0.3 0.1 0.5"
python -c "
import itertools

values = [0.1, 0.3, 0.5]
for combo in itertools.product(values, repeat=$TOPK):
    print(' '.join(map(str, combo)))
" \
| while read -r ratio_line
do
  # ratio_line now contains something like "0.1 0.1 0.3 0.5" 
  python3 run_qwen2.py --prune_ratio $ratio_line --topk $TOPK
done