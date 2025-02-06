TASK=videomme
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained="Qwen/Qwen2.5-VL-7B-Instruct",device_map="auto",max_num_frames=32 \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2.5_$TASK \
    --output_path ./logs/