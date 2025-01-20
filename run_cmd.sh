for ATTN in dense flexprefill minference;
do
    for MODEL in longVA;
    do
        for i in 64 256 512;
        do
            CUDA_VISIBLE_DEVICES=1 python3 lvb_eval.py --attn-type $ATTN --model-name $MODEL --num-frames $i --device cuda
        done
    done
done