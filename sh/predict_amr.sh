# replace xxx with real data paths; replace CHECKPOINTS with model checkpoints waiting for evaluation.
rm -rf ./predict_amr.log; CUDA_VISIBLE_DEVICES=0 python -u bin/predict_amrs.py \
    --datasets amr2.0/data/amrs/split/test/*.txt \
    --gold-path data/xxx/gold.amr.txt \
    --pred-path data/xxx/pred.amr.txt \
    --checkpoint CHECKPOINTS \
    --beam-size 5 \
    --batch-size 500 \
    --device cuda \
    --penman-linearization --use-pointer-tokens >./predict_amr.log 2>&1