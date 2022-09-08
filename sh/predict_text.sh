# replace xxx with real data paths; replace CHECKPOINTS with model checkpoints waiting for evaluation.
rm -rf ./predict_text.log; CUDA_VISIBLE_DEVICES=0 python -u bin/predict_sentences.py \
    --datasets amr2.0/data/amrs/split/test/*.txt \
    --gold-path data/xxx/gold.text.txt \
    --pred-path data/xxx/pred.text.txt \
    --checkpoint CHECKPOINTS \
    --beam-size 5 \
    --batch-size 500 \
    --device cuda \
    --penman-linearization --use-pointer-tokens > ./predict_text.log 2>&1;