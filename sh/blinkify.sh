# please replace xxx with real data paths and ensure BLINK is cloned into the project path (setup correctly)
rm -rf blinkify.log; CUDA_VISIBLE_DEVICES=0 python -u bin/blinkify.py \
    --datasets data/xxx/pred.amr.txt \
    --out data/xxx/pred.amr.blinkified.txt \
    --device cuda \
    --blink-models-dir BLINK/models/ >./blinkify.log 2>&1