# please ensure the data folder paths in config files are modified and corresponding folders are created
rm -rf ./train_text-g_0_15-r_0_5.log; CUDA_VISIBLE_DEVICES=0 python -u bin/train.py --config configs/config-g_0_15-r_0_5.yaml --direction text > ./train_text-g_0_15-r_0_5.log 2>&1