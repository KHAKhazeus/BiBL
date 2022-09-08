# please ensure the data folder paths in config files are modified and corresponding folders are created
rm -rf ./train_amr_base.log; CUDA_VISIBLE_DEVICES=0 python -u bin/train.py --config configs/config.yaml --direction amr > train_amr_base.log 2>&1