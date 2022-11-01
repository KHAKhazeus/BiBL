# BiBL
This is the repo for BiBL (Bidirectional Bayesian Learning).

## Best Model Checkpoints

Checkpoints are available here: https://mega.nz/file/U2hn3ZJQ#2VFNjPOho7fXagWr0zX_8Rny7BIT7XtHYMSc5po0Pps

## Instructions for Reproducing

We recommend using conda for dependency control.

### Create conda ENV
```shell script
conda create -n BIBL python=3.6
conda activate BIBL
```

### Install spring-amr for general seq2seq formulation
```shell script
cd BiBL
pip install -r requirements.txt
pip install -e .
```

### Preparing datasets
Download AMR2.0 and AMR3.0 dataset from LDC for training.
Place them right under the project folders and rename AMR2.0 dataset folder into amr2.0. (Keep the inner structure unchanged.)

### Training
```shell script
nohup ./sh/train_amr.sh & # lambda_g=0 + lambda_r=0 Text2AMR
nohup ./sh/train_text.sh & # lambda_g=0 + lambda_r=0 AMR2Text
nohup ./sh/train_amr-g_1-r_0_5.sh & # lambda_g=1.0 + lambda_r=0.5 Text2AMR
nohup ./sh/train_text-g_0_15-r_0_5.sh & # lambda_g=0.15 + lambda_r=0.5 AMR2Text
```
Best Checkpoints will be saved in "runs" folder.

### Inference and Evaluation
#### Generate inference results for benchmark dataset
```shell script
nohup ./predict_amr.sh & # please modify the model path and data generation path
nohup ./predict_text.sh & # please modify the model path and data generation path
```
#### Evaluation for Text2AMR

First we need to setup BLINK wiki entity linker. Please follow the instructions in https://github.com/facebookresearch/BLINK to complete the setup under BiBL project folder.

Then, blinkify the Text2AMR inference results.
```shell script
nohup ./blinkify.sh & # please modify the BLINK model path
```

Finally, to have comparable Smatch scores with previous works, we need to use https://github.com/mdtux89/amr-evaluation for Smatch metric evaluation. Follow the instructions in amr-evaluation project to conduct the evaluation based on the blinkified results.

#### Evaluation for AMR2Text

First we need to setup jamr. Please follow the instructions in https://github.com/jflanigan/jamr to complete the setup under BiBL project folder.

```shell script
nohup ./jamr_tokenize.sh & # please modify the jamr tokenizer script path
# please modify the generated inference data path & BLEU + chrF++ metrics will be generated
nohup ./eval_bleu.sh &
# please modify the generated inference data path & METEOR metrics will be generated
nohup ./eval_meteor.sh &
```
