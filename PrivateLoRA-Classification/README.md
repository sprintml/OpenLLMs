# PrivateLoRA for Text Classification Tasks

Privacy-preserving training and evaluation of Vicuna, Pythia, BERT and Llama based model architecture on 4 different text classification tasks.
 - SST2
 - Trec
 - MPQA
 - Disaster

### Privacy setting

This repo utilizes the  ```PrivacyEngine``` object from the ```opacus``` library. Please install recommended version of opacus : ```pip install opacus==1.4.1```

### LoRA adapter

This repo is using the LoRA adapter to train the different models. This is done via the ```peft``` library.

### How to use

 - The main running scripts are 'main_vicuna.py', 'main_roberta.py' and 'main_pythia.py'. Here is an example usage for the sst2 task:
 ```python main_vicuna.py \
    --model_checkpoint 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --task 'sst2' \
    --batch_size 1024 \
    --lr 0.001 \
    --epochs 10 \
    --accumulation_steps 1 \
    --privacy True \
    --num_labels 2 \
    --save_dir '' \
    --testing_split 'validation' \
    --target_epsilon 8.0 \
    --target_delta 1e-5 \
    --lr_scheduler 'constant' \
    --seed 0
 ```

