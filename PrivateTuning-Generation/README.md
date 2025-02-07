# Private Tuning for Generation Tasks
This code directory inlcudes code to run the private tuning methods PrivateLoRA, DP-FineTune, and PromptDPSGDGen for the generation tasks SAMSum, PFL-DocVQA, and MIT Movies trivia10k13.

## Install Dependencies
All dependencies are listed in the `requirements.txt` file.
Install the dependencies using:
```
pip install -r requirements.txt
```

Additionally, the https://github.com/microsoft/dp-transformers repository has to be installed. Therefore, download the repository, and within the repo execute:
```
pip install -e .
```

### Datasets:
We use 3 different dataset.
**SAMSum** is directly downloaded from Hugging Face using their dataset card https://huggingface.co/datasets/Samsung/samsum
**PFL-DocVQA** can be downloaded from https://benchmarks.elsa-ai.eu/?ch=2&com=downloads. 
**MIT Movies trivia10k13** can be downloaded from https://github.com/tonyzhaozh/few-shot-learning/tree/main/data/slot-movies.

## Running Code
### Start Training
The training is started using the `run.py`script.
An example execution for SAMSum and PrivateLoRA can be sene below.
```
python3 run.py \
--model_name_or_path "EleutherAI/pythia-1b"\
--task_name "samsum" \
--dataset_name "samsum" \
--max_seq_length 650 \
--do_train \
--do_eval \
--logging_step 10 \
--per_device_train_batch_size 16 \
--learning_rate 8e-4 \
--num_train_epochs 20 \
--output_dir "./checkpoints/SAMSum/pythia_1b_eps8DP_LoRA" \
--overwrite_output_dir \
--seed 11 \
--save_strategy no \
--disable_dp False \
--prefix False \
--lora True \
--gradient_accumulation_steps 16 \
--target_epsilon 8 \
--remove_unused_columns False \
--per_sample_max_grad_norm 0.1 \
--weight_decay 0.01 \
--max_grad_norm 0 \
--label_names labels \
--evaluation_strategy steps \
--eval_steps 100 \
--loading_4_bit True \
```

To enable PromptDPSGDGen, set `--prefix True` and `--lora False`
To enable DP-FineTune, set both prefix and lora to `False`

### Model Evaluation

The model that resulted from the above training can be evaluated with:
```
python SAMSum_eval.py --init_checkpoint "./checkpoints/SAMSum/pythia_1b_eps8DP_LoRA"
```

Additionally, `DocVQA_eval.py` is to evaluate models trained on PFL-DocVQA and `MITMovie_eval.py` is used to evaluate models trained on MIT-D or MIT-G.

## Acknowledgments
This code originally based on the code from https://github.com/THUDM/P-tuning-v2.
