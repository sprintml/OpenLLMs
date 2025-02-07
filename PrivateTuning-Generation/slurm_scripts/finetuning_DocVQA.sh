bs=1
lr=2e-4
epoch=1
max_seq_len=1500
seed=11

# openlm-research/open_llama_13b
# lmsys/vicuna-7b-v1.5
# EleutherAI/pythia-6.9b
# EleutherAI/pythia-1b
# EleutherAI/pythia-70m
# EleutherAI/pythia-410m
python3 run.py \
  --model_name_or_path "openlm-research/open_llama_13b" \
  --task_name "docvqa" \
  --dataset_name "docvqa" \
  --max_seq_length $max_seq_len \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --output_dir "./checkpoints/DocVQA/pythia_1b_noDP_LoRA" \
  --overwrite_output_dir \
  --seed $seed \
  --save_strategy no \
  --disable_dp True \
  --prefix False \
  --lora True \
  --gradient_accumulation_steps 10 \
  --logging_step 20 \
  --loading_4_bit True \
  --constant_scheduler False
