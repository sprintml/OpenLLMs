bs=16
lr=1e-3
epoch=20
max_seq_len=650
epsilon=8
seed=11

# openlm-research/open_llama_13b
# lmsys/vicuna-7b-v1.5
# EleutherAI/pythia-6.9b
# EleutherAI/pythia-1b
# EleutherAI/pythia-70m
# EleutherAI/pythia-410m
python3 run.py \
  --model_name_or_path "EleutherAI/pythia-1b"\
  --task_name "samsum" \
  --dataset_name "samsum" \
  --max_seq_length $max_seq_len \
  --do_train \
  --do_eval \
  --logging_step 10 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --output_dir "./checkpoints/SAMSum/pythia_1b_eps8DP_LoRA" \
  --overwrite_output_dir \
  --seed $seed \
  --save_strategy no \
  --disable_dp False \
  --prefix False \
  --lora True \
  --gradient_accumulation_steps 16 \
  --target_epsilon $epsilon \
  --remove_unused_columns False \
  --per_sample_max_grad_norm 0.1 \
  --weight_decay 0.01 \
  --max_grad_norm 0 \
  --label_names labels \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --loading_4_bit True \
