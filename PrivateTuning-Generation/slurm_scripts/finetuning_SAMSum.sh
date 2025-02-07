bs=1
lr=2e-4
epoch=2
max_seq_len=650
seed=11

# openlm-research/open_llama_13b
# lmsys/vicuna-7b-v1.5
# EleutherAI/pythia-6.9b
# EleutherAI/pythia-1b
# EleutherAI/pythia-70m
# EleutherAI/pythia-410m
python3 run.py \
  --model_name_or_path "EleutherAI/pythia-1b" \
  --task_name "samsum" \
  --dataset_name "samsum" \
  --max_seq_length $max_seq_len \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --output_dir "./checkpoints/SAMSum/pythia_1b_noDP_LoRA" \
  --overwrite_output_dir \
  --seed $seed \
  --save_strategy no \
  --disable_dp True \
  --prefix False \
  --lora True \
  --logging_step 20 \
  --loading_4_bit True \
  --gradient_accumulation_steps 10 \
  --constant_scheduler True
