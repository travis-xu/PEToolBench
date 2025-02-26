GPU_ID=5
MODEL_NAME="/data/models/Meta-Llama-3-8B-Instruct"    # gpt-4o, gpt-4o-mini, Meta-Llama-3-8B-Instruct
# HISTORY_TYPE="p"

CUDA_VISIBLE_DEVICES=$GPU_ID llamafactory-cli train dataset_sft/llama3_lora_sft.yaml 

# CUDA_VISIBLE_DEVICES=$GPU_ID llamafactory-cli train \
#     --stage sft \
#     --do_train \
#     --model_name_or_path $MODEL_NAME \
#     --dataset user_entries_train_p,user_entries_train_r,user_entries_train_c \
#     --dataset_dir dataset_train \
#     --template llama3 \
#     --finetuning_type lora \
#     --output_dir saves/LLaMA3-8B/sft \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 4096 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --logging_steps 50 \
#     --warmup_steps 20 \
#     --save_steps 50 \
#     --eval_steps 50 \
#     --evaluation_strategy steps \
#     --load_best_model_at_end \
#     --learning_rate 1.0e-4 \
#     --num_train_epochs 1.0 \
#     --max_samples 2000 \
#     --val_size 0.1 \
#     --plot_loss \
#     --bf16