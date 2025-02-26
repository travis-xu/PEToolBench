GPU_ID=7
MODEL_NAME="llama3-lora_dpo"   
# llama3-lora_dpo, llama3-lora_sft-dpo 
# gpt-4o, gpt-4o-mini, Meta-Llama-3-8B-Instruct, llama3-lora, Qwen2.5-7B-Instruct, vicuna-7b-v1.5, Mistral-7B-Instruct-v0.3
HISTORY_TYPE="p"
LORA_ADAPTER_PATH="saves/llama3-8b/lora/dpo/checkpoint-50"

CUDA_VISIBLE_DEVICES=$GPU_ID python run_experiment.py \
    --model_name $MODEL_NAME \
    --history_type $HISTORY_TYPE \
    --lora_adapter_path $LORA_ADAPTER_PATH \

HISTORY_TYPE="r"
CUDA_VISIBLE_DEVICES=$GPU_ID python run_experiment.py \
    --model_name $MODEL_NAME \
    --history_type $HISTORY_TYPE \
    --lora_adapter_path $LORA_ADAPTER_PATH \

HISTORY_TYPE="c"
CUDA_VISIBLE_DEVICES=$GPU_ID python run_experiment.py \
    --model_name $MODEL_NAME \
    --history_type $HISTORY_TYPE \
    --lora_adapter_path $LORA_ADAPTER_PATH \

