export EXPERT=$1
export MODEL_SIZE=$2
export BATCH_SIZE=$3
export BASE_DIR=/workspace
export WANDB_API_KEY=[redacted]
export WANDB_PROJECT=airoboros-lmoe-$MODEL_SIZE-2.1-$EXPERT

accelerate launch qlora/qlora.py \
    --model_name_or_path $BASE_DIR/llama-2-$MODEL_SIZE-hf \
    --output_dir $BASE_DIR/$WANDB_PROJECT \
    --num_train_epochs 5 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --data_seed 11422 \
    --evaluation_strategy no \
    --eval_dataset_size 2 \
    --max_new_tokens 4096 \
    --dataloader_num_workers 3 \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --bf16 \
    --bits 4 \
    --double_quant \
    --quant_type nf4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --dataset airoboros-lmoe-2.1/expert_$EXPERT.jsonl \
    --dataset_format airoboros \
    --model_max_len 4096 \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 11422 \
    --report_to wandb \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False
