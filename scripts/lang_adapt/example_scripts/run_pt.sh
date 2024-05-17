lr=1e-4
pretrained_model=meta-llama/Llama-2-13b-hf
irish_tokenizer_path=/home/administrator/irish_based_llm/output/tok_llama2_extend_10k_hf
dataset_dir=/home/administrator/irish_based_llm/data/final_parallel_then_mono_corpus.jsonl
trainer=custom
data_cache=cache
output_dir=output_model
per_device_train_batch_size=2
per_device_eval_batch_size=1
gradient_accumulation_steps=8

deepspeed_config_file=./scripts/lang_adapt/example_scripts/ds_zero3_no_offload.json

torchrun --nnodes 1 --nproc_per_node 6 --master_port 29502 ./scripts/lang_adapt/run_clm_pt_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${irish_tokenizer_path} \
    --trainer ${trainer} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed 24259 \
    --fp16 \
    --num_train_epochs 4 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 100 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size 4096 \
    --output_dir ${output_dir} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --overwrite_output_dir \
    --use_peft False
