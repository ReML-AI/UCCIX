python -m scripts.write_out \
    --output_base_path output/irish_cloze \
    --tasks irish_cloze \
    --sets test

lm_eval --model hf \
    --model_args pretrained=/home/administrator/irish_based_llm/final_model_output/correctly_saved_llama2-13b-irish-lora-2epoch \
    --tasks irish_cloze \
    --device cuda:0

lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-13b-hf \
    --tasks irish_cloze \
    --device cuda:0


python -m scripts.write_out \
    --output_base_path output/irish_qa \
    --tasks irish_qa \
    --sets validation

lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-13b-hf \
    --tasks irish_qa \
    --num_fewshot 5 \
    --output_path output/irish_qa/Llama-2-13b-hf \
    --log_samples \
    --device cuda:0

lm_eval --model openai-chat-completions \
    --model_args model=gpt-3.5-turbo-0125 \
    --tasks irish_qa \
    --num_fewshot 5 \
    --output_path temp \
    --log_samples

lm_eval --model openai-chat-completions \
    --model_args model=gpt-3.5-turbo-0125 \
    --tasks gaHealth \
    --num_fewshot 5 \
    --output_path output/temp \
    --log_samples

# IRISH CLOZE
for m in /home/administrator/irish_based_llm/final_model_output/correctly_saved_llama2-13b-irish-ia3-2epoch; do lm_eval --model hf --model_args pretrained=${m} --tasks irish_cloze --num_fewshot 0 --log_samples --device cuda:0 --output_path output/temp --wandb_args project=irish_llm_evaluation; done

# IRISH QA
for m in meta-llama/Llama-2-13b-hf /home/administrator/irish_based_llm/final_model_output/correctly_saved_llama2-13b-irish-ia3-1epoch /home/administrator/irish_based_llm/final_model_output/correctly_saved_llama2-13b-irish-ia3-2epoch /home/administrator/irish_based_llm/final_model_output/correctly_saved_llama2-13b-irish-ia3-3epoch /home/administrator/irish_based_llm/final_model_output/correctly_saved_llama2-13b-irish-lora-1epoch /home/administrator/irish_based_llm/final_model_output/correctly_saved_llama2-13b-irish-lora-2epoch /home/administrator/irish_based_llm/final_model_output/correctly_saved_llama2-13b-irish-lora-3epoch /home/administrator/irish_based_llm/final_model_output/correctly_saved_llama2-13b-irish-lora-4epoch; do for n in 3 5; do for t in irish_qa_context; do lm_eval --model hf --model_args pretrained=${m},dtype="float16" --tasks ${t} --num_fewshot ${n} --output_path output/temp --log_samples --device cuda:0 --wandb_args project=irish_llm_evaluation; done; done; done

lm_eval --model openai-chat-completions --model_args model=gpt-3.5-turbo-0125 --tasks irish_qas_context --num_fewshot 3 --output_path output/temp --log_samples --wandb_args project=irish_llm_evaluation; lm_eval --model openai-chat-completions --model_args model=gpt-3.5-turbo-0125 --tasks irish_qas_context --num_fewshot 5 --output_path temp --log_samples --wandb_args project=irish_llm_evaluation; 

# NQ OPEN
for m in /home/administrator/irish_based_llm/final_model_output/correctly_saved_llama2-13b-irish-ia3-2epoch; do for n in 3 5; do lm_eval --model hf --model_args pretrained=${m} --tasks nq_open --num_fewshot ${n} --output_path output/temp --log_samples --device cuda:0 --wandb_args project=irish_llm_evaluation; done; done

# GA HEALTH
for m in /home/administrator/irish_based_llm/final_model_output/correctly_saved_llama2-13b-irish-ia3-2epoch; do for n in 3 5; do lm_eval --model hf --model_args pretrained=${m},dtype="float16" --tasks gaHealth --num_fewshot ${n} --output_path output/temp --log_samples --device cuda:0 --wandb_args project=irish_llm_evaluation; done; done

# WINOGRANDE:
for m in home/administrator/irish_based_llm/final_model_output/correctly_saved_llama2-13b-irish-ia3-2epoch; do lm_eval --model hf --model_args pretrained=${m},dtype="float16" --tasks winogrande --output_path output/temp --log_samples --device cuda:0 --wandb_args project=irish_llm_evaluation --num_fewshot 5; done

lm_eval --model openai-chat-completions --model_args model=gpt-3.5-turbo-0125 --tasks winogrande --output_path output/temp --log_samples --device cuda:0 --wandb_args project=irish_llm_evaluation --num_fewshot 5; done

# SIB200
for m in /home/administrator/irish_based_llm/final_model_output/correctly_saved_llama2-13b-irish-ia3-2epoch; do for n in 10; do lm_eval --model hf --model_args pretrained=${m},dtype="float16" --tasks irish_sib200_en_prompt --output_path output/temp --log_samples --device cuda:0 --wandb_args project=irish_llm_evaluation --num_fewshot ${n}; done; done

# NOTE: run all for a new model
m=/home/administrator/irish_based_llm/final_model_output/correctly_saved_llama2-13b-irish-v2-lora-2epoch;
lm_eval --model hf --model_args pretrained=${m},dtype="float16" --tasks irish_sib200 --output_path output/temp --log_samples --device cuda:0 --wandb_args project=irish_llm_evaluation --num_fewshot 10;
lm_eval --model hf --model_args pretrained=${m},dtype="float16" --tasks irish_cloze --num_fewshot 0 --log_samples --device cuda:0 --output_path output/temp --wandb_args project=irish_llm_evaluation;
for n in 3 5; do lm_eval --model hf --model_args pretrained=${m},dtype="float16" --tasks irish_qas_context --num_fewshot ${n} --output_path output/temp --log_samples --device cuda:0 --wandb_args project=irish_llm_evaluation; done;
for n in 3 5; do lm_eval --model hf --model_args pretrained=${m},dtype="float16" --tasks gaHealth --num_fewshot ${n} --output_path output/temp --log_samples --device cuda:0 --wandb_args project=irish_llm_evaluation; done;
for n in 3 5; do lm_eval --model hf --model_args pretrained=${m},dtype="float16" --tasks nq_open --num_fewshot ${n} --output_path output/temp --log_samples --device cuda:0 --wandb_args project=irish_llm_evaluation; done;

m=/home/administrator/irish_based_llm/final_model_output/correctly_saved_llama2-13b-irish-v2-lora-2epoch;
lm_eval --model hf --model_args pretrained=${m},dtype="float16" --tasks winogrande --output_path output/temp --log_samples --device cuda:0 --wandb_args project=irish_llm_evaluation --num_fewshot 5;
lm_eval --model hf --model_args pretrained=${m},dtype="float16" --tasks hellaswag --output_path output/temp --log_samples --device cuda:0 --wandb_args project=irish_llm_evaluation --num_fewshot 10;
